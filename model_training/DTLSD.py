# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import mlflow

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, default="/mnt/c/Users/david.chaparro/Documents/Repos/DeformableLETR/LibraryWorkspace/DT_LSD/config/DTLSD/DTLSD_4scale_swin.py")
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    parser.add_argument('--benchmark', action='store_true',
                        help="Train segmentation head if the flag is provided")
    #parser.add_argument('--batch_size', default=2, type=int)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='data/wireframe_processed')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    parser.add_argument('--dataset', default='train', type=str, choices=('train', 'val'))
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--no_opt', action='store_true')
    parser.add_argument('--append_word', default=None, type=str, help="Name of the convolutional backbone to use")

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(args):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("LETR_Star_Detection")
    mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_params(vars(args))


        utils.init_distributed_mode(args)
        # load cfg file and update the args
        #print("Loading config file from {}".format(args.config_file))
        time.sleep(args.rank * 0.02)
        cfg = SLConfig.fromfile(args.config_file)
        if args.options is not None:
            cfg.merge_from_dict(args.options)
        if args.rank == 0:
            save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
            cfg.dump(save_cfg_path)
            save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
            with open(save_json_path, 'w') as f:
                json.dump(vars(args), f, indent=2)
        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k,v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                raise ValueError("Key {} can used by args only".format(k))

        # update some new args temporally
        if not getattr(args, 'use_ema', None):
            args.use_ema = False
        if not getattr(args, 'debug', None):
            args.debug = False

        # setup logger
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
        #logger.info("git:\n  {}\n".format(utils.get_sha()))
        logger.info("Command: "+' '.join(sys.argv))
        if args.rank == 0:
            save_json_path = os.path.join(args.output_dir, "config_args_all.json")
            with open(save_json_path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            logger.info("Full config saved to {}".format(save_json_path))
        logger.info('world size: {}'.format(args.world_size))
        logger.info('rank: {}'.format(args.rank))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info("args: " + str(args) + '\n')


        if args.frozen_weights is not None:
            assert args.masks, "Frozen training is meant for segmentation only"
        print(args)

        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # build model
        model, criterion, postprocessors = build_model_main(args)
        wo_class_error = False
        model.to(device)

        # ema
        if args.use_ema:
            ema_m = ModelEma(model, args.ema_decay)
        else:
            ema_m = None

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_n_parameters = sum(p.numel() for p in model.parameters())
        logger.info('number of params:'+str(n_parameters))
        logger.info('total number of params:'+str(total_n_parameters))
        #logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

        param_dicts = get_param_dict(args, model_without_ddp)

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                    weight_decay=args.weight_decay)

        if args.eval or args.test:
            dataset_val = build_dataset(image_set=args.dataset, args=args)

            if args.distributed:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        else:
            dataset_train = build_dataset(image_set='train', args=args)
            dataset_val = build_dataset(image_set='val', args=args)
            if args.distributed:
                sampler_train = DistributedSampler(dataset_train)
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)

            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers)
            data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


        base_ds = get_coco_api_from_dataset(dataset_val)

        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            model_without_ddp.detr.load_state_dict(checkpoint['model'])

        output_dir = Path(args.output_dir)
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
            args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        if args.resume:
            if args.pretrain:
                checkpoint = torch.load(args.resume, map_location='cpu')
                new_state_dict = {}
                for k in checkpoint['model']:
                    if ("class_embed" in k):
                        continue
                    if  ("label_enc" in k):
                        continue
                    new_state_dict[k] = checkpoint['model'][k]

                # Compare load model and current model
                current_param = [n for n,p in model_without_ddp.named_parameters()]
                current_buffer = [n for n,p in model_without_ddp.named_buffers()]
                load_param = new_state_dict.keys()
                for p in load_param:
                    if p not in current_param and p not in current_buffer:
                        print(p, 'NOT appear in current model.  ')
                for p in current_param:
                    if p not in load_param:
                        print(p, 'NEW parameter.  ')
                model_without_ddp.load_state_dict(new_state_dict, strict=False)
                print()
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
            if args.use_ema:
                if 'ema_model' in checkpoint:
                    ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
                else:
                    del ema_m
                    ema_m = ModelEma(model, args.ema_decay)

            if not args.no_opt and not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                import copy
                p_groups = copy.deepcopy(optimizer.param_groups)
                optimizer.load_state_dict(checkpoint['optimizer'])
                for pg, pg_old in zip(optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']
                #print(optimizer.param_groups)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                args.override_resumed_lr_drop = True
                if args.override_resumed_lr_drop:
                    print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                lr_scheduler.step(lr_scheduler.last_epoch)
                args.start_epoch = checkpoint['epoch'] + 1

        if (not args.resume) and args.pretrain_model_path:
            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
            from collections import OrderedDict
            _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
            ignorelist = []

            def check_keep(keyname, ignorekeywordlist):
                for keyword in ignorekeywordlist:
                    if keyword in keyname:
                        ignorelist.append(keyname)
                        return False
                return True

            logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
            _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

            _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
            logger.info(str(_load_output))

            if args.use_ema:
                if 'ema_model' in checkpoint:
                    ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
                else:
                    del ema_m
                    ema_m = ModelEma(model, args.ema_decay)        

        if args.eval:
            os.environ['EVAL_FLAG'] = 'TRUE'
            test_stats = evaluate(model, criterion, postprocessors,
                                                data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
            return

        if args.test:

            test_stats = test(model, criterion, postprocessors, 
                                                data_loader_val, base_ds, device,args. output_dir, wo_class_error=wo_class_error, args=args)
            return

        print("Start training")
        start_time = time.time()
        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']

            if not args.onecyclelr:
                lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': ema_m.module.state_dict(),
                        })
                    utils.save_on_master(weights, checkpoint_path)
                    
            # eval
            test_stats = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }
            mlflow.log_metrics(train_stats, epoch) 
            mlflow.log_metrics(test_stats, epoch) 

            # eval ema
            if args.use_ema:
                ema_test_stats = evaluate(
                    ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
                )
                log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
                
            log_stats.update(best_map_holder.summary())

            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # remove the copied files.
        copyfilelist = vars(args).get('copyfilelist')
        if copyfilelist and args.local_rank == 0:
            from datasets.data_util import remove
            for filename in copyfilelist:
                print("Removing: {}".format(filename))
                remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DTLSD training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    
    args.config_filec=True
    args.benchmark=True
    #args.batch_size=2, type=int)

    # dataset parameters
    args.remove_difficult=True
    args.fix_size=True

    # training parameters
    args.note=''
    args.device='cuda'
    args.seed=42
    args.resume=False
    args.pretrain_model_path=False
    args.start_epoch=0
    args.eval=False
    args.test=False
    args.num_workers=10
    args.debug=False
    args.find_unused_params=True

    args.save_results=True
    args.save_log=True

    args.dataset='train'
    args.pretrain=True
    args.no_opt=True
    args.append_word=None

    args.dataset_file='coco'
    args.coco_path='/mnt/c/Users/david.chaparro/Documents/data/ZScaledRME03AllStar'

    args.output_dir='/mnt/c/Users/david.chaparro/Documents/Repos/DeformableLETR/checkpoint_models/DTLSD'
    args.eval=False

    main(args)
