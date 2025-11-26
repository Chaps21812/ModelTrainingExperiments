import zipfile
import random
import os
import json
import shutil
import datetime as dt
import numpy as np
import copy
import random
from .raw_datset import raw_dataset, PickleSerializable
from tqdm import tqdm
from astropy.io import fits
import imageio.v3 as iio
from PIL import Image
from .collect_stats import collect_stats, collect_satsim_stats
from .image_chipping import generate_training_crops
from .constants import SPACECRAFT
from .preprocess_functions import raw_file

def merge_categories(category_list:list):
    name_to_category = {}
    
    for cat in category_list:
        name = cat['name']
        if name not in name_to_category:
            name_to_category[name] = {
                "id": cat["id"],
                "name": cat['name'],
                "supercategory": cat.get("supercategory", "")
            }

    return list(name_to_category.values())

def compress_data(data_paths:list[str], output_zip:str) -> None:
    """
    Zips all images in the image path into a zipfile in the output path. Compresses each file one by one to reduce overhead memory requirements 

    Args:
        input_data (list[str]): Input list of files to compress
        stars (str): Output ZIP path, must end in abc.zip

    Not used too much anymore, if you would like to use it, use
    if zip:
        compress_data(path_to_annotation.keys(), os.path.join(new_fits_path,"compressed_fits.zip"))


    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for image_path in tqdm(data_paths, desc="Compressing images", total=len(data_paths)):
            zipf.write(image_path, arcname=os.path.basename(image_path))  # Save only filename

def split_files(file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> tuple:
    """
    Generates a train test split given a list of files. 

    Args:
        input_data (list[str]): Input list of files to shuffle and generate a train test split
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """

    # Ensure the ratios add up to 1
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("The sum of train, val, and test ratios must be 1.")
    
    # Shuffle the file list to ensure random distribution
    random.shuffle(file_list)
    
    # Calculate the split sizes
    total_files = len(file_list)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size  # The remainder will be the test set

    # Split the files
    train_files = file_list[:train_size]
    val_files = file_list[train_size:train_size + val_size]
    test_files = file_list[train_size + val_size:]

    return train_files, val_files, test_files

def split_collections(collection_ids:dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> tuple:
    """
    Generates a train test split given a dictionary with keys being collect ids connected with a list of paths to each file

    Args:
        collection_ids (dict[str, list[str]]): Dict of collect_id with a list of fits file paths associated with the collect
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """

    train_files = {}
    val_files = {}
    test_files = {}
    train_len = 0
    val_len= 0
    test_len= 0

    # Ensure the ratios add up to 1
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("The sum of train, val, and test ratios must be 1.")
    
    total_images = sum([len(value) for key,value in collection_ids.items()])
    all_collects = list(collection_ids.keys())
    random.shuffle(all_collects)
    
    # Calculate the split sizes
    total_files = total_images
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size  # The remainder will be the test set

    for key in all_collects:
        if train_len < train_size:
            train_files[key] = collection_ids[key]
            train_len += len(collection_ids[key])
            continue
        if val_len < val_size:
            val_files[key] = collection_ids[key]
            val_len += len(collection_ids[key])
            continue
        if test_len < test_size:
            test_files[key] = collection_ids[key]
            test_len += len(collection_ids[key])
            continue

    return train_files, val_files, test_files

def split_attribute(collection_ids:dict, attribute_values:dict, num_splits:int) -> tuple:
    """
    Generates an ordered split of given a dictionary with keys being collect ids ad a function of an attribute in the dataset. 

    Args:
        collection_ids (dict[str, list[str]]): Dict of collect_id with a list of fits file paths associated with the collect
        attribute_values (dict[str, list[float]]): Dict of collect_id with a list of numeric attributes from each image in the collect
        num_splits (int): Number of partitions to divide your data into

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """

    partitions = [[] for i in range(num_splits)]
    avg_values = {}
    fraction = 1/(num_splits)
    bruh = {}

    for key,tlist in attribute_values.items():
        bruh[key] = len(tlist)
        if isinstance(tlist[0], list):
            bruh_list = []
            for l in tlist:
                bruh_list.extend(l)
            avg_values[key] = np.average(bruh_list)
        else:
            avg_values[key] = np.average(tlist)
        if key not in collection_ids:
            del avg_values[key]
            continue

    sorted_items = {k: v for k, v in sorted(avg_values.items(), key=lambda item: item[1])}
    
    total_images = sum([len(value) for key,value in collection_ids.items()])
    
    num_images = total_images*fraction

    partition_num=0
    for key in sorted_items:
        brhu =  avg_values[key]
        if len(partitions[partition_num]) < num_images:
            partitions[partition_num].extend(collection_ids[key])
        else:
            partition_num +=1
            partitions[partition_num].extend(collection_ids[key])

    return partitions

def merge_coco(coco_directories:list[str], destination_path:str, notes:str="", train_test_split:bool=False, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10, zip:bool=False):
    """
    Generates a coco dataset from a list of coco datasets and compiles them into one destination. Generates a train test split based on collect.

    Args:
        coco_directories (list[str]): List of paths of coco datasets. Raw datasets must be converted to coco before merging
        destination_path (str): Destination of the merged dataset
        notes (str): Notes to add onto the dataset annotation file for future use
        train_test_split (bool): Bool to create a train test split or not
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples
        zip (bool): Number of partitions to divide your data into

    Returns:
        None
    """
    path_to_image = []
    images_queue = []
    annotations_queue = []
    notes_queue = []
    catagories_queue = []
    collections_queue = {}
    id_to_index = {}
    multiframe_annotations_dictionary = {}
    for directory in tqdm(coco_directories, desc="Processing COCO Datasets"):
        anotations_file = os.path.join(directory, "annotations", "annotations.json")
        with open(anotations_file, 'r') as f:
            annotations = json.load(f)
        multiframe_annotation_file = os.path.join(directory, "multiframe_annotations", "multiframe_annotations.json")
        with open(multiframe_annotation_file, 'r') as f:
            multiframe_annotations = json.load(f)
        images_queue.extend(annotations["images"])
        annotations_queue.extend(annotations["annotations"])
        notes_queue.append({"info":annotations["info"], "notes":annotations["notes"], "directory": directory })
        multiframe_annotations_dictionary = multiframe_annotations_dictionary | multiframe_annotations
    
        for index, item in enumerate(annotations["images"]):
            if item['collect_id'] not in collections_queue:
                collections_queue[item['collect_id']] = []
                collections_queue[item['collect_id']].append(os.path.join(directory,item['file_name']))
            else:
                collections_queue[item['collect_id']].append(os.path.join(directory,item['file_name']))

        catagories_queue.extend(annotations["categories"]) #try and find a smarter way of dealing with categories pls
        templist = [os.path.join(directory, "images", image) for image in os.listdir(os.path.join(directory, "images"))]
        path_to_image.extend(templist)
    annotations_id_to_index = [[] for i in range(len(images_queue))]
    for index, item in enumerate(images_queue):
        id_to_index[item["id"]] = index
    for index,anot in enumerate(annotations_queue):
        annotations_id_to_index[id_to_index[anot["image_id"]]].append(index)
    merged_catagories = merge_categories(catagories_queue)

    filetype="fits"
    if ".png" in path_to_image[0]:
        filetype="png"
    
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": "Satellite detection of calsat dataset. ",
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": "Satellite BBox",
        "samples":len(path_to_image),
        "prev_notes":notes_queue}
    
    if train_test_split:
        titles = ["train","val","test"]
        file_list_split = split_collections(collections_queue, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        for j, collection_dictionary in enumerate(file_list_split):
            temp_list = [x for lst in collection_dictionary.values() for x in lst]
            indicies = [id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))] for single_path in temp_list]
            temp_anot_index = [annotations_id_to_index[id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))]] for single_path in temp_list]
            new_multiframe_annotations = {cid:multiframe_annotations_dictionary[cid] for cid in collection_dictionary.keys()}

            ### For one large dataset
            # Save annotation data to corresponding train test json
            #Make new data directory
            data_folder = destination_path
            subset_folder = os.path.join(data_folder, titles[j])
            images_alias = os.path.join(data_folder, titles[j], "images")
            annotations_alias = os.path.join(data_folder, titles[j], "annotations")
            multiframe_annotations_alias = os.path.join(data_folder, titles[j], "multiframe_annotations")
            os.makedirs(data_folder, exist_ok=True)
            os.makedirs(subset_folder, exist_ok=True)
            os.makedirs(images_alias, exist_ok=True)
            os.makedirs(annotations_alias, exist_ok=True)
            os.makedirs(multiframe_annotations_alias, exist_ok=True)

            info["train"]= train_ratio
            info["test"]= test_ratio
            info["val"]= val_ratio
            info["samples"]=len(temp_list)

            t_anot =  [[annotations_queue[j] for j in li ] for li in temp_anot_index]
            all_anot = []
            for t in t_anot:
                all_anot.extend(t)

            all_data = {
                "info": info,
                "images": [images_queue[i] for i in indicies],
                "annotations":all_anot,
                "categories": merged_catagories,
                "notes": notes}
            data_attributes_obj=json.dumps(all_data, indent=4)
            with open(os.path.join(annotations_alias, "annotations.json"), "w") as outfile:
                outfile.write(data_attributes_obj)
            with open(os.path.join(multiframe_annotations_alias,"multiframe_annotations.json"), 'w') as f:
                json.dump(new_multiframe_annotations, f)

            
            #Compressing images without copying them to folder to save on memory
            if zip:
                compress_data(temp_list, os.path.join(images_alias, "compressed_fits.zip"))
            else:
                for image in tqdm(temp_list, desc="Copying images", total=len(temp_list)):
                    shutil.copy(image, images_alias)
    else:
        ### For one large dataset
        # Save annotation data to corresponding train test json
        #Make new data directory
        data_folder = destination_path
        new_images_folder = os.path.join(data_folder, "images")
        new_annotations_folder = os.path.join(data_folder, "annotations")
        multiframe_annotations_alias = os.path.join(data_folder, "multiframe_annotations")
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(new_images_folder, exist_ok=True)
        os.makedirs(new_annotations_folder, exist_ok=True)
        os.makedirs(multiframe_annotations_alias, exist_ok=True)

        all_data = {
            "info": info,
            "images": images_queue,
            "annotations": annotations_queue,
            "categories": catagories_queue,
            "notes": notes}
        data_attributes_obj=json.dumps(all_data, indent=4)
        with open(os.path.join(new_annotations_folder, "annotations.json"), "w") as outfile:
            outfile.write(data_attributes_obj)
        with open(os.path.join(multiframe_annotations_alias,"multiframe_annotations.json"), 'w') as f:
            json.dump(multiframe_annotations_dictionary, f)

        #Compressing images without copying them to folder to save on memory
        if zip:
            compress_data(path_to_image, os.path.join(new_images_folder, "compressed_fits.zip"))
        else:
            for image in tqdm(path_to_image, desc="Copying images", total=len(path_to_image)):
                shutil.copy(image, new_images_folder)

def silt_to_coco(raw_dataset_path:str, include_sats:bool=True, include_stars:bool=False, convert_png:bool=True, process_func:list=None, notes:str=""):
    """
    Converts a SILT bucket annotation dataset into a coco dataset

    Args:
        raw_dataset_path (list[str]): Input list of files to shuffle and generate a train test split
        include_sats (bool): Ratio of training samples
        include_stars (bool): Ratio of validation samples
        zip (bool): Ratio of testing samples
        convert_png (bool): Convert to an intermediate PNG  
        process_func (src.preprocess_functions): Preprocess function to convert PNGs to
        notes (str)

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """
    assert isinstance(process_func, list)
    path_to_annotation = {}
    local_files =  raw_dataset(raw_dataset_path)

    ### For one large dataset
    #Make new data directory
    images_folder=os.path.join(raw_dataset_path, "images")
    annotations_folder=os.path.join(raw_dataset_path, "annotations")
    multiframe_annotations_folder=os.path.join(raw_dataset_path, "multiframe_annotations")
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        shutil.rmtree(images_folder)
        print(f"Deleted folder: {images_folder}")
    if os.path.exists(annotations_folder) and os.path.isdir(annotations_folder):
        shutil.rmtree(annotations_folder)
        print(f"Deleted folder: {annotations_folder}")
    if os.path.exists(multiframe_annotations_folder) and os.path.isdir(multiframe_annotations_folder):
        shutil.rmtree(multiframe_annotations_folder)
        print(f"Deleted folder: {multiframe_annotations_folder}")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)
    os.makedirs(multiframe_annotations_folder, exist_ok=True)

    filetype="fits"
    if convert_png:
        filetype="png"

    collect_dictionary:dict[str,list] = {}

    for annotation_path,fits_path in tqdm(local_files.annotation_to_fits.items(), desc="Converting Silt to COCO"):        
        with open(annotation_path, 'r') as f:
            json_data = json.load(f)
        hdu = fits.open(fits_path)

        sample_attributes, object_attributes = collect_stats(json_data, hdu, padding=20)

        hdul = hdu[0]
        header = hdul.header
        x_res = json_data["sensor"]["width"]
        y_res = json_data["sensor"]["height"]
        object_list = json_data["objects"]
        image_id= np.random.randint(0,9223372036854775806)
        annotations = []
        #Process all detected objects
        try:
            collection_id = json_data["image_set_id"]
            collection_start_time = json_data["exp_start_time"]
        except KeyError:
            collection_id = str(np.random.randint(0,9223372036854775806))
            collection_start_time = "2024-04-24T10:12:17.315000+00:00"
        try:
            label_type = object["datatype"]
        except:
            label_type = "Real"

        for object in object_list:


            if include_sats and object["class_name"] == "Satellite" and object["type"] == "line":
                continue
            elif include_sats and object["class_name"] == "Satellite": 
                #Create coco annotation for one image
                x1 = (object["x_center"]-object["bbox_width"]/2)*x_res
                y1 = (object["y_center"]-object["bbox_height"]/2)*y_res
                width = object["bbox_width"]*x_res
                height = object["bbox_height"]*y_res
                x_center = object["x_center"]*x_res
                y_center = object["y_center"]*y_res
                if abs(width*height) > 1000:
                    continue

                annotation = {
                    "id": np.random.randint(0,9223372036854775806),
                    "image_id": image_id,
                    "collect_id":collection_id,
                    "exp_start_time":collection_start_time,
                    "category_id": int(object["class_id"]),# Originallly class_id-1, not sure why
                    "category_name": object["class_name"],
                    "type": "bbox",
                    "centroid": [x_center,y_center],
                    "bbox": [x1,y1,width,height],
                    "area": abs(width*height),
                    "iso_flux": object["iso_flux"],
                    "exposure": header["EXPTIME"],
                    "iscrowd": 0,
                    "y_min": y_center-height/2,
                    "x_min": x_center-width/2,
                    "y_max": y_center+height/2,
                    "x_max": x_center+width/2,
                    "collect_id":collection_id,
                    "exp_start_time":collection_start_time,
                    "label_type":label_type
                    }

            if include_stars and object["class_name"] == "Star": 
                #Create coco annotation for one image
                dx1 =(object["x2"]-object["x1"])*x_res
                dy1 =(object["y2"]-object["y1"])*y_res
                deltax = abs((object["x2"]-object["x1"])*x_res)
                deltay = abs((object["y2"]-object["y1"])*y_res)
                minx = np.min([object["x2"], object["x1"]])*x_res
                miny = np.min([object["y2"], object["y1"]])*y_res
                x1 = object["x1"]*x_res
                y1 = object["y1"]*y_res
                x_center = object["x_center"]*x_res
                y_center = object["y_center"]*y_res


                annotation = {
                    "id": np.random.randint(0,9223372036854775806),
                    "image_id": image_id,
                    "collect_id":collection_id,
                    "exp_start_time":collection_start_time,
                    "category_id": int(object["class_id"]),# Originallly class_id-1, not sure why
                    "category_name": object["class_name"],
                    "type": "bbox",
                    "centroid": [x_center,y_center],
                    "bbox": [minx,miny,deltax,deltay],
                    "area": abs(deltax*deltay),
                    "line": [x1,y1,dx1,dy1],
                    "line_center": [x_center,y_center],
                    "iso_flux": object["iso_flux"],
                    "exposure": header["EXPTIME"],
                    "iscrowd": 0,
                    "collect_id":collection_id,
                    "exp_start_time":collection_start_time,
                    "label_type":label_type
                    }

            other_annotation_attributes = _find_dict(object["correlation_id"], object_attributes)
            annotation = _merge_dicts(annotation,other_annotation_attributes )
            annotations.append(annotation)

        image = {
            "id": image_id,
            "width": json_data["sensor"]["width"],
            "height": json_data["sensor"]["height"],
            "collect_id":collection_id,
            "exp_start_time":collection_start_time,
            "type":"siderial" if header["TELTKRA"] == 0.0 else "rate",
            "rate": header["TELTKRA"],
            "exposure_seconds":header["EXPTIME"],
            "gain":1,
            "lat":header["SITELAT"],
            "lon":header["CENTALT"],
            "file_name": os.path.join("images", f"{image_id}.{filetype}"),
            "original_path": fits_path,
            "date": header["DATE-OBS"],
            "label_type":label_type}
        
        image_collect_information = {
            "collect_id":collection_id,
            "exp_start_time":collection_start_time,
            "path":image["file_name"], 
            "image_id":image["id"]}
        is_part_of_collect = collection_id != "N/A"
        if is_part_of_collect:
            if collection_id not in collect_dictionary:
                collect_dictionary[collection_id] = []
                collect_dictionary[collection_id].append(image_collect_information)
            else:
                collect_dictionary[collection_id].append(image_collect_information)
        
        image = _merge_dicts(image, sample_attributes)

        #Add coco image to list of files
        path_to_annotation[fits_path] = {"annotation":annotations, "image":image, "new_id":image_id}
        
    #Compile final json information for folder
    category1 = {"id": 1,"name": "Satellite","supercategory": "Space Object",}
    category2 = {"id": 2,"name": "Star","supercategory": "Space Object",}
    categories = [category1, category2]
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": notes,
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S")}

    # Compiling information for annotation file
    images = []
    annotations = []
    for path in path_to_annotation.keys():
        images.append(path_to_annotation[path]["image"])
        annotations.extend(path_to_annotation[path]["annotation"])
        
    #Writing annotations to the json annotations file
    all_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "notes": notes}
    with open(os.path.join(annotations_folder, "annotations.json"), "w") as outfile:
        data_attributes_obj=json.dumps(all_data, indent=4)
        outfile.write(data_attributes_obj)

    #Compiling Multiframe annotation json
    final_collections_dictionary = {}
    for collect_id,collect_list in collect_dictionary.items():
        new_collect_info_order = []
        dates_list = []
        for sub_image_info in collect_list:
            dates_list.append(str(sub_image_info["exp_start_time"]))
        argsort_indices = sorted(range(len(dates_list)), key=lambda i: dt.datetime.fromisoformat(dates_list[i]))
        for index, arg_sorted_index in enumerate(argsort_indices):
            dict_copy = collect_list[arg_sorted_index]
            dict_copy["order"] = index
            new_collect_info_order.append(dict_copy)
        final_collections_dictionary[collect_id] = new_collect_info_order
    with open(os.path.join(multiframe_annotations_folder, "multiframe_annotations.json"), "w") as outfile:
        multiframe_annotations=json.dumps(final_collections_dictionary, indent=4)
        outfile.write(multiframe_annotations)

    for func in process_func:
        for image in tqdm(path_to_annotation.keys(), desc="Copying images", total=len(path_to_annotation.keys())):
            new_file_name = os.path.join(images_folder,str(path_to_annotation[image]["new_id"])+f".{filetype}")
            if convert_png:
                hdu = fits.open(image)
                hdul = hdu[0]
                data = hdul.data
                if data is None or data.size==0:
                    print(f"{new_file_name} failed to save")
                    continue
                if func is not None:
                    data = func(data)
                else: 
                    data = raw_file(data)
                data = np.transpose(data, (1,2,0))
                png = Image.fromarray(data)
                png.save(new_file_name)
            else:
                destination_path = shutil.copy(image, images_folder)
                shutil.move(destination_path,new_file_name)

def silt_to_coco_panoptic(silt_dataset_path:str, include_sats:bool=True, include_stars:bool=False, process_func=None, overlap:float=None, notes:str="", percent_empty_data:float=.33, new_image_size:int=512):
    """
    Converts a satasim generated dataset into a coco dataset

    Args:
        path (list[str]): Input list of files to shuffle and generate a train test split
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """
    path_to_annotation = {}
    local_files =  raw_dataset(silt_dataset_path)

    filetype="png"

    
    ### For one large dataset
    #Make new data directory
    images_folder=os.path.join(silt_dataset_path, "images")
    annotations_folder=os.path.join(silt_dataset_path, "annotations")
    multiframe_annotations_folder=os.path.join(silt_dataset_path, "multiframe_annotations")
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        shutil.rmtree(images_folder)
        print(f"Deleted folder: {images_folder}")
    if os.path.exists(annotations_folder) and os.path.isdir(annotations_folder):
        shutil.rmtree(annotations_folder)
        print(f"Deleted folder: {annotations_folder}")
    if os.path.exists(multiframe_annotations_folder) and os.path.isdir(multiframe_annotations_folder):
        shutil.rmtree(multiframe_annotations_folder)
        print(f"Deleted folder: {multiframe_annotations_folder}")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)
    os.makedirs(multiframe_annotations_folder, exist_ok=True)

    #Iterate for every image
    for annotation_path,fits_path in tqdm(local_files.annotation_to_fits.items(), desc="Converting Silt to COCO"):        
        #Loading the data
        with open(annotation_path, 'r') as f:
            json_data = json.load(f)
        hdu = fits.open(fits_path)
        hdu_copy = copy.deepcopy(hdu)

        #Creating copies and generating the crops necessary for splitting
        current_image = hdu[0].data
        random_overlap = 20*np.random.random()+10
        if overlap is None:
            overlap = random_overlap
        images, raw_images, jsons = generate_training_crops(current_image, json_data, process_func, shape_x=new_image_size, shape_y=new_image_size, overlap_x=overlap,overlap_y=overlap)

        #Calculate probabilities of drawing blank images
        images_with_target = 0
        total_num_images = len(jsons)
        for subjson in jsons:
            if len(subjson["objects"]) != 0:
                images_with_target += 1
        images_with_target = max(images_with_target,1)

        #For every cropped image
        for subimage, raw_subimage, subjson in zip(images,raw_images, jsons):
            hdu_copy[0].data = raw_subimage
            hdu_copy[0].header["NAXIS1"] = new_image_size
            hdu_copy[0].header["NAXIS2"] = new_image_size

            include = True if random.random() < percent_empty_data*images_with_target/(total_num_images-images_with_target) else False
            if len(subjson["objects"]) == 0 and not include:
                continue

            sample_attributes, object_attributes = collect_stats(subjson, hdu_copy, padding=20)

            hdul = hdu_copy[0]
            header = hdul.header
            x_res = subjson["sensor"]["width"]
            y_res = subjson["sensor"]["height"]
            object_list = subjson["objects"]
            image_id= np.random.randint(0,9223372036854775806)
            annotations = []
            #Process all detected objects
            collection_id = ""
            collection_start_time = ""
            label_type = ""
            try:
                collection_id = subjson["image_set_id"]
                collection_start_time = subjson["exp_start_time"]
            except KeyError:
                collection_id = "N/A"
                collection_start_time = "2024-04-24T10:12:17.315000+00:00"
            for object in object_list:
                try:
                    label_type = object["datatype"]
                except:
                    label_type = "real"

                if object["type"] == "line":
                    continue
                elif include_sats and object["class_name"] == "Satellite": 
                    #Create coco annotation for one image
                    x1 = (object["x_center"]-object["bbox_width"]/2)*x_res
                    y1 = (object["y_center"]-object["bbox_height"]/2)*y_res
                    width = object["bbox_width"]*x_res
                    height = object["bbox_height"]*y_res
                    x_center = object["x_center"]*x_res
                    y_center = object["y_center"]*y_res
                    if abs(width*height) > 1000:
                        continue

                    annotation = {
                        "id": np.random.randint(0,9223372036854775806),
                        "image_id": image_id,
                        "collect_id":collection_id,
                        "exp_start_time":collection_start_time,
                        "category_id": int(object["class_id"]),# Originallly class_id-1, not sure why
                        "category_name": object["class_name"],
                        "type": "bbox",
                        "centroid": [x_center,y_center],
                        "bbox": [x1,y1,width,height],
                        "area": abs(width*height),
                        "iso_flux": object["iso_flux"],
                        "exposure": header["EXPTIME"],
                        "iscrowd": 0,
                        "y_min": y_center-height/2,
                        "x_min": x_center-width/2,
                        "y_max": y_center+height/2,
                        "x_max": x_center+width/2,
                        "label_type":label_type
                        }

                if include_stars and object["class_name"] == "Star": 
                    #Create coco annotation for one image
                    dx1 =(object["x2"]-object["x1"])*x_res
                    dy1 =(object["y2"]-object["y1"])*y_res
                    deltax = abs((object["x2"]-object["x1"])*x_res)
                    deltay = abs((object["y2"]-object["y1"])*y_res)
                    minx = np.min([object["x2"], object["x1"]])*x_res
                    miny = np.min([object["y2"], object["y1"]])*y_res
                    x1 = object["x1"]*x_res
                    y1 = object["y1"]*y_res
                    x_center = object["x_center"]*x_res
                    y_center = object["y_center"]*y_res


                    annotation = {
                        "id": np.random.randint(0,9223372036854775806),
                        "image_id": image_id,
                        "collect_id":collection_id,
                        "exp_start_time":collection_start_time,
                        "category_id": int(object["class_id"]),# Originallly class_id-1, not sure why
                        "category_name": object["class_name"],
                        "type": "bbox",
                        "centroid": [x_center,y_center],
                        "bbox": [minx,miny,deltax,deltay],
                        "area": abs(deltax*deltay),
                        "line": [x1,y1,dx1,dy1],
                        "line_center": [x_center,y_center],
                        "iso_flux": object["iso_flux"],
                        "exposure": header["EXPTIME"],
                        "iscrowd": 0,
                        "label_type":label_type
                        }

                other_annotation_attributes = _find_dict(object["correlation_id"], object_attributes)
                annotation = _merge_dicts(annotation,other_annotation_attributes )
                annotations.append(annotation)

            new_file_name = os.path.join("images", f"{image_id}.{filetype}")
            image = {
                "id": image_id,
                "width": subjson["sensor"]["width"],
                "height": subjson["sensor"]["height"],
                "collect_id":collection_id,
                "exp_start_time":collection_start_time,
                "type":"siderial" if header["TELTKRA"] == 0.0 else "rate",
                "rate": header["TELTKRA"],
                "exposure_seconds":header["EXPTIME"],
                "gain":1,
                "lat":header["SITELAT"],
                "lat":header["SITELAT"],
                "lon":header["CENTALT"],
                "file_name": new_file_name,
                "original_path": fits_path,
                "date": header["DATE-OBS"],
                "label_type":label_type}
            
            image = _merge_dicts(image, sample_attributes)

            #Add coco image to list of files
            path_to_annotation[new_file_name] = {"annotation":annotations, "image":image, "new_id":image_id}

            new_file_name = os.path.join(images_folder,str(image_id)+f".{filetype}")
            data = hdu_copy[0].data

            if data is None or data.size==0:
                print(f"{new_file_name} failed to save")
                continue
            subimage = np.transpose(subimage, (1,2,0))
            png = Image.fromarray(subimage)
            png.save(new_file_name)


    #Compile final json information for folder
    category1 = {"id": 1,"name": "Satellite","supercategory": "Space Object",}
    category2 = {"id": 2,"name": "Star","supercategory": "Space Object",}
    categories = [category1, category2]
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": notes,
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S")}

    # Compiling information for annotation file
    images = []
    annotations = []
    for path in path_to_annotation.keys():
        images.append(path_to_annotation[path]["image"])
        annotations.extend(path_to_annotation[path]["annotation"])
        
    #Writing annotations to the json annotations file
    all_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "notes": notes}
    with open(os.path.join(annotations_folder, "annotations.json"), "w") as outfile:
        data_attributes_obj=json.dumps(all_data, indent=4)
        outfile.write(data_attributes_obj)

def satsim_to_coco(satsim_path:str, output_dataset:str, include_sats:bool=True, include_stars:bool=False, zip:bool=False, convert_png:bool=True, process_func=None, notes:str=""):
    """
    Converts a satasim generated dataset into a coco dataset

    Args:
        path (list[str]): Input list of files to shuffle and generate a train test split
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """
    path_to_annotation = {}
    satsim_paths = os.listdir(satsim_path)

    filetype="fits"
    if convert_png:
        filetype="png"


    for path in tqdm(satsim_paths, desc="Converting Satsim to COCO"):
        
        fits_path = os.path.join(satsim_path, path, "ImageFiles")
        annotations_path =  os.path.join(satsim_path, path, "Annotations")
        config_path = os.path.join(satsim_path, path, "config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        for frame_no, fits_name in enumerate(os.listdir(fits_path)):
            #Path and filename manipulation
            fits_file = os.path.join(fits_path, fits_name)
            json_file = os.path.join(annotations_path, fits_name.replace(".fits", ".json")) 
            image_num = float(fits_name.split(".")[1])
            with open(json_file, 'r') as f:
                #Load satellite annotation data
                json_data = json.load(f)
                object_list = json_data["data"]["objects"]
                data = json_data["data"]
                image_id= np.random.randint(0,9223372036854775806)
                annotations = []
                x_res = data["sensor"]["width"]
                y_res = data["sensor"]["height"]

                
                #Process all detected objects
                for object in object_list:
                    try:
                        label_type = object["datatype"]
                    except:
                        label_type = "simulated"
                    if include_sats and object["class_name"] == "Satellite": 
                        #Create coco annotation for one image
                        x1 = (object["x_center"]-object["bbox_width"]/2)*x_res
                        y1 = (object["y_center"]-object["bbox_height"]/2)*y_res
                        width = object["bbox_width"]*x_res
                        height = object["bbox_height"]*y_res
                        x_center = object["x_center"]*x_res
                        y_center = object["y_center"]*y_res

                        annotation = {
                            "id": np.random.randint(0,9223372036854775806),
                            "image_id": image_id,
                            "collect_id": path,
                            "exp_start_time":image_num,
                            "category_id": object["class_id"],# Originallly class_id-1, not sure why
                            "category_name": object["class_name"],
                            "type": "bbox",
                            "centroid": [x_center ,y_center ],
                            "bbox": [x1 ,y1 ,width ,height ],
                            "area": abs(width *height) ,
                            "iscrowd": 0,
                            "snr": object["snr"],
                            "y_min": object["y_min"]*y_res ,
                            "x_min": object["x_min"]*x_res ,
                            "y_max": object["y_max"]*y_res ,
                            "x_max": object["x_max"]*x_res ,
                            "source": object["source"],
                            "magnitude": object["magnitude"],
                            "label_type":label_type
                            }
                        annotations.append(annotation)

                    if include_stars and object["class_name"] == "Star": 
                        #Create coco annotation for one image
                        dx1 = (object["x_end"]-object["x_start"])*x_res
                        dy1 = (object["y_end"]-object["y_start"])*y_res
                        x1 = object["x_start"]*x_res
                        y1 = object["y_start"]*y_res
                        width = object["bbox_width"]*x_res
                        height = object["bbox_height"]*y_res
                        x_center = object["x_center"]*x_res
                        y_center = object["y_center"]*y_res

                        deltax = abs((object["x_end"]-object["x_start"])*x_res)
                        deltay = abs((object["y_end"]-object["y_start"])*y_res)
                        minx = np.min([object["x_end"], object["x_start"]])*x_res
                        miny = np.min([object["y_end"], object["y_start"]])*y_res

                        annotation = {
                            "id": np.random.randint(0,9223372036854775806),
                            "image_id": image_id,
                            "collect_id": path,
                            "exp_start_time":image_num,
                            "category_id": object["class_id"],# Originallly class_id-1, not sure why
                            "category_name": object["class_name"],
                            "type": "line",
                            "centroid": [x_center ,y_center ],
                            "bbox": [minx,miny ,deltax ,deltay ],
                            "area": abs(width *height) ,
                            "line": [x1,y1,dx1,dy1],
                            "line_center": [object["x_mid"]*x_res ,object["y_mid"]*y_res ],
                            "iscrowd": 0,
                            "y_min": object["y_min"]*y_res ,
                            "x_min": object["x_min"]*x_res ,
                            "y_max": object["y_max"]*y_res ,
                            "x_max": object["x_max"]*x_res ,
                            "source": object["source"],
                            "magnitude": object["magnitude"],
                            "label_type":label_type
                            }
                        annotations.append(annotation)

                image = {
                    "id": image_id,
                    "collect_id": path,
                    "exp_start_time":image_num,
                    "width": data["sensor"]["width"],
                    "height": data["sensor"]["height"],
                    "num_objects":len(object_list),
                    "y_fov": config_data["fpa"]["y_fov"],
                    "x_fov": config_data["fpa"]["x_fov"],
                    "type":config_data["geometry"]["site"]["track"]["mode"],
                    "exposure_seconds":config_data["fpa"]["time"]["exposure"],
                    "gain":1,
                    "alt":config_data["geometry"]["site"]["alt"],
                    "lat":config_data["geometry"]["site"]["lat"],
                    "lon":config_data["geometry"]["site"]["lon"],
                    "file_name": os.path.join("images", f"{image_id}.{filetype}"),
                    "original_path": fits_file,
                    "date": config_data["geometry"]["time"],
                    "label_type":"simulated"}

                #Add coco image to list of files
                path_to_annotation[fits_file] = {"annotation":annotations, "image":image, "new_id":image_id}

    #Compile final json information for folder
    category1 = {"id": 0,"name": "Satellite","supercategory": "Space Object",}
    category2 = {"id": 1,"name": "Star","supercategory": "Space Object",}
    categories = [category1, category2]
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": "Satellite detection of calsat dataset. ",
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": "Satellite BBox"}


    ### For one large dataset
    # Save annotation data to corresponding train test json
    #Make new data directory
    os.makedirs(output_dataset, exist_ok=True)
    images_folder=os.path.join(output_dataset, "images")
    annotations_folder=os.path.join(output_dataset, "annotations")
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        shutil.rmtree(images_folder)
        print(f"Deleted folder: {images_folder}")
    if os.path.exists(annotations_folder) and os.path.isdir(annotations_folder):
        shutil.rmtree(annotations_folder)
        print(f"Deleted folder: {annotations_folder}")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    images = []
    annotations = []
    for path in path_to_annotation.keys():
        images.append(path_to_annotation[path]["image"])
        annotations.extend(path_to_annotation[path]["annotation"])
    all_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "notes": notes}
    data_attributes_obj=json.dumps(all_data, indent=4)
    with open(os.path.join(annotations_folder, "annotations.json"), "w") as outfile:
        outfile.write(data_attributes_obj)

    #Compressing images without copying them to folder to save on memory
    if zip:
        compress_data(path_to_annotation.keys(), os.path.join(images_folder, "compressed_fits.zip"))
    else:
        for image in tqdm(path_to_annotation.keys(), desc="Copying images", total=len(path_to_annotation.keys())):
            new_file_name = os.path.join(images_folder,str(path_to_annotation[image]["new_id"])+f".{filetype}")
            if convert_png:
                hdu = fits.open(image)
                hdul = hdu[0]
                data = hdul.data
                if process_func is not None:
                    data = process_func(data)
                else: 
                    data = np.stack([data,data,data], axis=0)
                    data = (data / 256).astype(np.uint8)
                data = np.transpose(data, (1,2,0))
                png = Image.fromarray(data)
                png.save(new_file_name)
            else:
                destination_path = shutil.copy(image, images_folder)
                shutil.move(destination_path,new_file_name)

def partition_dataset(coco_directories:str, destination_paths:list[str], partition_attribute:str=None, dataset_size:int=None, notes:str=""):
    """
    Partitions coco datasets into multiple partitions based on a partition attribute. It splits datasets into least to greatest by collect. 

    Args:
        coco_directories (list[str]): Coco dataset to partition or reduce
        destination_paths (list[str]): Directories to split this coco dataset into
        partition_attribute (str): String attribute of the image specific attribute to sort by
        dataset_size (int): Limit to the size of the dataset
        notes (str): Notes to add onto the dataset annotation file for future use

    Returns:
        None
    """
    path_to_image = []
    images_queue = []
    annotations_queue = []
    notes_queue = []
    catagories_queue = []
    collections_queue = {}
    attribute_values = {}

    id_to_index = {}
    anotations_file = os.path.join(coco_directories, "annotations", "annotations.json")
    with open(anotations_file, 'r') as f:
        annotations = json.load(f)
    images_queue.extend(annotations["images"])
    annotations_queue.extend(annotations["annotations"])
    notes_queue.append({"info":annotations["info"], "notes":annotations["notes"], "directory": coco_directories })

    contains_objects = {}
    for item in annotations_queue:
        contains_objects[item["image_id"]] = True

    if dataset_size is not None:
        images_queue = images_queue[:dataset_size]

    for index, item in enumerate(images_queue):
        if item['collect_id'] not in collections_queue:
            collections_queue[item['collect_id']] = [os.path.join(coco_directories,item['file_name'])]
        else:
            collections_queue[item['collect_id']].append(os.path.join(coco_directories,item['file_name']))

        # if item["num_objects"] ==0 and item['collect_id'] not in attribute_values:
        #     attribute_values[item['collect_id']] = [0]
        # elif item["num_objects"] ==0 and item['collect_id'] in attribute_values:
        #     attribute_values[item['collect_id']].append(0)
        if item["id"] not in contains_objects and item['collect_id'] not in attribute_values:
            attribute_values[item['collect_id']] = [0]
        elif item["id"] not in contains_objects and item['collect_id'] in attribute_values:
            attribute_values[item['collect_id']].append(0)
    for index, item in enumerate(annotations["annotations"]):
        if partition_attribute is not None:
            if item['collect_id'] not in attribute_values:
                attribute_values[item['collect_id']] = [item[partition_attribute]]
            else:
                attribute_values[item['collect_id']].append(item[partition_attribute])
        else:
            if item['collect_id'] not in attribute_values:
                attribute_values[item['collect_id']] = [0]
            else:
                attribute_values[item['collect_id']].append(0)


    catagories_queue.extend(annotations["categories"]) #try and find a smarter way of dealing with categories pls
    templist = [os.path.join(coco_directories, "images", image) for image in os.listdir(os.path.join(coco_directories, "images"))]
    path_to_image.extend(templist)
    annotations_id_to_index = [[] for i in range(len(images_queue))]
    for index, item in enumerate(images_queue):
        id_to_index[item["id"]] = index
    for index,anot in enumerate(annotations_queue):
        try: annotations_id_to_index[id_to_index[anot["image_id"]]].append(index)
        except KeyError: pass
    merged_catagories = merge_categories(catagories_queue)

    filetype="fits"
    if ".png" in path_to_image[0]:
        filetype="png"
    
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": "Satellite detection of calsat dataset. ",
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": "Satellite BBox",
        "samples":len(path_to_image),
        "prev_notes":notes_queue}
    
    file_list_split = split_attribute(collections_queue, attribute_values, len(destination_paths))
    for j,temp_list in enumerate(file_list_split):
        indicies = [id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))] for single_path in temp_list]
        temp_anot_index = [annotations_id_to_index[id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))]] for single_path in temp_list]

        ### For one large dataset
        # Save annotation data to corresponding train test json
        #Make new data directory
        data_folder = destination_paths[j]
        subset_folder = os.path.join(destination_paths[j])
        images_alias = os.path.join(destination_paths[j], "images")
        annotations_alias = os.path.join(destination_paths[j], "annotations")
        multiframe_annotations_alias = os.path.join(destination_paths[j], "multiframe_annotations")
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(subset_folder, exist_ok=True)
        os.makedirs(images_alias, exist_ok=True)
        os.makedirs(annotations_alias, exist_ok=True)
        os.makedirs(multiframe_annotations_alias, exist_ok=True)

        t_anot =  [[annotations_queue[j] for j in li ] for li in temp_anot_index]
        all_anot = []
        for t in t_anot:
            all_anot.extend(t)

        all_data = {
            "info": info,
            "images": [images_queue[i] for i in indicies],
            "annotations":all_anot,
            "categories": merged_catagories,
            "notes": notes}
        data_attributes_obj=json.dumps(all_data, indent=4)
        with open(os.path.join(annotations_alias, "annotations.json"), "w") as outfile:
            outfile.write(data_attributes_obj)

        
        for image in tqdm(temp_list, desc="Copying images", total=len(temp_list)):
            shutil.copy(image, images_alias)
    
def build_coco_satnet(satnet_directory:str, destination_path:str, preprocess_functions:list=[raw_file], notes:str=""):
    raw_annotation_directory = os.path.join(satnet_directory, "raw_data")
    path_to_annotation = {}

    ### For one large dataset
    #Make new data directory
    images_folder=os.path.join(destination_path, "images")
    annotations_folder=os.path.join(destination_path, "annotations")
    multiframe_annotations_folder=os.path.join(destination_path, "multiframe_annotations")
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        shutil.rmtree(images_folder)
        print(f"Deleted folder: {images_folder}")
    if os.path.exists(annotations_folder) and os.path.isdir(annotations_folder):
        shutil.rmtree(annotations_folder)
        print(f"Deleted folder: {annotations_folder}")
    if os.path.exists(multiframe_annotations_folder) and os.path.isdir(multiframe_annotations_folder):
        shutil.rmtree(multiframe_annotations_folder)
        print(f"Deleted folder: {multiframe_annotations_folder}")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)
    os.makedirs(multiframe_annotations_folder, exist_ok=True)

    filetype="png"

    collect_dictionary:dict[str,list] = {}

    for folder in tqdm(os.listdir(raw_annotation_directory), desc="Converting SatNet to COCO"):
        annotation_folder = os.path.join(raw_annotation_directory, folder, "Annotations")
        fits_folder =os.path.join(raw_annotation_directory, folder, "ImageFiles")

        for annotation_path,fits_path in tqdm(zip(os.listdir(annotation_folder), os.listdir(fits_folder)), desc=f"Converting {folder} to COCO"):        
            annotation_path = os.path.join(annotation_folder, annotation_path)
            fits_path = os.path.join(fits_folder, fits_path)
            
            with open(annotation_path, 'r') as f:
                try: 
                    json_data = json.load(f)
                    try:
                        json_data = json_data["data"]
                    except KeyError:
                        json_data = json_data
                except  UnicodeDecodeError: continue
            hdu = fits.open(fits_path)


            sample_attributes, object_attributes = collect_stats(json_data, hdu, padding=20)

            hdul = hdu[0]
            header = hdul.header
            x_res = json_data["sensor"]["width"]
            y_res = json_data["sensor"]["height"]
            object_list = json_data["objects"]
            image_id= np.random.randint(0,9223372036854775806)
            annotations = []
            #Process all detected objects
            try:
                collection_id = json_data["image_set_id"]
                collection_start_time = json_data["exp_start_time"]
            except KeyError:
                collection_id = str(np.random.randint(0,9223372036854775806))
                collection_start_time = "2024-04-24T10:12:17.315000+00:00"

            label_type = "SatNet"

            try:
                norad_id = fits_path.split(".")[0]
                spacecraft = SPACECRAFT[norad_id]
            except:
                spacecraft = "None"

            for object in object_list:
                #Create coco annotation for one image
                x1 = (object["x_center"]-object["bbox_width"]/2)*x_res
                y1 = (object["y_center"]-object["bbox_height"]/2)*y_res
                width = object["bbox_width"]*x_res
                height = object["bbox_height"]*y_res
                x_center = object["x_center"]*x_res
                y_center = object["y_center"]*y_res
                if abs(width*height) > 1000:
                    continue

                annotation = {
                    "id": np.random.randint(0,9223372036854775806),
                    "image_id": image_id,
                    "collect_id":collection_id,
                    "exp_start_time":collection_start_time,
                    "category_id": int(object["class_id"]),# Originallly class_id-1, not sure why
                    "category_name": object["class_name"],
                    "type": "bbox",
                    "centroid": [x_center,y_center],
                    "bbox": [x1,y1,width,height],
                    "area": abs(width*height),
                    "exposure": header["EXPTIME"],
                    "iscrowd": 0,
                    "y_min": y_center-height/2,
                    "x_min": x_center-width/2,
                    "y_max": y_center+height/2,
                    "x_max": x_center+width/2,
                    "collect_id":collection_id,
                    "exp_start_time":collection_start_time,
                    "label_type":label_type
                    }

                try: other_annotation_attributes = _find_dict(object["correlation_id"], object_attributes)
                except KeyError: other_annotation_attributes = {}
                annotation = _merge_dicts(annotation,other_annotation_attributes )
                annotations.append(annotation)

            image = {
                "id": image_id,
                "width": json_data["sensor"]["width"],
                "height": json_data["sensor"]["height"],
                "collect_id":collection_id,
                "exp_start_time":collection_start_time,
                "type":"siderial" if header["TELTKRA"] == 0.0 else "rate",
                "rate": header["TELTKRA"],
                "exposure_seconds":header["EXPTIME"],
                "gain":1,
                "lat":header["SITELAT"],
                "lon":header["CENTALT"],
                "file_name": os.path.join("images", f"{image_id}.{filetype}"),
                "original_path": fits_path,
                "date": header["DATE-OBS"],
                "spacecraft":spacecraft,
                "label_type":label_type}
            
            image_collect_information = {
                "collect_id":collection_id,
                "exp_start_time":collection_start_time,
                "path":image["file_name"], 
                "image_id":image["id"]}
            is_part_of_collect = collection_id != "N/A"
            if is_part_of_collect:
                if collection_id not in collect_dictionary:
                    collect_dictionary[collection_id] = []
                    collect_dictionary[collection_id].append(image_collect_information)
                else:
                    collect_dictionary[collection_id].append(image_collect_information)
            
            image = _merge_dicts(image, sample_attributes)

            #Add coco image to list of files
            path_to_annotation[fits_path] = {"annotation":annotations, "image":image, "new_id":image_id}
        
    #Compile final json information for folder
    category1 = {"id": 1,"name": "Satellite","supercategory": "Space Object",}
    category2 = {"id": 2,"name": "Star","supercategory": "Space Object",}
    categories = [category1, category2]
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": notes,
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S")}

    # Compiling information for annotation file
    images = []
    annotations = []
    for path in path_to_annotation.keys():
        images.append(path_to_annotation[path]["image"])
        annotations.extend(path_to_annotation[path]["annotation"])
        
    #Writing annotations to the json annotations file
    all_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "notes": notes}
    with open(os.path.join(annotations_folder, "annotations.json"), "w") as outfile:
        data_attributes_obj=json.dumps(all_data, indent=4)
        outfile.write(data_attributes_obj)

    #Compiling Multiframe annotation json
    final_collections_dictionary = {}
    for collect_id,collect_list in collect_dictionary.items():
        new_collect_info_order = []
        dates_list = []
        for sub_image_info in collect_list:
            dates_list.append(str(sub_image_info["exp_start_time"]))
        argsort_indices = sorted(range(len(dates_list)), key=lambda i: dt.datetime.fromisoformat(dates_list[i]))
        for index, arg_sorted_index in enumerate(argsort_indices):
            dict_copy = collect_list[arg_sorted_index]
            dict_copy["order"] = index
            new_collect_info_order.append(dict_copy)
        final_collections_dictionary[collect_id] = new_collect_info_order
    with open(os.path.join(multiframe_annotations_folder, "multiframe_annotations.json"), "w") as outfile:
        multiframe_annotations=json.dumps(final_collections_dictionary, indent=4)
        outfile.write(multiframe_annotations)

    for func in preprocess_functions:
        os.makedirs(os.path.join(destination_path, f"images_{func.__name__}"), exist_ok=True)

    for image in tqdm(path_to_annotation.keys(), desc="Copying images", total=len(path_to_annotation.keys())):
        for func in preprocess_functions:
            preprocess_path = os.path.join(destination_path, f"images_{func.__name__}")
            new_file_name = os.path.join(preprocess_path,str(path_to_annotation[image]["new_id"])+f".{filetype}")
            hdu = fits.open(image)
            hdul = hdu[0]
            data = hdul.data
            if data is None or data.size==0:
                print(f"{new_file_name} failed to save")
                continue
            if func is not None:
                data = func(data)
            else: 
                data = raw_file(data)
            # data = np.transpose(data, (1,2,0))
            # png = Image.fromarray(data)
            # png.save(new_file_name)
            iio.imwrite(new_file_name, data)
            # if process_func is not None:
            #     data = process_func(data)
            # else: 
            #     data = np.stack([data,data,data], axis=0)
            #     data = (data / 256).astype(np.uint8)
            # data = np.transpose(data, (1,2,0))
            # png = Image.fromarray(data)
            # png.save(new_file_name)


# New age coco handling
def train_test_split(directory:str, notes:str="", train_ratio=0.8, val_ratio=0.10, test_ratio=0.10):
    """
    Generates a coco dataset from a list of coco datasets and compiles them into one destination. Generates a train test split based on collect.

    Args:
        coco_directories (list[str]): List of paths of coco datasets. Raw datasets must be converted to coco before merging
        destination_path (str): Destination of the merged dataset
        notes (str): Notes to add onto the dataset annotation file for future use
        train_test_split (bool): Bool to create a train test split or not
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples
        zip (bool): Number of partitions to divide your data into

    Returns:
        None
    """
    path_to_image = []
    images_queue = []
    annotations_queue = []
    notes_queue = []
    catagories_queue = []
    collections_queue = {}
    id_to_index = {}
    multiframe_annotations_dictionary = {}
    
    anotations_file = os.path.join(directory, "annotations", "annotations.json")
    with open(anotations_file, 'r') as f:
        annotations = json.load(f)
    multiframe_annotation_file = os.path.join(directory, "multiframe_annotations", "multiframe_annotations.json")
    with open(multiframe_annotation_file, 'r') as f:
        multiframe_annotations = json.load(f)
    images_queue.extend(annotations["images"])
    annotations_queue.extend(annotations["annotations"])
    notes_queue.append({"info":annotations["info"], "notes":annotations["notes"], "directory": directory })
    multiframe_annotations_dictionary = multiframe_annotations_dictionary | multiframe_annotations

    for index, item in enumerate(annotations["images"]):
        if item['collect_id'] not in collections_queue:
            collections_queue[item['collect_id']] = []
            collections_queue[item['collect_id']].append(os.path.join(directory,item['file_name']))
        else:
            collections_queue[item['collect_id']].append(os.path.join(directory,item['file_name']))

    catagories_queue.extend(annotations["categories"]) #try and find a smarter way of dealing with categories pls
    templist = [os.path.join(directory, "images", image) for image in os.listdir(os.path.join(directory, "images"))]
    path_to_image.extend(templist)

    annotations_id_to_index = [[] for i in range(len(images_queue))]
    for index, item in enumerate(images_queue):
        id_to_index[item["id"]] = index
    for index,anot in enumerate(annotations_queue):
        annotations_id_to_index[id_to_index[anot["image_id"]]].append(index)
    merged_catagories = merge_categories(catagories_queue)

    filetype="png"
    
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": "Satellite detection of calsat dataset. ",
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": "Satellite BBox",
        "samples":len(path_to_image),
        "prev_notes":notes_queue}
    
    titles = ["train","val","test"]
    file_list_split = split_collections(collections_queue, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    for j, collection_dictionary in enumerate(file_list_split):
        temp_list = [x for lst in collection_dictionary.values() for x in lst]
        indicies = [id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))] for single_path in temp_list]
        temp_anot_index = [annotations_id_to_index[id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))]] for single_path in temp_list]
        new_multiframe_annotations = {cid:multiframe_annotations_dictionary[cid] for cid in collection_dictionary.keys()}

        ### For one large dataset
        # Save annotation data to corresponding train test json
        #Make new data directory
        data_folder = directory
        subset_folder = os.path.join(data_folder, titles[j])
        images_alias = os.path.join(data_folder, titles[j], "images")
        annotations_alias = os.path.join(data_folder, titles[j], "annotations")
        multiframe_annotations_alias = os.path.join(data_folder, titles[j], "multiframe_annotations")
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(subset_folder, exist_ok=True)
        os.makedirs(images_alias, exist_ok=True)
        os.makedirs(annotations_alias, exist_ok=True)
        os.makedirs(multiframe_annotations_alias, exist_ok=True)

        info["train"]= train_ratio
        info["test"]= test_ratio
        info["val"]= val_ratio
        info["samples"]=len(temp_list)

        t_anot =  [[annotations_queue[j] for j in li ] for li in temp_anot_index]
        all_anot = []
        for t in t_anot:
            all_anot.extend(t)

        all_data = {
            "info": info,
            "images": [images_queue[i] for i in indicies],
            "annotations":all_anot,
            "categories": merged_catagories,
            "notes": notes}
        data_attributes_obj=json.dumps(all_data, indent=4)
        with open(os.path.join(annotations_alias, "annotations.json"), "w") as outfile:
            outfile.write(data_attributes_obj)
        with open(os.path.join(multiframe_annotations_alias,"multiframe_annotations.json"), 'w') as f:
            json.dump(new_multiframe_annotations, f)

        folders = [
            name for name in os.listdir(directory)
            if name.startswith("images_") and os.path.isdir(os.path.join(directory, name))
        ]
        for f in folders:
            os.makedirs(os.path.join(data_folder, titles[j],f), exist_ok=True)

        for image in tqdm(temp_list, desc="Copying images", total=len(temp_list)):
            base_image_name = os.path.basename(image)
            for f in folders:
                original_image = os.path.join(directory,f,base_image_name)
                new_location = os.path.join(data_folder, titles[j],f)
                shutil.copy(original_image, new_location)
            shutil.copy(image, images_alias)


    def archive_ttv_splits(self, name):
        archived_ttv_splits_path = os.path.join(self.directory,"archived_TTV_splits")
        os.makedirs(archived_ttv_splits_path, exist_ok=True)
        titles = ["train","val","test"]
        for title in titles:
            annotations_alias = os.path.join(self.directory, title, "annotations", "annotations.json")
            shutil.copy(annotations_alias, os.path.join(archived_ttv_splits_path, f"{name}-{title}-annotations.json"))

    def convert_test_images(self, preprocess_func=raw_file):
        """
        Generates a coco dataset from a list of coco datasets and compiles them into one destination. Generates a train test split based on collect.

        Args:
            coco_directories (list[str]): List of paths of coco datasets. Raw datasets must be converted to coco before merging
            destination_path (str): Destination of the merged dataset
            notes (str): Notes to add onto the dataset annotation file for future use
            train_test_split (bool): Bool to create a train test split or not
            train_ratio (float): Ratio of training samples
            val_ratio (float): Ratio of validation samples
            test_ratio (float): Ratio of testing samples
            zip (bool): Number of partitions to divide your data into

        Returns:
            None
        """
        titles = ["test"]
        for j, split_name in enumerate(titles):
            ### For one large dataset
            # Save annotation data to corresponding train test json
            #Make new data directory
            data_folder = self.directory
            subset_folder = os.path.join(data_folder, titles[j])
            images_alias = os.path.join(data_folder, titles[j], "images")
            annotations_alias = os.path.join(data_folder, titles[j], "annotations")
            annotations_json = os.path.join(data_folder, titles[j], "annotations", "annotations.json")
            with open(annotations_json, 'r') as f:
                data = json.load(f)
                for image in tqdm(data["images"], desc="Copying images", total=len(data["images"])):
                    base_image_name = os.path.basename(f"{image["id"]}.png")
                    original_image = image["raw_fits_path"]
                    new_location = os.path.join(images_alias, base_image_name)

                    hdu = fits.open(original_image)
                    hdul = hdu[0]
                    data = hdul.data
                    if data is None or data.size==0:
                        print(f"{original_image} failed to save")
                        continue
                    if "16bit" in preprocess_func.__name__:
                        data = preprocess_func(data)
                        iio.imwrite(new_location, data)
                    else: 
                        data = preprocess_func(data)
                        data = np.transpose(data, (1,2,0))
                        png = Image.fromarray(data)
                        png.save(new_location)
        self._set_active_preprocess(preprocess_func)
        self._save_coco_pkl()

class COCODataset(PickleSerializable):
    def __init__(self,data_directory):
        self.directory = data_directory
        self.active_preprocessing = None
        if "COCO_dataset_training.pkl" in os.listdir(self.directory):
            self.statistics_file = self.load(os.path.join(self.directory, "COCO_dataset_training.pkl"))
        else:
            self._save_coco_pkl()

    def _save_coco_pkl(self):
        self.save(os.path.join(self.directory, "COCO_dataset_training.pkl"))

    def _set_active_preprocess(self, function):
        self.active_preprocessing = function.__name__
        with open(os.path.join(self.directory, "active_preprocessing.txt"),'w') as f:
            f.write(f"Active preprocessing: {function.__name__}")
    
    def _remove_active_preprocess(self):
        self.active_preprocessing = None
        if os.path.exists(os.path.join(self.directory, "active_preprocessing.txt")):
            os.remove(os.path.join(self.directory, "active_preprocessing.txt"))


    def generate_TTV_split(self, notes:str="", train_ratio=0.8, val_ratio=0.10, test_ratio=0.10):
        """
        Generates a coco dataset from a list of coco datasets and compiles them into one destination. Generates a train test split based on collect.

        Args:
            coco_directories (list[str]): List of paths of coco datasets. Raw datasets must be converted to coco before merging
            destination_path (str): Destination of the merged dataset
            notes (str): Notes to add onto the dataset annotation file for future use
            train_test_split (bool): Bool to create a train test split or not
            train_ratio (float): Ratio of training samples
            val_ratio (float): Ratio of validation samples
            test_ratio (float): Ratio of testing samples
            zip (bool): Number of partitions to divide your data into

        Returns:
            None
        """
        path_to_image = []
        images_queue = []
        annotations_queue = []
        notes_queue = []
        catagories_queue = []
        collections_queue = {}
        id_to_index = {}
        multiframe_annotations_dictionary = {}
        
        anotations_file = os.path.join(self.directory, "annotations", "annotations.json")
        with open(anotations_file, 'r') as f:
            annotations = json.load(f)
        multiframe_annotation_file = os.path.join(self.directory, "multiframe_annotations", "multiframe_annotations.json")
        with open(multiframe_annotation_file, 'r') as f:
            multiframe_annotations = json.load(f)
        images_queue.extend(annotations["images"])
        annotations_queue.extend(annotations["annotations"])
        notes_queue.append({"info":annotations["info"], "notes":annotations["notes"], "directory": self.directory })
        multiframe_annotations_dictionary = multiframe_annotations_dictionary | multiframe_annotations

        for index, item in enumerate(annotations["images"]):
            if item['collect_id'] not in collections_queue:
                collections_queue[item['collect_id']] = []
                collections_queue[item['collect_id']].append(os.path.join(self.directory,item['file_name']))
            else:
                collections_queue[item['collect_id']].append(os.path.join(self.directory,item['file_name']))

        catagories_queue.extend(annotations["categories"]) #try and find a smarter way of dealing with categories pls

        annotations_id_to_index = [[] for i in range(len(images_queue))]
        for index, item in enumerate(images_queue):
            id_to_index[item["id"]] = index
        for index,anot in enumerate(annotations_queue):
            annotations_id_to_index[id_to_index[anot["image_id"]]].append(index)
        merged_catagories = merge_categories(catagories_queue)

        filetype="png"
        
        now = dt.datetime.now()
        info = {
            "year": now.year,
            "version": "1.0",
            "description": "Satellite detection of calsat dataset. ",
            "contributor":"EO Solutions",
            "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
            "annotation": "Satellite BBox",
            "samples":len(annotations["images"]),
            "prev_notes":notes_queue}
        
        titles = ["train","val","test"]
        file_list_split = split_collections(collections_queue, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        for j, collection_dictionary in enumerate(file_list_split):
            temp_list = [x for lst in collection_dictionary.values() for x in lst]
            indicies = [id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))] for single_path in temp_list]
            temp_anot_index = [annotations_id_to_index[id_to_index[int(os.path.basename(single_path).replace(f".{filetype}",""))]] for single_path in temp_list]
            new_multiframe_annotations = {cid:multiframe_annotations_dictionary[cid] for cid in collection_dictionary.keys()}

            ### For one large dataset
            # Save annotation data to corresponding train test json
            #Make new data directory
            data_folder = self.directory
            subset_folder = os.path.join(data_folder, titles[j])
            images_alias = os.path.join(data_folder, titles[j], "images")
            annotations_alias = os.path.join(data_folder, titles[j], "annotations")
            multiframe_annotations_alias = os.path.join(data_folder, titles[j], "multiframe_annotations")
            os.makedirs(data_folder, exist_ok=True)
            os.makedirs(subset_folder, exist_ok=True)
            os.makedirs(images_alias, exist_ok=True)
            os.makedirs(annotations_alias, exist_ok=True)
            os.makedirs(multiframe_annotations_alias, exist_ok=True)

            info["train"]= train_ratio
            info["test"]= test_ratio
            info["val"]= val_ratio
            info["samples"]=len(temp_list)

            t_anot =  [[annotations_queue[j] for j in li ] for li in temp_anot_index]
            all_anot = []
            for t in t_anot:
                all_anot.extend(t)

            all_data = {
                "info": info,
                "images": [images_queue[i] for i in indicies],
                "annotations":all_anot,
                "categories": merged_catagories,
                "notes": notes}
            data_attributes_obj=json.dumps(all_data, indent=4)
            with open(os.path.join(annotations_alias, "annotations.json"), "w") as outfile:
                outfile.write(data_attributes_obj)
            with open(os.path.join(multiframe_annotations_alias,"multiframe_annotations.json"), 'w') as f:
                json.dump(new_multiframe_annotations, f)
        self._save_coco_pkl()

    def move_fits_to_train_test_split(self, preprocess_func=raw_file):
        """
        Generates a coco dataset from a list of coco datasets and compiles them into one destination. Generates a train test split based on collect.

        Args:
            coco_directories (list[str]): List of paths of coco datasets. Raw datasets must be converted to coco before merging
            destination_path (str): Destination of the merged dataset
            notes (str): Notes to add onto the dataset annotation file for future use
            train_test_split (bool): Bool to create a train test split or not
            train_ratio (float): Ratio of training samples
            val_ratio (float): Ratio of validation samples
            test_ratio (float): Ratio of testing samples
            zip (bool): Number of partitions to divide your data into

        Returns:
            None
        """
        titles = ["train","val","test"]
        for j, split_name in enumerate(titles):
            ### For one large dataset
            # Save annotation data to corresponding train test json
            #Make new data directory
            data_folder = self.directory
            subset_folder = os.path.join(data_folder, titles[j])
            images_alias = os.path.join(data_folder, titles[j], "images")
            annotations_alias = os.path.join(data_folder, titles[j], "annotations")
            annotations_json = os.path.join(data_folder, titles[j], "annotations", "annotations.json")
            with open(annotations_json, 'r') as f:
                data = json.load(f)
                for image in tqdm(data["images"], desc="Copying images", total=len(data["images"])):
                    base_image_name = os.path.basename(f"{image["id"]}.png")
                    original_image = image["raw_fits_path"]
                    new_location = os.path.join(images_alias, base_image_name)

                    hdu = fits.open(original_image)
                    hdul = hdu[0]
                    data = hdul.data
                    if data is None or data.size==0:
                        print(f"{original_image} failed to save")
                        continue
                    if "16bit" in preprocess_func.__name__:
                        data = preprocess_func(data)
                        iio.imwrite(new_location, data)
                    else: 
                        data = preprocess_func(data)
                        data = np.transpose(data, (1,2,0))
                        png = Image.fromarray(data)
                        png.save(new_location)
        self._set_active_preprocess(preprocess_func)
        self._save_coco_pkl()

    def build_annotations(self, include_sats:bool=True, include_stars:bool=False, notes:str=""):
        """
        Converts a SILT bucket annotation dataset into a coco dataset

        Args:
            raw_dataset_path (list[str]): Input list of files to shuffle and generate a train test split
            include_sats (bool): Ratio of training samples
            include_stars (bool): Ratio of validation samples
            zip (bool): Ratio of testing samples
            convert_png (bool): Convert to an intermediate PNG  
            process_func (src.preprocess_functions): Preprocess function to convert PNGs to
            notes (str)

        Returns:
            train, test, split (tuple): List of files present in the train test split
        """
        path_to_annotation = {}
        local_files =  raw_dataset(self.directory)

        ### For one large dataset
        #Make new data directory
        annotations_folder=os.path.join(self.directory, "annotations")
        multiframe_annotations_folder=os.path.join(self.directory, "multiframe_annotations")
        if os.path.exists(annotations_folder) and os.path.isdir(annotations_folder):
            shutil.rmtree(annotations_folder)
            print(f"Deleted folder: {annotations_folder}")
        if os.path.exists(multiframe_annotations_folder) and os.path.isdir(multiframe_annotations_folder):
            shutil.rmtree(multiframe_annotations_folder)
            print(f"Deleted folder: {multiframe_annotations_folder}")
        os.makedirs(annotations_folder, exist_ok=True)
        os.makedirs(multiframe_annotations_folder, exist_ok=True)

        filetype="png"

        collect_dictionary:dict[str,list] = {}

        for annotation_path,fits_path in tqdm(local_files.annotation_to_fits.items(), desc="Building COCO Annotations"):        
            with open(annotation_path, 'r') as f:
                json_data = json.load(f)
            hdu = fits.open(fits_path)

            sample_attributes, object_attributes = collect_stats(json_data, hdu, padding=20)

            hdul = hdu[0]
            header = hdul.header
            x_res = json_data["sensor"]["width"]
            y_res = json_data["sensor"]["height"]
            object_list = json_data["objects"]
            image_id= np.random.randint(0,9223372036854775806)
            annotations = []
            #Process all detected objects
            try:
                collection_id = json_data["image_set_id"]
                collection_start_time = json_data["exp_start_time"]
            except KeyError:
                collection_id = str(np.random.randint(0,9223372036854775806))
                collection_start_time = "2024-04-24T10:12:17.315000+00:00"
            try:
                label_type = object["datatype"]
            except:
                label_type = "Real"

            for object in object_list:


                if include_sats and object["class_name"] == "Satellite" and object["type"] == "line":
                    continue
                elif include_sats and object["class_name"] == "Satellite": 
                    #Create coco annotation for one image
                    x1 = (object["x_center"]-object["bbox_width"]/2)*x_res
                    y1 = (object["y_center"]-object["bbox_height"]/2)*y_res
                    width = object["bbox_width"]*x_res
                    height = object["bbox_height"]*y_res
                    x_center = object["x_center"]*x_res
                    y_center = object["y_center"]*y_res
                    if abs(width*height) > 1000:
                        continue

                    annotation = {
                        "id": np.random.randint(0,9223372036854775806),
                        "image_id": image_id,
                        "collect_id":collection_id,
                        "exp_start_time":collection_start_time,
                        "category_id": int(object["class_id"]),# Originallly class_id-1, not sure why
                        "category_name": object["class_name"],
                        "type": "bbox",
                        "centroid": [x_center,y_center],
                        "bbox": [x1,y1,width,height],
                        "area": abs(width*height),
                        "iso_flux": object["iso_flux"],
                        "exposure": header["EXPTIME"],
                        "iscrowd": 0,
                        "y_min": y_center-height/2,
                        "x_min": x_center-width/2,
                        "y_max": y_center+height/2,
                        "x_max": x_center+width/2,
                        "collect_id":collection_id,
                        "exp_start_time":collection_start_time,
                        "label_type":label_type
                        }

                if include_stars and object["class_name"] == "Star": 
                    #Create coco annotation for one image
                    dx1 =(object["x2"]-object["x1"])*x_res
                    dy1 =(object["y2"]-object["y1"])*y_res
                    deltax = abs((object["x2"]-object["x1"])*x_res)
                    deltay = abs((object["y2"]-object["y1"])*y_res)
                    minx = np.min([object["x2"], object["x1"]])*x_res
                    miny = np.min([object["y2"], object["y1"]])*y_res
                    x1 = object["x1"]*x_res
                    y1 = object["y1"]*y_res
                    x_center = object["x_center"]*x_res
                    y_center = object["y_center"]*y_res


                    annotation = {
                        "id": np.random.randint(0,9223372036854775806),
                        "image_id": image_id,
                        "collect_id":collection_id,
                        "exp_start_time":collection_start_time,
                        "category_id": int(object["class_id"]),# Originallly class_id-1, not sure why
                        "category_name": object["class_name"],
                        "type": "bbox",
                        "centroid": [x_center,y_center],
                        "bbox": [minx,miny,deltax,deltay],
                        "area": abs(deltax*deltay),
                        "line": [x1,y1,dx1,dy1],
                        "line_center": [x_center,y_center],
                        "iso_flux": object["iso_flux"],
                        "exposure": header["EXPTIME"],
                        "iscrowd": 0,
                        "collect_id":collection_id,
                        "exp_start_time":collection_start_time,
                        "label_type":label_type
                        }

                other_annotation_attributes = _find_dict(object["correlation_id"], object_attributes)
                annotation = _merge_dicts(annotation,other_annotation_attributes )
                annotations.append(annotation)

            image = {
                "id": image_id,
                "width": json_data["sensor"]["width"],
                "height": json_data["sensor"]["height"],
                "collect_id":collection_id,
                "exp_start_time":collection_start_time,
                "type":"siderial" if header["TELTKRA"] == 0.0 else "rate",
                "rate": header["TELTKRA"],
                "exposure_seconds":header["EXPTIME"],
                "gain":1,
                "lat":header["SITELAT"],
                "lon":header["CENTALT"],
                "file_name": os.path.join("images", f"{image_id}.{filetype}"),
                "original_path": fits_path,
                "date": header["DATE-OBS"],
                "label_type":label_type,
                "raw_annotation_path":annotation_path,
                "raw_fits_path":fits_path
                }
            
            image_collect_information = {
                "collect_id":collection_id,
                "exp_start_time":collection_start_time,
                "path":image["file_name"], 
                "image_id":image["id"]}
            is_part_of_collect = collection_id != "N/A"
            if is_part_of_collect:
                if collection_id not in collect_dictionary:
                    collect_dictionary[collection_id] = []
                    collect_dictionary[collection_id].append(image_collect_information)
                else:
                    collect_dictionary[collection_id].append(image_collect_information)
            
            image = _merge_dicts(image, sample_attributes)

            #Add coco image to list of files
            path_to_annotation[fits_path] = {"annotation":annotations, "image":image, "new_id":image_id}
            
        #Compile final json information for folder
        category1 = {"id": 1,"name": "Satellite","supercategory": "Space Object",}
        category2 = {"id": 2,"name": "Star","supercategory": "Space Object",}
        categories = [category1, category2]
        now = dt.datetime.now()
        info = {
            "year": now.year,
            "version": "1.0",
            "description": notes,
            "contributor":"EO Solutions",
            "date_created": now.strftime("%Y-%m-%d %H:%M:%S")}

        # Compiling information for annotation file
        images = []
        annotations = []
        for path in path_to_annotation.keys():
            images.append(path_to_annotation[path]["image"])
            annotations.extend(path_to_annotation[path]["annotation"])
            
        #Writing annotations to the json annotations file
        all_data = {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "notes": notes}
        with open(os.path.join(annotations_folder, "annotations.json"), "w") as outfile:
            data_attributes_obj=json.dumps(all_data, indent=4)
            outfile.write(data_attributes_obj)

        #Compiling Multiframe annotation json
        final_collections_dictionary = {}
        for collect_id,collect_list in collect_dictionary.items():
            new_collect_info_order = []
            dates_list = []
            for sub_image_info in collect_list:
                dates_list.append(str(sub_image_info["exp_start_time"]))
            argsort_indices = sorted(range(len(dates_list)), key=lambda i: dt.datetime.fromisoformat(dates_list[i]))
            for index, arg_sorted_index in enumerate(argsort_indices):
                dict_copy = collect_list[arg_sorted_index]
                dict_copy["order"] = index
                new_collect_info_order.append(dict_copy)
            final_collections_dictionary[collect_id] = new_collect_info_order
        with open(os.path.join(multiframe_annotations_folder, "multiframe_annotations.json"), "w") as outfile:
            multiframe_annotations=json.dumps(final_collections_dictionary, indent=4)
            outfile.write(multiframe_annotations)
        self._save_coco_pkl()

    def preprocess_images(self, preprocess_functions:list=[raw_file]):
        for func in preprocess_functions:
            os.makedirs(os.path.join(self.directory, f"images_{func.__name__}"), exist_ok=True)

        annotations_path = os.path.join(self.directory,"annotations","annotations.json")
        with open(annotations_path, 'r') as f:
            json_data = json.load(f)
            for im in tqdm(json_data["images"], desc=f"Converting images", total=len(json_data["images"])):
                # im["id"]
                # im["file_name"]
                # im["raw_fits_path"]
                for func in preprocess_functions:
                    preprocess_path = os.path.join(self.directory, f"images_{func.__name__}")
                    new_file_name = os.path.join(preprocess_path, f"{im["id"]}.png")
                    hdu = fits.open(im["raw_fits_path"])
                    hdul = hdu[0]
                    data = hdul.data
                    if data is None or data.size==0:
                        print(f"{new_file_name} failed to save")
                        continue
                    if "16bit" in preprocess_func.__name__:
                        data = preprocess_func(data)
                        iio.imwrite(new_file_name, data)
                    else: 
                        data = preprocess_func(data)
                        data = np.transpose(data, (1,2,0))
                        png = Image.fromarray(data)
                        png.save(new_file_name)
        self._save_coco_pkl()

    def set_primary_images_folder(self, preprocess_functions=raw_file):
        original_directory = os.path.join(self.directory, f"images_{preprocess_functions.__name__}")
        main_images_directory = os.path.join(self.directory, f"images")
        os.makedirs(main_images_directory, exist_ok=True)
        for file in tqdm(os.listdir(original_directory), desc=f"Transferring {preprocess_functions.__name__} Images..."):
            original_file = os.path.join(original_directory, file)
            shutil.copy(original_file, main_images_directory)
        self._set_active_preprocess(preprocess_func)
        self._save_coco_pkl()

    def set_ttv_primary_images_folder(self, preprocess_functions=raw_file):
        for subfolder in ["train","test","val"]:
            original_directory = os.path.join(self.directory, subfolder, f"images_{preprocess_functions.__name__}")
            main_images_directory = os.path.join(self.directory, subfolder, f"images")
            os.makedirs(main_images_directory, exist_ok=True)
            for file in tqdm(os.listdir(original_directory), desc=f"Transferring {preprocess_functions.__name__} Images..."):
                original_file = os.path.join(original_directory, file)
                shutil.copy(original_file, main_images_directory)
            with open(os.path.join(self.directory, subfolder, f"Mode-{preprocess_functions.__name__}.txt"), 'w') as f:
                f.write(f"Current preprocessed in images: {preprocess_functions.__name__}")
        self._set_active_preprocess(preprocess_func)
        self._save_coco_pkl()
            
    def search_for_nan(self):
        annotations_path = os.path.join(self.directory,"annotations","annotations.json")
        nans = 0
        with open(annotations_path,'r') as f:
            for i, line in enumerate(f, start=1):
                if "NaN" in line:
                    nans+=1
                    print(f"Found 'nan' on line {i}: {line.strip()}")
        self._save_coco_pkl()
        return nans

    def clear_extraneous_cache(self):
        remove_list = ["raw_annotation", "raw_fits", "no_sidereal_images","UDL_info","wcs"]
        folders = [name for name in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, name))
            and name not in remove_list]
        
        for folder in folders:
            shutil.rmtree(os.path.join(self.directory, folder))
        self._remove_active_preprocess()
        self._save_coco_pkl()

    def archive_ttv_splits(self, name):
        archived_ttv_splits_path = os.path.join(self.directory,"archived_TTV_splits")
        os.makedirs(archived_ttv_splits_path, exist_ok=True)
        titles = ["train","val","test"]
        for title in titles:
            annotations_alias = os.path.join(self.directory, title, "annotations", "annotations.json")
            shutil.copy(annotations_alias, os.path.join(archived_ttv_splits_path, f"{name}-{title}-annotations.json"))

    def convert_test_images(self, preprocess_func=raw_file):
        """
        Generates a coco dataset from a list of coco datasets and compiles them into one destination. Generates a train test split based on collect.

        Args:
            coco_directories (list[str]): List of paths of coco datasets. Raw datasets must be converted to coco before merging
            destination_path (str): Destination of the merged dataset
            notes (str): Notes to add onto the dataset annotation file for future use
            train_test_split (bool): Bool to create a train test split or not
            train_ratio (float): Ratio of training samples
            val_ratio (float): Ratio of validation samples
            test_ratio (float): Ratio of testing samples
            zip (bool): Number of partitions to divide your data into

        Returns:
            None
        """
        titles = ["test"]
        for j, split_name in enumerate(titles):
            ### For one large dataset
            # Save annotation data to corresponding train test json
            #Make new data directory
            data_folder = self.directory
            subset_folder = os.path.join(data_folder, titles[j])
            images_alias = os.path.join(data_folder, titles[j], "images")
            annotations_alias = os.path.join(data_folder, titles[j], "annotations")
            annotations_json = os.path.join(data_folder, titles[j], "annotations", "annotations.json")
            with open(annotations_json, 'r') as f:
                data = json.load(f)
                for image in tqdm(data["images"], desc="Copying images", total=len(data["images"])):
                    base_image_name = os.path.basename(f"{image["id"]}.png")
                    original_image = image["raw_fits_path"]
                    new_location = os.path.join(images_alias, base_image_name)

                    hdu = fits.open(original_image)
                    hdul = hdu[0]
                    data = hdul.data
                    if data is None or data.size==0:
                        print(f"{original_image} failed to save")
                        continue
                    if "16bit" in preprocess_func.__name__:
                        data = preprocess_func(data)
                        iio.imwrite(new_location, data)
                    else: 
                        data = preprocess_func(data)
                        data = np.transpose(data, (1,2,0))
                        png = Image.fromarray(data)
                        png.save(new_location)
        self._set_active_preprocess(preprocess_func)
        self._save_coco_pkl()

class CalsatDataset(COCODataset):
    def __init__(self, data_directory):
        super().__init__(data_directory)

    def return_files(self):
        with open(os.path.join(self.directory, "original_paths.json"),'r') as f:
            data = json.load(f)
            dirnames = []
            try: 
                for key,value in data["original_jsons"].items():
                    dirname = os.path.dirname(os.path.dirname(value))
                    if dirname not in dirnames:
                        dirnames.append(dirname)
                    if os.path.exists(value):
                        continue
                    else:
                        shutil.move(key,value)
                for key,value in data["original_fits"].items():
                    if os.path.exists(value):
                        continue
                    else:
                        shutil.move(key,value)
            except FileNotFoundError:
                print(f"{key}:{value}")

            for dname in dirnames:
                raw_dataset(dname).recalculate_statistics()

def _merge_dicts(dict1:dict, dict2:dict):
    for key, value in dict2.items():
        if key in dict1:
            continue
        else:
            dict1[key] = value
    return dict1

def _find_dict(correlation_id, annotation_dict:list):
    for dict in annotation_dict:
        if dict["correlation_id"] == correlation_id:
            return dict
    return {}

if __name__ == "__main__":
    from preprocess_functions import adaptiveIQR, zscale, channel_mixture_C
    # silt_dataset_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-24"
    # # silt_to_coco(Process_pathB, include_sats=False, include_stars=True, zip=False, notes="RME01 dataset with stars only")
    # # silt_to_coco(silt_dataset_path, include_sats=True, convert_png=True, process_func=zscale, notes="Z Scaled Initial Dataset for testing")
    # # silt_to_coco(Process_pathB, include_sats=False, include_stars=True, zip=False, notes="RME01 dataset with stars only")
    # silt_to_coco_panoptic(silt_dataset_path, process_func=channel_mixture_C, percent_empty_data=.33, notes="Mixture of ZScale, raw, and log-IQR for target injection imagery")

    # final_data_path="/data/Sentinel_Datasets/Finalized_datasets/"
    # training_set_output_path = os.path.join(final_data_path, f"Panoptic_MC_LMNT01_train")
    # merge_coco([silt_dataset_path], training_set_output_path, train_test_split=True, train_ratio=.9, val_ratio=.1, test_ratio=0, notes="3000 samples of RME04 panoptic stitching mixture C preprocessing ")

    # LA1 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-08"
    # LA2 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-09"
    # LA3 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-10"
    # LA4 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-11"
    # LA5 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-12"
    # LA6 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-13"
    # LA7 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-29"

    # LA8 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-08-04"
    # LA9 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-08-20"
    # LA10 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-09-13"
    # LA11 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-09-25"
    # LA12 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-06"
    # LA13 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-15"
    # LA14 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-23"
    # LA15 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-30"
    # LA16 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-07"
    # LA17 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-15"
    # LA18 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-26"
    # LA19 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-06"
    # LA20 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-17"
    # LA21 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-20"
    # LA22 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-30"
    # LA23 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-07"
    # LA24 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-10"
    # LA25 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-23"
    # LA26 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-03"
    # LA27 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-10"
    # LA28 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-16"
    # LA29 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-25"

    # final_data_path="/data/Sentinel_Datasets/Finalized_datasets/"
    # all_origins = [LA1, LA2, LA3, LA4, LA5, LA6, LA7, LA8, LA9, LA10, LA11, LA12, LA13, LA14, LA15, LA16, LA17, LA18, LA19, LA20, LA21, LA22, LA23, LA24, LA25, LA26, LA27, LA28, LA29]

    # training_set_origins = [LA1, LA2, LA3, LA4, LA5, LA6, LA7]
    # training_set_output_path = os.path.join(final_data_path, f"Panoptic_MC_LMNT01_train")

    # eval_origins =  [LA8, LA9, LA10, LA11, LA12, LA13, LA14, LA15, LA16, LA17, LA18, LA19, LA20, LA21, LA22, LA23, LA24, LA25, LA26, LA27, LA28, LA29]
    # eval_finals = [os.path.join(final_data_path, f"{os.path.basename(ESet)}_Panoptic_MC_Eval") for ESet in eval_origins]

    # preprocess_func = channel_mixture_C

    # merge_coco(training_set_origins, training_set_output_path, train_test_split=True, train_ratio=.9, val_ratio=.1, test_ratio=0, notes="3000 samples of RME04 panoptic stitching mixture C preprocessing ")

    from preprocess_functions import channel_mixture_A, channel_mixture_B, channel_mixture_C, adaptiveIQR, zscale, iqr_clipped, iqr_log, raw_file
    from preprocess_functions import _median_column_subtraction, _median_row_subtraction, _background_subtract
    from utilities import get_folders_in_directory, summarize_local_files, clear_local_caches, clear_local_cache, apply_bbox_corrections
    import os
    from utilities import clear_local_caches
    R1 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-24"
    R2 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-25"
    R3 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-26"
    R4 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-27"
    R5 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-28"
    R6 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-29"
    R7 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-04-30"
    R8 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-01"
    R9 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-02"
    R10 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-03"
    R11 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-07"
    R12 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-08"
    R13 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-09"
    R14 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-10"
    R15 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-30"
    R16 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-31"
    T1 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-12"
    T2 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-08"
    T3 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-13"
    T4 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-09"
    T5 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-10"
    T6 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-07"
    T7 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-08-05"
    T8 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-25"
    T9 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-08-21"
    T10 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-11"
    T11 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-08-30"

    LA1 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-08"
    LA2 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-09"
    LA3 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-10"
    LA4 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-11"
    LA5 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-12"
    LA6 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-13"
    LA7 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-29"

    LA8 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-08-04"
    LA9 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-08-20"
    LA10 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-09-13"
    LA11 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-09-25"
    LA12 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-06"
    LA13 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-15"
    LA14 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-23"
    LA15 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-30"
    LA16 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-07"
    LA17 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-15"
    LA18 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-26"
    LA19 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-06"
    LA20 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-17"
    LA21 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-20"
    LA22 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-30"
    LA23 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-07"
    LA24 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-10"
    LA25 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-23"
    LA26 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-03"
    LA27 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-10"
    LA28 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-16"
    LA29 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-25"

    T1 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-12"
    T2 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-08"
    T3 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-13"
    T4 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-09"
    T5 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-10"
    T6 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-07"
    T7 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-08-05"
    T8 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-25"
    T9 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-08-21"
    T10 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-07-11"
    T11 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-08-30"

    E1 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-09-06"
    E2 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-11-14"
    E3 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-12-07"
    E4 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-09-14"
    E5 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-11-07"
    E6 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-08"
    E7 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-30"
    E8 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-10"
    E9 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-12-24"
    E10 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-12-31"
    E11 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-11-19"
    E12 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-31"
    E13 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-04"
    E14 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-09-10"
    E15 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-11-08"
    E16 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2025-01-08"
    E17 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-12-20"
    E18 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-09-20"
    E19 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-17"
    E20 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2025-01-04"
    E21 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-11-27"
    E22 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-12-17"
    E23 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-20"
    E24 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-09-12"
    E25 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-10-21"
    E26 = "/data/Sentinel_Datasets/LMNT02_Raw/LMNT02Sat-2024-11-15"


    # training_set_origins_LMNT02 = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11]
    training_set_origins_LMNT02 = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11]

    all_origins = [LA1, LA2, LA3, LA4, LA5, LA6, LA7, LA8, LA9, LA10, LA11, LA12, LA13, LA14, LA15, LA16, LA17, LA18, LA19, LA20, LA21, LA22, LA23, LA24, LA25, LA26, LA27, LA28, LA29]

    training_set_origins_LMNT01 = [LA1, LA2, LA3, LA4, LA5, LA6, LA7]
    # training_set_output_path_LMNT01 = os.path.join(final_data_path, f"Panoptic_MC_LMNT01_train")

    eval_origins =  [LA8, LA9, LA10, LA11, LA12, LA13, LA14, LA15, LA16, LA17, LA18, LA19, LA20, LA21, LA22, LA23, LA24, LA25, LA26, LA27, LA28, LA29]
    # eval_finals = [os.path.join(final_data_path, f"{os.path.basename(ESet)}_Panoptic_MC_Eval") for ESet in eval_origins]

    preprocess_func = channel_mixture_C

    # merge_coco(training_set_origins_LMNT01, training_set_output_path_LMNT01, train_test_split=True, train_ratio=.7, val_ratio=.15, test_ratio=.15, notes="3000 samples of RME04 panoptic stitching mixture C preprocessing ")

    training_set_origins_RME04 = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16]
    training_set_origins_LMNT01 = [LA8, LA9, LA10, LA11, LA12, LA13, LA14, LA15]
    training_set_origins_LMNT02 = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11]
    training_set_origins_RME04 = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14]
    training_set_origins_LMNT01 = [LA8, LA9, LA10, LA11, LA12, LA13, LA14]
    training_set_origins_LMNT02 = [E1, E2, E3, E4, E5, E6, E7, E8]

    merge_coco(training_set_origins_RME04, "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_R4_Test", train_test_split=False)
    merge_coco(training_set_origins_LMNT02, "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L2_Test", train_test_split=False)
    merge_coco(training_set_origins_LMNT01, "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_Test", train_test_split=False)


    # destination_paths = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_Low_SNR","/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_High_SNR"]
    # destination_paths = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_1_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_2_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_3_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_4_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_5_SNR"]
    # partition_dataset(training_set_output_path_LMNT01, destination_paths, "local_snr", 10000)
    # merge_coco(training_set_origins_LMNT01, training_set_output_path_LMNT01, train_test_split=False)
    # merge_coco(destination_paths, "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1", train_test_split=False)
    # merge_coco(destination_paths, "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_TTS", train_test_split=True)
    # for path in destination_paths:
    #     tpath = path+"_TTS"
    #     merge_coco([path], tpath, train_test_split=True)

    # satsim_path = "/home/davidchaparro/Repos/SatSim/output"
    # # output_dataset = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/SatsimMixtureC"
    # output_dataset = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim"

    # # destination_paths = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_Low_SNR","/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_High_SNR"]
    
    # destination_paths = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_1_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_2_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_3_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_4_SNR",
    #                      "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_5_SNR"]
    # partition_dataset(output_dataset, destination_paths, "snr", 10000)
    # merge_coco(destination_paths,"/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_TTS", train_test_split=True)
    # merge_coco(destination_paths,"/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim", train_test_split=False)
    # for path in destination_paths:
    #     tpath = path+"_TTS"
    #     merge_coco([path], tpath, train_test_split=True)

    # for path in training_set_origins_LMNT01:
    #     silt_to_coco_panoptic(path, process_func=preprocess_func, percent_empty_data=.5, overlap=10, notes="Mixture of ZScale, raw, and log-IQR for Panoptic COCO imagery at 10 percent overlap")