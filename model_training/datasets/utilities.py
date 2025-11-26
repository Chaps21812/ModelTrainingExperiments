from .plots import detect_column_type, plot_categorical_column, plot_numerical_column, plot_time_column, plot_lines, plot_scatter
from .raw_datset import raw_dataset
import os
from IPython.display import clear_output
import torch
import json
import torchvision
import shutil
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def search_image_id(directory, fits_uuid):
    for file in os.listdir(os.path.join(directory,"raw_annotation")):
        with open(os.path.join(os.path.join(directory,"raw_annotation",file)),'r') as f:
            data = json.load(f)
            if fits_uuid in data["file"]["filename"]:
                print(os.path.join(os.path.join(directory,"raw_annotation",file)))

def save_torch_script_model(model_path:str, output_path:str, name:str):
    model =  torchvision.models.detection.retinanet_resnet50_fpn()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    example_input = [torch.randn(1, 3, 2048, 2048)]
    # Trace the model
    traced_script_module = torch.jit.script(model, example_input)
    path = os.path.join(output_path,f"{name}-TS.pt" )
    traced_script_module.save(path)

def get_folders_in_directory(directory_path) ->list:
  if not os.path.exists(directory_path):
    return []
  folders = [
      entry.name for entry in os.scandir(directory_path) if entry.is_dir()
  ]
  return folders

def delete_large_annotations(storage_path, area=1200):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = raw_dataset(os.path.join(storage_path,file))
        print(f"File: {file}, {len(local_files)}")

        image_attributes = local_files.statistics_file.sample_attributes
        annotation_attributes = local_files.statistics_file.annotation_attributes
        try: 
            large = len(annotation_attributes[annotation_attributes['area'] >= area])
            local_files.delete_files_from_sample(annotation_attributes[annotation_attributes['area'] >= area])
            print(f"Deleting: {large}/{len(annotation_attributes)} Large Annotation Images")
            regenerate_plots(os.path.join(storage_path,file))
        except KeyError: pass

def summarize_local_files(storage_path):
    total_images = 0
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = raw_dataset(os.path.join(storage_path,file))
        total_images += len(local_files)
        print(f"Path: {file} Num Samples: {len(local_files)}")
    print(f"Total Samples: {total_images}")

def clear_local_caches(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = raw_dataset(os.path.join(storage_path,file))
        local_files.clear_cache()

def clear_local_cache(storage_path):
    local_files = raw_dataset(storage_path)
    local_files.clear_cache()

def apply_bbox_corrections(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        print(file)
        local_files = raw_dataset(os.path.join(storage_path,file))
        local_files.correct_annotations(apply_changes=True, require_approval=False)
        
def apply_bbox_corrections_list(paths:list):
    for file in paths:
        #Local file handling tool
        print(file)
        local_files = raw_dataset(file)
        local_files.correct_annotations(apply_changes=True, require_approval=False)
        
def recalculate_all_statistics(storage_path):
    for file in get_folders_in_directory(storage_path):
        print(file)
        #Local file handling tool
        local_files = raw_dataset(os.path.join(storage_path,file))
        local_files.recalculate_statistics()

def generate_plots(storage_path):
    for file in get_folders_in_directory(storage_path):
        local_files = raw_dataset(os.path.join(storage_path,file))

        plots_save_path = os.path.join(storage_path,file, "plots")
        data_statistics = local_files.statistics_file
        #Plot all statistics collected in the file
        for col_name, col_data in data_statistics.sample_attributes.items():
            column_type = detect_column_type(col_data)
            print(column_type)
            if column_type == "categorical":
                try: 
                    plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
            elif column_type == "numerical":
                try: 
                    plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
            elif column_type == "time":
                try: 
                    plot_time_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
        for col_name, col_data in data_statistics.annotation_attributes.items():
            column_type = detect_column_type(col_data)
            if column_type == "categorical":
                try: 
                    plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
            elif column_type == "numerical":
                try: 
                    plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")

        try:
            #Plot the x and y locations of the annotations
            x_locations=data_statistics.annotation_attributes["x_center"]
            y_locations=data_statistics.annotation_attributes["y_center"]
            plot_scatter(x_locations, y_locations, alpha=.05, filepath=plots_save_path, dpi=500)
            _clear_plots()
        except: print("Plotting Error: Centroid positions")

        #Plot line segments
        try: 
            plot_lines(data_statistics.annotation_attributes["x1"], data_statistics.annotation_attributes["y1"],
                    data_statistics.annotation_attributes["x2"], data_statistics.annotation_attributes["y2"],
                    filepath=plots_save_path, dpi=500, alpha=.10)
            _clear_plots()
        except: print("Plotting Error: Line Streaks")

def regenerate_plots(storage_path):
    local_files = raw_dataset(os.path.join(storage_path))

    plots_save_path = os.path.join(storage_path, "plots")
    data_statistics = local_files.statistics_file
    #Plot all statistics collected in the file
    for col_name, col_data in data_statistics.sample_attributes.items():
        column_type = detect_column_type(col_data)
        print(column_type)
        if column_type == "categorical":
            try: 
                plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
        elif column_type == "numerical":
            try: 
                plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
        elif column_type == "time":
            try: 
                plot_time_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
    for col_name, col_data in data_statistics.annotation_attributes.items():
        column_type = detect_column_type(col_data)
        if column_type == "categorical":
            try: 
                plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
        elif column_type == "numerical":
            try: 
                plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")

    try:
        #Plot the x and y locations of the annotations
        x_locations=data_statistics.annotation_attributes["x_center"]
        y_locations=data_statistics.annotation_attributes["y_center"]
        plot_scatter(x_locations, y_locations, alpha=.05, filepath=plots_save_path, dpi=500)
        _clear_plots()
    except: print("Plotting Error: Centroid positions")

    #Plot line segments
    try: 
        plot_lines(data_statistics.annotation_attributes["x1"], data_statistics.annotation_attributes["y1"],
                data_statistics.annotation_attributes["x2"], data_statistics.annotation_attributes["y2"],
                filepath=plots_save_path, dpi=500, alpha=.10)
        _clear_plots()
    except: print("Plotting Error: Line Streaks")

def _clear_plots():
    plt.clf()   # Clears the current figure
    plt.cla()   # Clears the current axes (optional)
    plt.close()
    clear_output(wait=True)

def get_folders_in_directory(directory_path):
    if not os.path.exists(directory_path):
        return []
    folders = [
        os.path.join(directory_path,entry.name) for entry in os.scandir(directory_path) if entry.is_dir()
    ]
    folders.sort(key=lambda path: os.path.basename(path).lower())
    return folders

def view_folders(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = raw_dataset(file)
        print(f"{file}: {len(local_files)}")

def count_images_in_datasets(storage_path):
    training_set_directories = ["train", "test", "val"]
    raw_directories =  ["raw_annotation", "raw_fits"]
    total=0
    folder_list = get_folders_in_directory(storage_path)
    if "raw_annotation" in [os.path.basename(fold) for fold in folder_list]:
        directory = os.path.join(storage_path,"raw_fits")
        total_images = os.listdir(directory)
        total+=len(total_images)
        print(f"{directory}: {len(total_images)}")

    elif "images" in [os.path.basename(fold) for fold in folder_list]:
        directory = os.path.join(folder_list[0], "images")
        total_images = os.listdir(directory)
        total+=len(total_images)
        print(f"{directory}: {len(total_images)}")

    elif "train" in [os.path.basename(fold) for fold in folder_list]:
        for subfolder in training_set_directories:
            directory = os.path.join(folder_list[0], subfolder, "images")
            total_images = os.listdir(directory)
            total+=len(total_images)
            print(f"{directory}: {len(total_images)}")
        total+=len(total_images)
    return total

def clear_cache(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = raw_dataset(file)
        local_files.clear_cache()

def recalculate_statistics(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = raw_dataset(file)
        local_files.recalculate_statistics()

def get_list_attribute(directory, attribute:str):
    save_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Plots/NumpyArrays"
    annotation_path = os.path.join(directory, "annotations","annotations.json")
    images = {}
    sample_array = []
    with open(annotation_path, 'r') as f:
        json_annot = json.load(f)
        for annotation in json_annot["annotations"]:
            if annotation["image_id"] not in images:
                images[annotation["image_id"]] = []
                if isinstance(annotation[attribute], list):
                    images[annotation["image_id"]].extend(annotation[attribute])
                else:
                    images[annotation["image_id"]].append(annotation[attribute])
            else:
                if isinstance(annotation[attribute], list):
                    images[annotation["image_id"]].extend(annotation[attribute])
                else:
                    images[annotation["image_id"]].append(annotation[attribute])
        for annotation in json_annot["images"]:
            if annotation["id"] not in images:
                images[annotation["id"]] = [0]
    for key,value in images.items():
        sample_array.append(np.average(value))
    base_path = os.path.basename(directory)
    print(f"Directory: {directory}")
    print(f"Save path: {os.path.join(save_path,f"{base_path}_{attribute}.npy")}")
    print(f"\tLength: {len(sample_array)}")
    print(f"\tMin: {np.min(sample_array)}")
    print(f"\tMedian: {np.median(sample_array)}")
    print(f"\tMean: {np.average(sample_array)}")
    print(f"\tMax: {np.max(sample_array)}")
    print(f"\tstd: {np.std(sample_array)}")
    print(f"{len(sample_array):.2f} & {np.min(sample_array):.2f} & {np.median(sample_array):.2f} & {np.max(sample_array):.2f}")
    np.save(os.path.join(save_path,f"{base_path}_{attribute}.npy"), sample_array)

def generate_original_dataset(directory, output_dir):
    annotations_path = os.path.join(directory, "annotations", "annotations.json")

    annotation_output_dir = os.path.join(output_dir, "raw_annotation")
    fits_output_dir = os.path.join(output_dir, "raw_fits")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(annotation_output_dir, exist_ok=True)
    os.makedirs(fits_output_dir, exist_ok=True)

    with open(annotations_path, "r") as f:
        json_annotations = json.load(f)
    for image in tqdm(json_annotations["images"]):
        original_fits = image["original_path"]
        base_fits_name:str = os.path.basename(original_fits)
        base_annotation_name = base_fits_name.replace(".fits",".json")
        original_annotations = os.path.join(os.path.dirname(os.path.dirname(original_fits)), "raw_annotation", base_annotation_name)
        new_annotation_name = os.path.join(annotation_output_dir, base_annotation_name)
        new_destination = os.path.join(fits_output_dir, base_fits_name)
        shutil.copy(original_fits, new_destination)
        shutil.copy(original_annotations, new_annotation_name)

def get_list_of_empty_images(directory):
    json_path = os.path.join(directory,"annotations","annotations.json")
    with open(json_path, 'r') as f:
        json_contents = json.load(f)
    id_counter = Counter()
    # for im in json_contents["images"]:
    #     im["id"]
    for annot in json_contents["annotations"]:
        id_counter.update([annot["image_id"]])


    num_images = len(json_contents["images"])
    num_images_with_target = len(id_counter.keys())
    num_without_target = num_images-len(id_counter.keys())
    Ratio = num_without_target/num_images_with_target
    percent_empty = (num_without_target)/num_images

    print(directory)
    print(f"Num Images = {num_images:.2f}, Num images with target = {num_images_with_target:.2f}, Num images without target = {num_without_target:.2f}, ratio={Ratio:.2f}, Percent empty = {percent_empty:.2f}")
    print(f"&{num_images:.2f}&{num_images_with_target:.2f}&{num_without_target:.2f}&{Ratio:.2f}&{percent_empty:.2f}\\\\")

def get_bbox_sizes(directory, save_directory=None, save_type="pdf", color="#AABBCCFF"):
    print(directory)
    json_path = os.path.join(directory,"annotations","annotations.json")
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    with open(json_path, 'r') as f:
        json_contents = json.load(f)
    for annot in json_contents["annotations"]:
        bbox_areas.append(annot["area"]/(2394*1595)*100)
        bbox_widths.append(annot["bbox"][2]/2394)
        bbox_heights.append(annot["bbox"][3]/1595)

    if save_directory is not None:
        plt.hist(bbox_areas, bins=30, color=color,edgecolor='black')
        plt.title("Area Distribution")
        plt.xlabel("Area Percentage (%)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_directory, f"area_distribution.{save_type}"))
        plt.close()

        plt.hist(bbox_widths, bins=30, color=color,edgecolor='black')
        plt.title("Width Distribution")
        plt.xlabel("Width Percentage")
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_directory, f"width_distribution.{save_type}"))
        plt.close()

        plt.hist(bbox_heights, bins=30, color=color,edgecolor='black')
        plt.title("Height Distribution")
        plt.xlabel("Height Percentage")
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_directory, f"height_distribution.{save_type}"))
        plt.close()
        

    # "width": 2394,
    # "height": 1595,

    print(f"AVG area={np.average(bbox_areas):.2f}, Median area={np.median(bbox_areas):.2f}, Min area={np.min(bbox_areas):.2f}, max area={np.max(bbox_areas):.2f}")
    print(f"AVG width={np.average(bbox_widths):.2f}, Median width={np.median(bbox_widths):.2f}, Min width={np.min(bbox_widths):.2f}, max width={np.max(bbox_widths):.2f}")
    print(f"AVG height={np.average(bbox_heights):.2f}, Median height={np.median(bbox_heights):.2f}, Min height={np.min(bbox_heights):.2f}, max height={np.max(bbox_widths):.2f}")






if __name__ == "__main__":
    # model_path = "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat_Training_Channel_Mixture_C/models/LMNT01_MixtureC/retinanet_weights_E249.pt"
    # output_path = "/data/Sentinel_Datasets/Best_models"
    # name = "LMNT01"
    # save_torch_script_model(model_path, output_path,name)
    real_paths = []
    sim_paths = []

    real_paths = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_1_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_2_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_3_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_4_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_5_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_High_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_Low_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_Test",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L2_Test",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_R4_Test"]
    sim_paths = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_1_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_2_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_3_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_4_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_5_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_High_SNR",
    "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_Low_SNR"]

    for path in real_paths:
        get_list_attribute(path,"local_snr")
    for path in sim_paths:
        get_list_attribute(path,"snr")