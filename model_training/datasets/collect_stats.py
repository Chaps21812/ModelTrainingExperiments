from astropy.io import fits
from datetime import datetime
import numpy as np
from math import atan2
from .constants import SPACECRAFT, SITE_PARAMS

def _directed_circular_stats(angles_deg:list) -> tuple[float, float, float]:
    """
    Compute the mean and spread of a list of angles using circular statistics.

    Args:
        angles_deg (list or np.ndarray): List of angles in degrees.

    Returns:
        mean_angle_deg (float): Mean angle in degrees (0 to 360)
        spread (float): Circular standard deviation (in radians)
        R (float): Resultant vector length (0 to 1), inversely proportional to spread
    """
    angles_rad = np.deg2rad(angles_deg)
    sin_vals = np.sin(angles_rad)
    cos_vals = np.cos(angles_rad)

    mean_sin = np.mean(sin_vals)
    mean_cos = np.mean(cos_vals)

    # Mean angle in radians
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360

    # Resultant vector length
    R = np.hypot(mean_cos, mean_sin)

    # Circular standard deviation
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.inf

    return mean_angle_deg, circular_std, R

def _circular_stats(angles_deg:list) -> tuple[float, float, float]:
    """
    Compute circular mean and spread of undirected angles (mod 180°).
    
    Args:
        angles_deg (list or np.ndarray): List of angles in degrees

    Returns:
        mean_angle_deg (float): Mean orientation (0-180°)
        circular_std (float): Circular standard deviation (radians)
        R (float): Resultant vector length (0 to 1), inverse of spread
    """
    angles_rad = np.deg2rad(angles_deg)
    doubled_angles = 2 * angles_rad

    sin_vals = np.sin(doubled_angles)
    cos_vals = np.cos(doubled_angles)

    mean_sin = np.mean(sin_vals)
    mean_cos = np.mean(cos_vals)

    mean_angle_2rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_rad = mean_angle_2rad / 2
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 180

    R = np.hypot(mean_cos, mean_sin)
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.inf

    return mean_angle_deg, circular_std, R

def collect_stats(json_content:dict, fits_content:fits, padding:int=20) -> tuple[dict,dict]:
    """
    Given a json annotation file and fits file, collects all information and compiles it into a coco annotation

    Args:
        json_content (dict): JSON annotation information
        fits_content (fits): Raw Fits file
        padding (int): Padding to consider for the bounding box

    Returns:
        None
    """
    sample_attributes = {}
    object_attributes = []
    padding = padding

    hdu = fits_content[0].header
    data = fits_content[0].data

    sample_attributes = sample_attributes|dict(hdu)
    sample_attributes.pop("COMMENT")

    x_res = hdu["NAXIS1"]
    y_res = hdu["NAXIS2"]
    
    sample_attributes["filename"] = json_content["file"]["filename"]
    sample_attributes = sample_attributes|SITE_PARAMS[json_content["file"]["id_sensor"]]

    if "OBJECT" in sample_attributes.keys() and sample_attributes["OBJECT"] in SPACECRAFT.keys():
        norad_id = sample_attributes["OBJECT"]
        sample_attributes["spacecraft"] = SPACECRAFT[norad_id]
        sample_attributes["spacecraft_name"] = SPACECRAFT[norad_id]["name"]
        sample_attributes["spacecraft_sp3"] = SPACECRAFT[norad_id]["sp3"]
        sample_attributes["spacecraft_norad"] = SPACECRAFT[norad_id]["norad"]
    else:
        sample_attributes["spacecraft"] = None
        sample_attributes["spacecraft_name"] = None
        sample_attributes["spacecraft_sp3"] = None
        sample_attributes["spacecraft_norad"] = None
    try: sample_attributes["id_sensor"] = json_content["file"]["id_sensor"]
    except KeyError: sample_attributes["id_sensor"] = "N/A"
    try: sample_attributes["QA"] = json_content["approved"]
    except KeyError: sample_attributes["QA"] = "N/A"
    try: sample_attributes["label_created"] = json_content["created"]
    except KeyError: sample_attributes["label_created"] = "N/A"
    try:sample_attributes["label_updated"] = json_content["updated"]
    except KeyError:sample_attributes["label_updated"] = "N/A"
    # sample_attributes["too_few_stars"] = json_content["too_few_stars"]
    # sample_attributes["empty_image"] = json_content["empty_image"]
    sample_attributes["num_objects"] = len(json_content["objects"])
    sample_attributes["exposure"] = hdu["EXPTIME"]
    sample_attributes["std_intensity"] = np.std(data)
    sample_attributes["median_intensity"] = np.median(data)
    
    if sample_attributes["median_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10
    if sample_attributes["std_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10

    # date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
    # date_object = datetime.strptime(json_content["created"], date_format)
    try:
        date_format = "%Y-%m-%dT%H:%M:%S.%f"
        date_object = datetime.strptime(hdu["DATE-OBS"], date_format)
    except ValueError:
        date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
        date_object = datetime.strptime(hdu["DATE-OBS"], date_format)
    sample_attributes["dates"] = date_object.strftime("%Y-%m-%d")
    sample_attributes["times"] = date_object.strftime("%H:%M:%S")

    sats = 0
    stars = 0
    streak_angles = []
    streak_lengths = []
    for object in json_content["objects"]:
        detection_dict = {}
        try: detection_dict["correlation_id"] = object["correlation_id"]
        except: detection_dict["correlation_id"] = "Error in collect ID"
        try: detection_dict["flux"] = object['iso_flux']
        except KeyError: detection_dict["flux"] = "N/A"
        try:
            detection_dict["label_type"] = object['datatype']
        except:
            detection_dict["label_type"] = "real"

        x_cord= object["x_center"]*x_res
        y_cord= object["y_center"]*y_res
        try: signal = data[int(y_cord), int(x_cord)]
        except IndexError: signal = data[int(y_cord)-1, int(x_cord)-1]

        try: objtype = object["type"]
        except KeyError: objtype = "bbox"

        if object['class_name']=="Satellite" and objtype != "line": 
            sats+=1
            detection_dict["filename"] = json_content["file"]["filename"]
            detection_dict["object_type"] = object['class_name']
            detection_dict["x_center"] = object['x_center']
            detection_dict["y_center"] = object['y_center']
            detection_dict["x_min"] = object['x_min']
            detection_dict["y_min"] = object['y_min']
            detection_dict["x_max"] = object['x_max']
            detection_dict["y_max"] = object['y_max']
            detection_dict["delta_x"] = object['x_max']-object['x_min']
            detection_dict["delta_y"] = object['y_max']-object['y_min']
            # detection_dict["snr"] = object['snr']
            detection_dict["area"] = detection_dict["delta_x"]*detection_dict["delta_y"]*x_res*y_res

            x_max = max(detection_dict["x_min"], detection_dict["x_max"])*x_res
            x_min = min(detection_dict["x_min"], detection_dict["x_max"])*x_res
            y_max = max(detection_dict["y_min"], detection_dict["y_max"])*y_res
            y_min = min(detection_dict["y_min"], detection_dict["y_max"])*y_res

        elif object['class_name']=="Star": 
            stars+=1
            detection_dict["filename"] = json_content["file"]["filename"]
            detection_dict["object_type"] = object['class_name']
            detection_dict["x_center"] = object['x_center']
            detection_dict["y_center"] = object['y_center']
            detection_dict["x1"] = object['x1']
            detection_dict["y1"] = object['y1']
            detection_dict["x2"] = object['x2']
            detection_dict["y2"] = object['y2']
            detection_dict["delta_x"] = (object['x2']-object['x1'])*x_res
            detection_dict["delta_y"] = (object['y2']-object['y1'])*y_res
            detection_dict["length"] = np.sqrt(detection_dict["delta_x"]**2 + detection_dict["delta_y"]**2)
            detection_dict["angle"] = atan2(detection_dict["delta_y"], detection_dict["delta_x"])*180/np.pi

            x_max = max(detection_dict["x1"], detection_dict["x2"])*x_res
            x_min = min(detection_dict["x1"], detection_dict["x2"])*x_res
            y_max = max(detection_dict["y1"], detection_dict["y2"])*y_res
            y_min = min(detection_dict["y1"], detection_dict["y2"])*y_res

            streak_angles.append(detection_dict["angle"])
            streak_lengths.append(detection_dict["length"])
        else:
            continue    

        y_start = max(0, y_min - padding)
        y_end   = min(data.shape[0], y_max + padding)
        x_start = max(0, x_min - padding)
        x_end   = min(data.shape[1], x_max + padding)
        window = data[int(y_start):int(y_end),int(x_start):int(x_end)]

        x_center = (x_max+x_min)/2
        y_center = (y_max+y_min)/2
        w = abs(x_max-x_min)
        h = abs(y_max-y_min)
        local_backgrounds = []
        for x_shift in (-1,0,1):
            for y_shift in (-1,0,1):
                if y_shift == 0 and x_shift ==0:
                    continue

                temp_x_center = x_center+x_shift*w
                temp_y_center = y_center+y_shift*h

                y_start = max(0, temp_y_center - h/2)
                y_end   = min(data.shape[0], temp_y_center + h/2)
                x_start = max(0, temp_x_center - w/2)
                x_end   = min(data.shape[1], temp_x_center + w/2)
                bkg_window = data[int(y_start):int(y_end),int(x_start):int(x_end)]
                local_backgrounds.append(bkg_window.ravel())

        local_bkgs = np.concatenate(local_backgrounds)
        local_minimum = np.min(window)
        local_median = np.median(window)
        local_mean = np.mean(window)
        local_maximum = np.max(window)
        local_std = np.std(window)
        
        local_bkg_mean = np.mean(local_bkgs)
        if len(local_bkgs) < 1:
            local_bkg_mean = local_minimum

        local_bkg_std = np.std(local_bkgs)

        detection_dict["local_prominence"] = float((signal-local_minimum)/(local_std+1*10**-5))
        detection_dict["local_snr"] = float(-1 if np.isnan((signal-local_bkg_mean)/(local_bkg_std+1*10**-5)) else (signal-local_bkg_mean)/(local_bkg_std+1*10**-5))
        detection_dict["max_signal_diff"] = float(1-(local_maximum-signal)/(local_maximum-local_minimum+1*10**-5))
        # detection_dict["global_snr"] = (signal-local_bkg_mean)/sample_attributes["std_intensity"]
        
        object_attributes.append(detection_dict)

    if len(streak_angles) > 0: 
        mean_angle_deg, circular_std, R = _directed_circular_stats(streak_angles)
        sample_attributes["streak_direction_std"] = circular_std
        sample_attributes["streak_direction_mean"] = mean_angle_deg
        sample_attributes["streak_length_mean"] = np.mean(streak_lengths)
        sample_attributes["streak_length_std"] = np.std(streak_lengths)
    else:
        sample_attributes["streak_direction_std"] = 0
        sample_attributes["streak_direction_mean"] = 0
        sample_attributes["streak_length_mean"] = 0
        sample_attributes["streak_length_std"] = 0

    sample_attributes["num_stars"] = stars
    sample_attributes["num_sats"] = sats        

    desired_attributes = ["rain_condition", "rain", "humidity", "windspeed"]
    for key in desired_attributes:
        if key not in sample_attributes.keys():
            sample_attributes[key] = None
        else:
            sample_attributes["rain_condition"] = hdu["SK.WEATHER.RAINCONDITION"]
            sample_attributes["rain"] = hdu["SK.WEATHER.RAIN"]
            sample_attributes["humidity"] = hdu["SK.WEATHER.HUMIDITY"]
            sample_attributes["windspeed"] = hdu["SK.WEATHER.WINDSPEED"]

    return sample_attributes, object_attributes

def collect_satsim_stats(json_content:dict, fits_content:fits, padding:int=20) -> tuple[dict,dict]:
    """
    Given a json annotation file and fits file, collects all information and compiles it into a coco annotation for satsim datasets

    Args:
        json_content (dict): JSON annotation information
        fits_content (fits): Raw Fits file
        padding (int): Padding to consider for the bounding box

    Returns:
        None
    """
    sample_attributes = {}
    object_attributes = []
    padding = padding

    hdu = fits_content[0].header
    data = fits_content[0].data

    x_res = hdu["NAXIS1"]
    y_res = hdu["NAXIS2"]

    json_content = json_content["data"]

    sample_attributes["filename"] = json_content["file"]["filename"]
    sample_attributes["num_objects"] = len(json_content["objects"])
    sample_attributes["exposure"] = hdu["EXPTIME"]
    sample_attributes["std_intensity"] = np.std(data)
    sample_attributes["median_intensity"] = np.median(data)
    
    if sample_attributes["median_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10
    if sample_attributes["std_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10

    date_format = "%Y-%m-%dT%H-%M-%S.%f"
    date_object = datetime.strptime(json_content["file"]["dirname"], date_format)
    sample_attributes["dates"] = date_object.strftime("%Y-%m-%d")
    sample_attributes["times"] = date_object.strftime("%H:%M:%S")

    sats = 0
    stars = 0
    streak_angles = []
    streak_lengths = []
    for object in json_content["objects"]:
        detection_dict = {}

        try:
            detection_dict["label_type"] = object['datatype']
        except:
            detection_dict["label_type"] = "simulated"

        x_cord= object["x_center"]*x_res
        y_cord= object["y_center"]*y_res
        if x_cord < 0 or x_cord > data.shape[1] or y_cord < 0 or y_cord > data.shape[0]:
            continue
        signal = data[int(y_cord), int(x_cord)]


        if object['class_name']=="Satellite": 
            sats+=1
            detection_dict["filename"] = json_content["file"]["filename"]
            detection_dict["object_type"] = object['class_name']
            detection_dict["x_center"] = object['x_center']
            detection_dict["y_center"] = object['y_center']
            detection_dict["x_min"] = object['x_min']
            detection_dict["y_min"] = object['y_min']
            detection_dict["x_max"] = object['x_max']
            detection_dict["y_max"] = object['y_max']
            detection_dict["delta_x"] = object['bbox_width']
            detection_dict["delta_y"] = object['bbox_height']
            detection_dict["magnitude"] = object['magnitude']
            detection_dict["area"] = detection_dict["delta_x"]*detection_dict["delta_y"]*x_res*y_res

            x_max = max(detection_dict["x_min"], detection_dict["x_max"])*x_res
            x_min = min(detection_dict["x_min"], detection_dict["x_max"])*x_res
            y_max = max(detection_dict["y_min"], detection_dict["y_max"])*y_res
            y_min = min(detection_dict["y_min"], detection_dict["y_max"])*y_res

        if object['class_name']=="Star": 
            stars+=1
            detection_dict["filename"] = json_content["file"]["filename"]
            detection_dict["object_type"] = object['class_name']
            detection_dict["x1"] = object['y_start']
            detection_dict["y1"] = object['y_start']
            detection_dict["x2"] = object['x_end']
            detection_dict["y2"] = object['y_end']
            detection_dict["x_center"] = object['x_center']
            detection_dict["y_center"] = object['y_center']
            detection_dict["x_min"] = object['x_min']
            detection_dict["y_min"] = object['y_min']
            detection_dict["x_max"] = object['x_max']
            detection_dict["y_max"] = object['y_max']
            detection_dict["delta_x"] = (detection_dict['x2']-detection_dict['x1'])*x_res
            detection_dict["delta_y"] = (detection_dict['y2']-detection_dict['y1'])*y_res
            detection_dict["length"] = np.sqrt(detection_dict["delta_x"]**2 + detection_dict["delta_y"]**2)
            detection_dict["angle"] = atan2(detection_dict["delta_y"], detection_dict["delta_x"])*180/np.pi
            detection_dict["magnitude"] = object['magnitude']

            x_max = max(detection_dict["x1"], detection_dict["x2"])*x_res
            x_min = min(detection_dict["x1"], detection_dict["x2"])*x_res
            y_max = max(detection_dict["y1"], detection_dict["y2"])*y_res
            y_min = min(detection_dict["y1"], detection_dict["y2"])*y_res

            streak_angles.append(detection_dict["angle"])
            streak_lengths.append(detection_dict["length"])    

        y_start = max(0, y_min - padding)
        y_end   = min(data.shape[0], y_max + padding)
        x_start = max(0, x_min - padding)
        x_end   = min(data.shape[1], x_max + padding)
        window = data[int(y_start):int(y_end),int(x_start):int(x_end)]

        x_center = (x_max+x_min)/2
        y_center = (y_max+y_min)/2
        w = abs(x_max-x_min)
        h = abs(y_max-y_min)
        local_backgrounds = []
        for x_shift in (-1,0,1):
            for y_shift in (-1,0,1):
                if y_shift == 0 and x_shift ==0:
                    continue

                temp_x_center = x_center+x_shift*w
                temp_y_center = y_center+y_shift*h

                y_start = max(0, temp_y_center - h/2)
                y_end   = min(data.shape[0], temp_y_center + h/2)
                x_start = max(0, temp_x_center - w/2)
                x_end   = min(data.shape[1], temp_x_center + w/2)
                bkg_window = data[int(y_start):int(y_end),int(x_start):int(x_end)]
                local_backgrounds.append(bkg_window.ravel())

        local_bkgs = np.concatenate(local_backgrounds)
        local_minimum = np.min(window)
        local_median = np.median(window)
        local_mean = np.mean(window)
        local_maximum = np.max(window)
        local_std = np.std(window)
        
        local_bkg_mean = np.mean(local_bkgs)
        local_bkg_std = np.std(local_bkgs)

        detection_dict["local_prominence"] = (signal-local_minimum)/(local_std+1*10**-5)
        detection_dict["local_snr"] = (signal-local_bkg_mean)/(local_bkg_std+1*10**-5)
        detection_dict["max_signal_diff"] = 1-(local_maximum-signal)/(local_maximum-local_minimum+1*10**-5)
        # detection_dict["global_snr"] = (signal-local_bkg_mean)/sample_attributes["std_intensity"]
        
        object_attributes.append(detection_dict)

    if len(streak_angles) > 0: 
        mean_angle_deg, circular_std, R = _directed_circular_stats(streak_angles)
        sample_attributes["streak_direction_std"] = circular_std
        sample_attributes["streak_direction_mean"] = mean_angle_deg
        sample_attributes["streak_length_mean"] = np.mean(streak_lengths)
        sample_attributes["streak_length_std"] = np.std(streak_lengths)
    else:
        sample_attributes["streak_direction_std"] = 0
        sample_attributes["streak_direction_mean"] = 0
        sample_attributes["streak_length_mean"] = 0
        sample_attributes["streak_length_std"] = 0

    sample_attributes["num_stars"] = stars
    sample_attributes["num_sats"] = sats        

    try:
        sample_attributes["rain_condition"] = hdu["SK.WEATHER.RAINCONDITION"]
        sample_attributes["rain"] = hdu["SK.WEATHER.RAIN"]
        sample_attributes["humidity"] = hdu["SK.WEATHER.HUMIDITY"]
        sample_attributes["windspeed"] = hdu["SK.WEATHER.WINDSPEED"]
    except Exception as e:
        pass


    return sample_attributes, object_attributes

def _find_centroid_COM(image, bbox, padding=10):

    width = max(5, bbox[2])
    height = max(5, bbox[3])
    # if width==5:
        # print("Box to small,  resize to 5")

    x_center = bbox[0]+bbox[2]/2
    y_center = bbox[1]+bbox[3]/2

    x_min = x_center-width/2
    y_min = y_center-height/2
    x_max = x_center+width/2
    y_max = y_center+height/2

    y_start = max(0, y_min)
    y_end   = min(image.shape[0], y_max)
    x_start = max(0, x_min)
    x_end   = min(image.shape[1], x_max)

    window1 = image[int(y_start):int(y_end),int(x_start):int(x_end)]
    median = np.median(window1)/65535
    mean = np.mean(window1)/65535
    stdev = np.std(window1)/65535
    image_min = np.min(window1)/65535

    total_intensity = 1e-8
    intensity_by_position_x = 0
    intensity_by_position_y = 0
    for x in range(int(x_start), int(x_end)):
        for y in range(int(y_start), int(y_end)):
            intensity = image[int(y),int(x)]/65535
            # if intensity > mean+1*stdev:
            if intensity > median+1*stdev:
                weighted_intensity = (intensity*10)**3
                intensity_by_position_x += weighted_intensity*x
                intensity_by_position_y += weighted_intensity*y
                total_intensity += weighted_intensity

    COM_x = intensity_by_position_x/total_intensity
    COM_y = intensity_by_position_y/total_intensity

    width = (y_end-y_start)#*1.2
    height = (x_end-x_start)#*1.2

    if COM_x + width/2>image.shape[1]:
        width = (image.shape[1]-COM_x)*2
    if COM_x - width/2<0:
        width = (COM_x-0)*2
    if COM_y + height/2>image.shape[0]:
        height = (image.shape[0]-COM_y)*2
    if COM_y - height/2<0:
        height = (COM_y-0)*2


    best_bbox = (COM_x-width/2, COM_y-height/2, width,height) 
    # print(best_bbox)
    
    
    # y_start = max(0, y_min - padding)
    # y_end   = min(image.shape[0], y_max + padding)
    # x_start = max(0, x_min - padding)
    # x_end   = min(image.shape[1], x_max + padding)

    # window2 = image[int(y_start):int(y_end),int(x_start):int(x_end)]
    # median = np.median(window2)/65535
    # mean = np.mean(window2)/65535
    # stdev = np.std(window2)/65535
    # image_min = np.min(window2)/65535
    # image_max = np.max(window2)/65535

    # max_ratio = 1-(image_max-window2)/(image_max-image_min)
    # stdeviation = (window2-mean)/stdev
    # median_stdeviation = (window2-median)/stdev
    # plt.show()

    # fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    # im = ax[0].imshow(max_ratio)
    # ax[0].set_title('Maximum Signal Difference')
    # fig.colorbar(im, ax=ax[0])

    # im = ax[1].imshow(stdeviation)
    # ax[1].set_title('Mean standard deviation')
    # fig.colorbar(im, ax=ax[1])

    # im = ax[2].imshow(median_stdeviation)
    # ax[2].set_title('Median standard deviation')
    # fig.colorbar(im, ax=ax[2])

    # plt.tight_layout()
    # plt.show()


    # for x in range(int(x_start)-1, int(x_end)+1):
    #     for y in range(int(y_start)-1, int(y_end)+1):
    #         intensity = image[int(y),int(x)]/65535
    #         max_signal_diff = 1-(image_max-intensity)/(image_max-image_min)
    #         if max_signal_diff > .5 or intensity > median+3*stdev:
    #             count +=1

    # width = 2*np.sqrt(count/3.14)+1
    

    return best_bbox

def _find_new_centroid(image, bbox, padding=20):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[0]+bbox[2]
    y_max = bbox[1]+bbox[3]

    y_start = max(0, y_min - padding)
    y_end   = min(image.shape[0], y_max + padding)
    x_start = max(0, x_min - padding)
    x_end   = min(image.shape[1], x_max + padding)

    best_max_signal_diff = 0
    best_bbox = bbox
    for x in range(int(x_end-x_start)):
        for y in range(int(y_end-y_start)):
            new_bbox = (x_start+x, y_start+y, bbox[2],bbox[3]) ### THIS IS WHERE YOU LEFT OFF PLEASE CALCULATE WHERE THE NEW CENTER IS
            max_signal_diff = _calculate_max_signal_diff(image, new_bbox, padding)
            if max_signal_diff > best_max_signal_diff:
                best_bbox = new_bbox
                best_max_signal_diff = max_signal_diff
    best_bbox_x_center = best_bbox[0]+best_bbox[2]/2
    best_bbox_y_center = best_bbox[1]+best_bbox[3]/2
    num_above_threshold = 0
    for width in range(int(max(bbox[2],bbox[3])+2*padding)):
        new_bbox=(best_bbox_x_center-width/2, best_bbox_y_center-width/2, width+1, width+1)
        width_measurement = _calculate_intensity(image, new_bbox, 50)
        if width_measurement:
            best_bbox = new_bbox
            break
    return best_bbox

def _calculate_intensity(image,bbox, padding): 
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[0]+bbox[2]
    y_max = bbox[1]+bbox[3]

    y_start1 = max(0, y_min - padding)
    y_end1   = min(image.shape[0], y_max + padding)
    x_start1 = max(0, x_min - padding)
    x_end1   = min(image.shape[1], x_max + padding)

    y_start2 = max(0, y_min)
    y_end2   = min(image.shape[0], y_max)
    x_start2 = max(0, x_min)
    x_end2   = min(image.shape[1], x_max)

    window1 = image[int(y_start1):int(y_end1),int(x_start1):int(x_end1)]
    window2 = image[int(y_start2):int(y_end2),int(x_start2):int(x_end2)]


    #### Attempted algorithm to resize square
    # padded_local_max = np.max(window1)
    # padded_local_min = np.min(window1)

    # max_signal_diff = 1-(padded_local_max-window2)/(padded_local_max-padded_local_min+1*10**-5)
    # minimum_max_signal_diff = np.min(max_signal_diff)

    # if minimum_max_signal_diff<.1:
    #     return True
    # else:
    #     return False

    signal = np.min(window2)
    q1 = np.percentile(window1, 15)
    q2 = np.percentile(window1, 50)
    q3 = np.percentile(window1, 75)
    local_std = np.std(window1)
    mean = np.mean(window1)
    local_median = np.median(window1)

    if signal<q1:
        return True
    else:
        return False

def _calculate_num_high_pixels(image, new_bbox, bbox, padding=20):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[0]+bbox[2]
    y_max = bbox[1]+bbox[3]
    y_start1 = max(0, y_min - padding)
    y_end1   = min(image.shape[0], y_max + padding)
    x_start1 = max(0, x_min - padding)
    x_end1   = min(image.shape[1], x_max + padding)

    window1 = image[int(y_start1):int(y_end1),int(x_start1):int(x_end1)]
    q3 = np.percentile(window1, 75)

    x_min = new_bbox[0]
    y_min = new_bbox[1]
    x_max = new_bbox[0]+new_bbox[2]
    y_max = new_bbox[1]+new_bbox[3]
    y_start2 = max(0, y_min)
    y_end2   = min(image.shape[0], y_max)
    x_start2 = max(0, x_min)
    x_end2   = min(image.shape[1], x_max)

    window2 = image[int(y_start2):int(y_end2),int(x_start2):int(x_end2)]
    num_above_q3 = np.sum(window2 > q3)

    return num_above_q3

def _calculate_max_signal_diff(image, bbox, padding=20):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[0]+bbox[2]
    y_max = bbox[1]+bbox[3]
    x_center = bbox[0]+bbox[2]/2
    y_center = bbox[1]+bbox[3]/2
    y_center = max(0, y_center)
    y_center   = min(image.shape[0]-1, y_center)
    x_center = max(0, x_center )
    x_center   = min(image.shape[1]-1, x_center )

    signal = image[int(y_center), int(x_center)]

    y_start = max(0, y_min - padding)
    y_end   = min(image.shape[0], y_max + padding)
    x_start = max(0, x_min - padding)
    x_end   = min(image.shape[1], x_max + padding)

    window = image[int(y_start):int(y_end),int(x_start):int(x_end)]
    local_minimum = np.min(window)
    local_maximum = np.max(window)

    max_signal_diff = 1-(local_maximum-signal)/(local_maximum-local_minimum+1*10**-5)
    return max_signal_diff