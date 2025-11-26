import boto3
import json
import io
import os
from astropy.io import fits
from tqdm import tqdm
from collections import Counter
import numpy as np
import requests
import json
from datetime import datetime
from botocore.exceptions import ClientError

from .raw_datset import StatisticsFile
from .collect_stats import collect_stats
from .documentation import write_count
from .UDL_KEY import UDL_KEY
from .constants import SPACECRAFT


class S3Client:
    def __init__(self, directory:str):
        """
        Scans a directory on the S3 silt bucket and has functions to download and organize FITS files. Needs a key from the AWS AI_Autonomy Dev cloud to work correctly, and a refreshed UDL key to download stuff.

        Args:
            directory (str): Annotation directory on AWS S3 bucket. 

        Returns:
            S3Client
        """
        self.client = boto3.client('s3')
        self.bucket = "silt-annotations"
        self.directory = directory

        self.folders = []
        self.annotation_to_fits = {}
        
        self.fits_filename_to_path = {}
        self.annotation_filename_to_path = {}
        self.date_to_keystring = {}

        self.unmatched_annotations = {}
        
        self.basenames = Counter()
        self.dirnames = Counter()
        self.annotation_dates = Counter()

        self.errors = 0
        self.get_data(self.directory)

    def get_data(self, directory):
        """
        Matches all annotations in the directory to their corresponding FITS file. If no FITS file is recorded, it searches UDL

        Args:
            directory (str): S3 directory of all data.

        Returns:
            None
        """
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=directory)

        for i, page in enumerate(tqdm(page_iterator, desc="Loading file locations")):
            if "Contents" in page:
                for key in page[ "Contents" ]:
                    keyString = key["Key"]
                    date = key["LastModified"].date()
                    dirname = os.path.dirname(keyString)
                    basename = os.path.basename(keyString)
                    if ".fits" in keyString:
                        self.fits_filename_to_path[basename] = keyString
                        self.basenames.update([basename])
                    elif ".json" in keyString:
                        self.annotation_filename_to_path[basename] = keyString
                        if date in self.date_to_keystring:
                            self.date_to_keystring[date].append(keyString)
                        else: 
                            self.date_to_keystring[date] = [keyString]
                        self.annotation_dates.update([date])
                        self.dirnames.update([dirname])
                        self.basenames.update([basename])
                    else:
                        self.folders.append(keyString)
        self._create_annotation_mapping()
        print(f"FITS:{len(self.fits_filename_to_path)}, Annot:{len(self.annotation_filename_to_path)}, Unmatched Annotations:{len(self.unmatched_annotations)}, TotalSynced:{len(self.annotation_to_fits)}")

        sorted_dates = sorted(set(self.annotation_dates.keys()))
        print("Annotation Dates: ")
        for d in sorted_dates:
            count = self.annotation_dates.get(d, 0)
            print(f"{d}: {count}")

    def _create_annotation_mapping(self) -> dict:
        """
        If the fits file and annotations are in the same folder in SILT, it finds the corresponding FITS file and pairs them. 

        Args:
            None

        Returns:
            self.annotation_to_fits (dict): Dictionary matching annotation path to the fits path
        """
        for annotation_filename, annot_full_path in self.annotation_filename_to_path.items():
            fits_file = annotation_filename.replace(".json", ".fits")
            if fits_file in self.fits_filename_to_path:
                fits_full_path = self.fits_filename_to_path[fits_file]
                self.annotation_to_fits[annot_full_path] = fits_full_path
            else:
                self.unmatched_annotations[annotation_filename] = annot_full_path
                self.errors +=1

        return self.annotation_to_fits

    def _download_annotation(self, annotation_path:str) -> dict:
        """
        Downloads a JSON annotation from the S3 Bucket

        Args:
            annotation_path (str): S3 json annotation path

        Returns:
            json_content (dict): Dictionary of JSON annotation Content
        """
        response = self.client.get_object(Bucket=self.bucket, Key=annotation_path)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        return json_content
    
    def _download_fits(self, fits_path:str) -> fits:
        """
        Downloads a FITS file from the SILT S3 Bucket

        Args:
            annotation_path (str): S3 fits image path

        Returns:
            json_content (dict): fits object
        """
        response = self.client.get_object(Bucket=self.bucket, Key=fits_path)
        fits_content = io.BytesIO(response['Body'].read())
        return fits.open(fits_content)
    
    def summarize_s3_structure(self):
        """
        Searches through a folder and gives a count of the number of files in each directory

        Args:
            None

        Returns:
            None
        """
        print(f"# of annotations: {len(self.annotation_filename_to_path)}")
        print(f"# of fits: {len(self.fits_filename_to_path)}")
        print(f"# of folders: {len(self.folders)}")
        for dirname,count in self.dirnames.items():
            print(f"{dirname}: {count}")
        # for tuplee in self.basenames.most_common(5):
        #     print(f"Filename: {tuplee[0]}, Count: {tuplee[1]}")

    def summarize_s3_file_sizes(self):
        """
        Summarizes the file sizes in the listed SILT s3 bucket. Good for validation

        Args:
            annotation_path (str): S3 fits image path

        Returns:
            json_content (dict): fits object
        """
        array = np.array(self.file_sizes)
        summation = np.sum((array > 265).astype(int))
        avg = np.average(array)

        print(f"# of Annotations on S3: {len(array)}")
        print(f"Avg Annotation Size: {avg}")
        print(f"# of Annots above 265B: {summation}")

    def check_annotations(self):
        """
        Checks to see how many images are missing from a bucket

        Args:
            None

        Returns:
            None
        """
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=self.directory)

        for i, page in enumerate(tqdm(page_iterator, desc="Loading file locations")):
            if "Contents" in page:
                for key in page[ "Contents" ]:
                    keyString = key["Key"]
                    date = key["LastModified"]
                    size = key["Size"]
                    if "fits" in keyString:
                        continue
                    elif ".json" in keyString:
                        if size > 265:
                            self._contains_objects[keyString] = date
                        else:
                            self._contains_unknown_structure[keyString] = date
        normal_structure = list(self._contains_objects.values())
        weird_structure = list(self._contains_unknown_structure.values())

        normal_dates = Counter([dt.date() for dt in normal_structure])
        weird_dates = Counter([dt.date() for dt in weird_structure])

        all_dates = sorted(set(normal_dates.keys()) | set(weird_dates.keys()))

        for date in all_dates:
            normal_count = normal_dates.get(date, 0)
            weird_count = weird_dates.get(date, 0)
            print(f"{date}: normal={normal_count} missing_objects={weird_count}")

    def download_annotation_dates(self, date:str, download_directory:str, sort_by_date = True):
        """
        Downloads annotations and images from SILT or UDL. Given a annotation date, you can download all annotations for that day. 

        Args:
            date (str): Annotation date, you can find it when the s3 bucket is initialized
            download_directory (str): Local path to store downloaded data
            sort_by_date (bool): Saves files in date sorted folders according to collect date

        Returns:
            None
        """
        db = StatisticsFile()
        os.makedirs(download_directory, exist_ok=True)

        counter = Counter()

        date = datetime.strptime(date, "%Y-%m-%d").date()

        annotations = self.date_to_keystring[date]

        for annotT in tqdm(annotations, desc="Downloading and Collecting Statistics"):
            sample_attributes = {}
            object_attributes = []

            try:
                json_content = self._download_annotation(annotT)

                if "image_set_id" in json_content:
                    filename = f"{json_content["image_set_id"]}.{json_content["sequence_id"]}"
                else: 
                    filename = str(np.random.randint(0,9223372036854775806))

                
                fits_filename = filename+".fits"
                json_filename = filename+".json"

                if annotT in self.annotation_to_fits:
                    fitsT = self.annotation_to_fits[annotT]
                    fits_content = self._download_fits(fitsT)
                else:
                    url = f"https://unifieddatalibrary.com/udl/skyimagery/getFile/{json_content["image_id"]}"
                    headers = {
                        "Authorization": f"Basic {UDL_KEY}",
                        "accept": "application/octet-stream"
                    }
                    with requests.get(url, headers=headers, stream=True) as response:
                        response.raise_for_status()  # Raises an exception for HTTP errors
                        file_location=os.path.join(download_directory, "temp_fits.fits")
                        with open(file_location, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:  # filter out keep-alive chunks
                                    f.write(chunk)
                    fits_content = fits.open(file_location)
                hdu = fits_content[0].header

                if "TRKMODE" in hdu.keys():
                    if hdu["TRKMODE"] != 'rate':
                        continue #This causes discrepancy between downloaded files and the local files. Only downloads rate tracked images
                
                sample_attributes, object_attributes = collect_stats(json_content, fits_content)
                counter.update([sample_attributes["dates"]])

                if sort_by_date:
                    #With Date Sorting
                    json_annotation_folder = os.path.join(download_directory,sample_attributes["dates"], "raw_annotation")
                    fits_annotation_folder = os.path.join(download_directory,sample_attributes["dates"], "raw_fits")
                    os.makedirs(json_annotation_folder, exist_ok=True)
                    os.makedirs(fits_annotation_folder, exist_ok=True)
                    annot_local_path = os.path.join(json_annotation_folder, json_filename)
                    fits_local_path = os.path.join(fits_annotation_folder, fits_filename)
                else:
                    #Without Date Sorting
                    json_annotation_folder = os.path.join(download_directory, "raw_annotation")
                    fits_annotation_folder = os.path.join(download_directory, "raw_fits")
                    os.makedirs(json_annotation_folder, exist_ok=True)
                    os.makedirs(fits_annotation_folder, exist_ok=True)
                    annot_local_path = os.path.join(json_annotation_folder, json_filename)
                    fits_local_path = os.path.join(fits_annotation_folder, fits_filename)

                sample_attributes["json_path"] = annot_local_path
                sample_attributes["fits_path"] = fits_local_path
                for dictionary in object_attributes:
                    dictionary["json_path"] = annot_local_path
                    dictionary["fits_path"] = fits_local_path

                if "image_set_id" in json_content:
                    if json_content["image_set_id"] not in db.collect_dict.keys():
                        db.collect_dict[json_content["image_set_id"]] = []
                        db.collect_dict[json_content["image_set_id"]].append({"json_path":annot_local_path,"fits_path":fits_local_path})
                    else:
                        db.collect_dict[json_content["image_set_id"]].append({"json_path":annot_local_path,"fits_path":fits_local_path})
                else: 
                    if "No_Collect" not in db.collect_dict.keys():
                        db.collect_dict["No_Collect"] = []
                        db.collect_dict["No_Collect"].append({"json_path":annot_local_path,"fits_path":fits_local_path})
                    else:
                        db.collect_dict["No_Collect"].append({"json_path":annot_local_path,"fits_path":fits_local_path})

                fits_content.writeto(fits_local_path, overwrite=True)
                with open(annot_local_path, "w") as f:
                    json.dump(json_content, f, indent=4)
            
                db.add_sample_attributes(sample_attributes)
                db.add_annotation_attributes(object_attributes)
            
            except ClientError:
                print(f"Key expired on date: {date}")
                break

            except UnboundLocalError:
                pass
                print(f"No JSON found: {annotT}")
        
            except Exception as e:
                print(f"Error processing {annotT}: {type(e).__name__}: {e}")

        write_count(os.path.join(download_directory, "count.txt"),len(db.annotation_attributes), len(db.sample_attributes),counter)

    def download_annotation_dates_calsats(self, date:str, download_directory:str, sort_by_date = True, limit=1000):
        """
        Downloads annotations and images from SILT or UDL. Given a annotation date, you can download all annotations for that day. 

        Args:
            date (str): Annotation date, you can find it when the s3 bucket is initialized
            download_directory (str): Local path to store downloaded data
            sort_by_date (bool): Saves files in date sorted folders according to collect date

        Returns:
            None
        """
        db = StatisticsFile()
        os.makedirs(download_directory, exist_ok=True)

        counter = Counter()

        date = datetime.strptime(date, "%Y-%m-%d").date()

        annotations = self.date_to_keystring[date]

        for annotT in tqdm(annotations, desc="Downloading and Collecting Statistics"):
        # for annotT,fitsT in tqdm(self.annotation_to_fits.items(), desc="Downloading and Collecting Statistics"):

            sample_attributes = {}
            object_attributes = []

            try:
                norad_id = os.path.basename(annotT).split(".")[0].split("sat_")[1]
                calsat = SPACECRAFT[norad_id]
            except: 
                continue


            try:
                json_content = self._download_annotation(annotT)

                if "image_set_id" in json_content:
                    filename = f"{json_content["image_set_id"]}.{json_content["sequence_id"]}"
                else: 
                    filename = str(np.random.randint(0,9223372036854775806))

                
                fits_filename = filename+".fits"
                json_filename = filename+".json"

                if annotT in self.annotation_to_fits:
                    fitsT = self.annotation_to_fits[annotT]
                    fits_content = self._download_fits(fitsT)
                else:
                    url = f"https://unifieddatalibrary.com/udl/skyimagery/getFile/{json_content["image_id"]}"
                    headers = {
                        "Authorization": f"Basic {UDL_KEY}",
                        "accept": "application/octet-stream"
                    }
                    with requests.get(url, headers=headers, stream=True) as response:
                        response.raise_for_status()  # Raises an exception for HTTP errors
                        file_location=os.path.join(download_directory, "temp_fits.fits")
                        with open(file_location, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:  # filter out keep-alive chunks
                                    f.write(chunk)
                    fits_content = fits.open(file_location)
                hdu = fits_content[0].header

                # if "TRKMODE" in hdu.keys():
                #     if hdu["TRKMODE"] != 'rate':
                        # print(f"Trackmode: {hdu["TRKMODE"]}")
                        # continue #This causes discrepancy between downloaded files and the local files. Only downloads rate tracked images
                
                sample_attributes, object_attributes = collect_stats(json_content, fits_content)
                counter.update([sample_attributes["dates"]])

                if sort_by_date:
                    #With Date Sorting
                    json_annotation_folder = os.path.join(download_directory,sample_attributes["dates"], "raw_annotation")
                    fits_annotation_folder = os.path.join(download_directory,sample_attributes["dates"], "raw_fits")
                    os.makedirs(json_annotation_folder, exist_ok=True)
                    os.makedirs(fits_annotation_folder, exist_ok=True)
                    annot_local_path = os.path.join(json_annotation_folder, json_filename)
                    fits_local_path = os.path.join(fits_annotation_folder, fits_filename)
                else:
                    #Without Date Sorting
                    json_annotation_folder = os.path.join(download_directory, "raw_annotation")
                    fits_annotation_folder = os.path.join(download_directory, "raw_fits")
                    os.makedirs(json_annotation_folder, exist_ok=True)
                    os.makedirs(fits_annotation_folder, exist_ok=True)
                    annot_local_path = os.path.join(json_annotation_folder, json_filename)
                    fits_local_path = os.path.join(fits_annotation_folder, fits_filename)

                sample_attributes["json_path"] = annot_local_path
                sample_attributes["fits_path"] = fits_local_path
                for dictionary in object_attributes:
                    dictionary["json_path"] = annot_local_path
                    dictionary["fits_path"] = fits_local_path

                if "image_set_id" in json_content:
                    if json_content["image_set_id"] not in db.collect_dict.keys():
                        db.collect_dict[json_content["image_set_id"]] = []
                        db.collect_dict[json_content["image_set_id"]].append({"json_path":annot_local_path,"fits_path":fits_local_path})
                    else:
                        db.collect_dict[json_content["image_set_id"]].append({"json_path":annot_local_path,"fits_path":fits_local_path})
                else: 
                    if "No_Collect" not in db.collect_dict.keys():
                        db.collect_dict["No_Collect"] = []
                        db.collect_dict["No_Collect"].append({"json_path":annot_local_path,"fits_path":fits_local_path})
                    else:
                        db.collect_dict["No_Collect"].append({"json_path":annot_local_path,"fits_path":fits_local_path})

                fits_content.writeto(fits_local_path, overwrite=True)
                with open(annot_local_path, "w") as f:
                    json.dump(json_content, f, indent=4)
            
                db.add_sample_attributes(sample_attributes)
                db.add_annotation_attributes(object_attributes)
            
            except ClientError:
                print(f"Key expired on date: {date}")
                break

            except UnboundLocalError:
                pass
                print(f"No JSON found: {annotT}")
        
            except Exception as e:
                print(f"Error processing {annotT}: {type(e).__name__}: {e}")

        write_count(os.path.join(download_directory, "count.txt"),len(db.annotation_attributes), len(db.sample_attributes),counter)

    def download_data(self, download_directory:str, sort_by_date = True):
        """
        Downloads matching annotations in the download directory. Does not pull from UDL, useful for data annotated before May 2025. 

        Args:
            download_directory (str): Local path to store downloaded data
            sort_by_date (bool): Saves files in date sorted folders according to collect date

        Returns:
            None
        """
        db = StatisticsFile()
        annotations_path = os.path.join(download_directory, "raw_annotation")
        fits_path = os.path.join(download_directory, "raw_fits")
        # if statistics_filename is None: statistics_filename=os.path.basename(download_directory)
        # statistics_filename = f"{statistics_filename}_Statistics.pkl"
        os.makedirs(download_directory, exist_ok=True)
        os.makedirs(annotations_path, exist_ok=True)
        os.makedirs(fits_path, exist_ok=True)

        counter = Counter()

        for annotT,fitsT in tqdm(self.annotation_to_fits.items(), desc="Downloading and Collecting Statistics"):
        # for annotT,fitsT in self.annotation_to_fits.items():
            sample_attributes = {}
            object_attributes = []
            try:
                json_content = self._download_annotation(annotT)
                fits_content = self._download_fits(fitsT)

                json_content = self._download_annotation(annotT)

                if "image_set_id" in json_content:
                    filename = f"{json_content["image_set_id"]}.{json_content["sequence_id"]}"
                else: 
                    filename = str(np.random.randint(0,9223372036854775806))
                fits_filename = filename+".fits"
                json_filename = filename+".json"

                hdu = fits_content[0].header

                if "TRKMODE" in hdu.keys():
                    if hdu["TRKMODE"] != 'rate':
                        print(f"Trackmode: {hdu["TRKMODE"]}") #This causes discrepancy between downloaded files and the local files. Only downloads rate tracked images
                
                sample_attributes, object_attributes = collect_stats(json_content, fits_content)
                counter.update([sample_attributes["dates"]])

                if sort_by_date:
                    #With Date Sorting
                    os.makedirs(os.path.join(fits_path,sample_attributes["dates"]), exist_ok=True)
                    os.makedirs(os.path.join(annotations_path,sample_attributes["dates"]), exist_ok=True)
                    fits_local_path = os.path.join(fits_path, sample_attributes["dates"], fits_filename)
                    annot_local_path = os.path.join(annotations_path, sample_attributes["dates"], json_filename)
                else:
                    #Without Date Sorting
                    fits_local_path = os.path.join(fits_path, fits_filename)
                    annot_local_path = os.path.join(annotations_path, json_filename)

                sample_attributes["json_path"] = annot_local_path
                sample_attributes["fits_path"] = fits_local_path
                for dictionary in object_attributes:
                    dictionary["json_path"] = annot_local_path
                    dictionary["fits_path"] = fits_local_path

                fits_content.writeto(fits_local_path, overwrite=True)
                with open(annot_local_path, "w") as f:
                    json.dump(json_content, f, indent=4)
            
                db.add_sample_attributes(sample_attributes)
                db.add_annotation_attributes(object_attributes)

            except Exception as e:
                print(f"Error processing {annotT}: {e}")

            # db.save(os.path.join(download_directory, statistics_filename))
        write_count(os.path.join(download_directory, "count.txt"),len(db.annotation_attributes), len(db.sample_attributes),counter)

    def download_UDL_data(self, download_directory:str, Authorization_key:str=UDL_KEY, sort_by_date = True):
        """
        Download unmatched data from the UDL. This does not sort

        Args:
            download_directory (str): Local path to store downloaded data
            sort_by_date (bool): Saves files in date sorted folders according to collect date

        Returns:
            None
        """

        db = StatisticsFile()
        annotations_path = os.path.join(download_directory, "raw_annotation")
        fits_path = os.path.join(download_directory, "raw_fits")
        os.makedirs(download_directory, exist_ok=True)
        os.makedirs(annotations_path, exist_ok=True)
        os.makedirs(fits_path, exist_ok=True)

        counter = Counter()
        
        for annot_name,annot_path in tqdm(self.unmatched_annotations.items(), desc="Downloading and Collecting Statistics"):
        # for annotT,fitsT in self.annotation_to_fits.items():
            sample_attributes = {}
            object_attributes = []
            try:
                json_content = self._download_annotation(annot_path)
                if "image_set_id" in json_content:
                    filename = f"{json_content["image_set_id"]}.{json_content["sequence_id"]}"
                else: 
                    filename = str(np.random.randint(0,9223372036854775806))
                fits_filename = filename+".fits"
                json_filename = filename+".json"

                url = f"https://unifieddatalibrary.com/udl/skyimagery/getFile/{json_content["image_id"]}"
                headers = {
                    "Authorization": f"Basic {Authorization_key}",
                    "accept": "application/octet-stream"
                }
                with requests.get(url, headers=headers, stream=True) as response:
                    response.raise_for_status()  # Raises an exception for HTTP errors
                    file_location=os.path.join(fits_path, fits_filename)
                    with open(file_location, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                fits_content = fits.open(file_location)
                


                hdu = fits_content[0].header

                if "TRKMODE" in hdu.keys():
                    if hdu["TRKMODE"] != 'rate':
                        os.remove(file_location)
                        continue #This causes discrepancy between downloaded files and the local files. Only downloads rate tracked images
                
                sample_attributes, object_attributes = collect_stats(json_content, fits_content)
                counter.update([sample_attributes["dates"]])


                if sort_by_date:
                    #With Date Sorting
                    os.makedirs(os.path.join(fits_path,sample_attributes["dates"]), exist_ok=True)
                    os.makedirs(os.path.join(annotations_path,sample_attributes["dates"]), exist_ok=True)
                    fits_local_path = os.path.join(fits_path, sample_attributes["dates"], fits_filename)
                    annot_local_path = os.path.join(annotations_path, sample_attributes["dates"], json_filename)
                else:
                    #Without Date Sorting
                    fits_local_path = os.path.join(fits_path, fits_filename)
                    annot_local_path = os.path.join(annotations_path, json_filename)

                sample_attributes["json_path"] = annot_local_path
                sample_attributes["fits_path"] = fits_local_path
                for dictionary in object_attributes:
                    dictionary["json_path"] = annot_local_path
                    dictionary["fits_path"] = fits_local_path

                with open(annot_local_path, "w") as f:
                    json.dump(json_content, f, indent=4)
            
                db.add_sample_attributes(sample_attributes)
                db.add_annotation_attributes(object_attributes)

            except TypeError as e:
                print(f"Error processing {annot_path}: {e}")

            # except Exception as e:
            #     print(f"Error processing {annot_path}: {e}")

        write_count(os.path.join(download_directory, "count.txt"),len(db.annotation_attributes), len(db.sample_attributes),counter)




if __name__ == "__main__":
    aws_directory = "third-party-data/PDS-RME01/Satellite/Annotations/"
    s3_client = S3Client(aws_directory)
    s3_client.download_annotation_dates("2025-10-14","/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Testing_Download/By_date_annotations" )
