import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import os
import mlflow
import json
import numpy as np
from matplotlib.lines import Line2D
from datetime import datetime
from scipy.optimize import curve_fit

def generate_color_gradient(hex1, hex2, n):
    """
    Generate a gradient of n colors between two hex colors (6 or 8 digit).
    Supports alpha channel if included in the hex (last two digits).
    
    Args:
        hex1 (str): Starting hex color, e.g. "#40B9DB" or "#40B9DB00".
        hex2 (str): Ending hex color, e.g. "#FFA21F" or "#FFA21FFF".
        n (int): Number of colors to generate.

    Returns:
        list of tuples: RGBA colors normalized to [0,1].
    """
    def hex_to_rgba(h):
        h = h.lstrip('#')
        if len(h) == 6:
            r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
            a = 255
        elif len(h) == 8:
            r, g, b, a = (int(h[i:i+2], 16) for i in (0, 2, 4, 6))
        else:
            raise ValueError("Hex color must be 6 or 8 digits")
        return r, g, b, a

    rgba1 = hex_to_rgba(hex1)
    rgba2 = hex_to_rgba(hex2)

    gradient = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        interp = tuple(
            (rgba1[j] + (rgba2[j] - rgba1[j]) * t) / 255.0
            for j in range(4)
        )
        gradient.append(interp)

    return gradient

def scaling_law(N, M_inf, A, alpha):
    return M_inf - A / (N**alpha)

class Single_Dataset_Study():
    def __init__(self, out_put_directory:str, testing_sets:list, title:str):
        self.testing_sets = testing_sets
        self.output_directory = out_put_directory
        self.basename = os.path.basename(out_put_directory)

        self.overal_metrics_path = os.path.join(self.output_directory, "overal_metrics.txt")
        self.figure_folder = os.path.join(self.output_directory, "figures")
        self.pickle_path = os.path.join(self.output_directory, f"{self.basename}_{title}.pkl")
        self.json_path = os.path.join(self.output_directory, f"{self.basename}_{title}_predictions.json")
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.figure_folder, exist_ok=True)
        self.cumulative_metrics = {}
        self.predictions = None
        self.GTs = None
        self.image_ids = None

    def save(self):
        """Save the object as a pickle file."""
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self, f)
    
    def save_json(self):
        detections = {"ground_truth":self.GTs,
                      "predictions":self.predictions}
        with open(self.json_path, "w") as f:
            json.dump(detections, f)

    @classmethod
    def load(self, directory:str):
        """Load an object from a pickle file."""
        pk = None
        for file in os.listdir(directory):
            if file.endswith(".pkl"):
                pk = file
        pkl_path = os.path.join(directory, pk)
        with open(pkl_path, 'rb') as f:
            object = pickle.load(f)
            object.pickle_path = pkl_path
            object.output_directory = os.path.dirname(pkl_path)
            object.figure_folder = os.path.join(object.output_directory, "figures")
            return object

    def add_metrics(self, dictionary:dict):
        assert isinstance(dictionary, dict)
        self.cumulative_metrics.update(dictionary)

    def add_predictions(self, predictions:list):
        self.predictions = predictions

    def add_gts(self, ground_truth:list):
        self.GTs = ground_truth

    def add_image_ids(self, ids:list):
        self.image_ids = ids

    def plot_PR_Curves(self, save_path=None, plot_format:str="pdf", best_precision=0.99):
        plt.figure(figsize=(12,8)) 
        for conf_index, confidence_thresh in enumerate(self.cumulative_metrics["PR_Curve_Confidence"]):
            recall = self.cumulative_metrics["PR_Curve_Recall"][conf_index,:]
            precision = self.cumulative_metrics["PR_Curve_Precision"][conf_index,:]
            plt.plot(recall, precision, 'o-', label=f'Tc: {confidence_thresh}')

        # Labels and title
        if save_path is None:
            save_name = os.path.join(self.figure_folder, f"PR_Curve_Confidence-{self.basename}.{plot_format}")
        else:
            save_name = os.path.join(save_path, f"PR_Curve_Confidence-{self.basename}.{plot_format}")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.legend(loc='lower left')
        plt.savefig(save_name)
        plt.close()

        best_f1_table = "Tf & Tc* & F1*\\\\\n\\hline\n"
        best_precision_table = "Tf & Tc & Precision & Recall & F1\\\\\n\\hline\n"

        plt.figure(figsize=(12,8)) 
        plt.axhline(y=0.99, linestyle='--', color='red', linewidth=1.5)
        plt.axvline(x=0.9, linestyle='--', color='red', linewidth=1.5)
        for fit_index, fit_thresh in enumerate(self.cumulative_metrics["PR_Curve_Fit"]):
            recall = self.cumulative_metrics["PR_Curve_Recall"][:,fit_index]
            precision = self.cumulative_metrics["PR_Curve_Precision"][:,fit_index]
            F1 = [2*recall[i]*precision[i]/(recall[i]+precision[i]+.01) for i in range(len(recall))]
            best_f1 = np.argmax(F1)

            best_f1_table += f"{fit_thresh} & {self.cumulative_metrics["PR_Curve_Confidence"][best_f1]:.2f} &{F1[best_f1]:.2f}\\\\\n"
            for pindex,p in enumerate(precision):
                if p>best_precision:
                    best_precision_table+= f"{fit_thresh} & {self.cumulative_metrics["PR_Curve_Confidence"][pindex]:.2f} & {precision[pindex]:.2f} & {recall[pindex]:.2f} & {F1[pindex]:.2f}\\\\\n"
    
            plt.plot(recall, precision, 'o-', label=f'Tf: {fit_thresh}, F1*={F1[best_f1]:.2f}') 

        # Labels and title
                # Labels and title
        if save_path is None:
            save_name = os.path.join(self.figure_folder, f"PR_Curve_Fit-{self.basename}.{plot_format}")
            with open(os.path.join(self.figure_folder,"best_results"), "w") as f:
                f.write(best_f1_table)
                f.write("\n")
                f.write(best_precision_table)
        else:
            save_name = os.path.join(save_path, f"PR_Curve_Fit-{self.basename}.{plot_format}")
            with open(os.path.join(save_path,"best_results"), "w") as f:
                f.write(best_f1_table)
                f.write("\n")
                f.write(best_precision_table)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.legend(loc='upper left')
        plt.savefig(save_name)
        plt.close()

    def plot_per_attribute_PR(self, attribute:str, n_bins:int=20, plot_format:str="pdf", log_y=False):
        annotations_by_image = self.cumulative_metrics["original_tgt_attributes"]

        attribute_list = []
        tp_list = []
        fp_list = []
        fn_list = []

        # average_annotation_attributes = [sum(annot[attribute] for annot in im)/len(im) if len(im) > 0 else -1 for im in annotations_by_image]
        for index,image_gts in enumerate(annotations_by_image):
            if len(image_gts)>0:
                attribute_list.append(sum(annot[attribute] for annot in image_gts)/len(image_gts))
                tp_list.append(self.cumulative_metrics["True_Positives"][index])
                fp_list.append(self.cumulative_metrics["False_Positives"][index])
                fn_list.append(self.cumulative_metrics["False_Negatives"][index])
            else: #Excluding images with no targets in them
                pass

        if log_y:
            attrs = np.log(np.absolute(np.array(attribute_list)))
        else:
            attrs = np.array(attribute_list)
        tp_list = np.array(tp_list)
        fp_list = np.array(fp_list)
        fn_list = np.array(fn_list)

        # Bin edges and bin indices
        bin_edges = np.linspace(attrs.min(), attrs.max(), n_bins + 1)
        bin_indices = np.digitize(attrs, bins=bin_edges, right=False) - 1

        # Store metrics per bin
        precision_bins = []
        recall_bins = []
        f1_bins = []
        bin_centers = []

        for i in range(n_bins):
            bin_mask = bin_indices == i
            tp = tp_list[bin_mask].sum()
            fp = fp_list[bin_mask].sum()
            fn = fn_list[bin_mask].sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precision_bins.append(precision)
            recall_bins.append(recall)
            f1_bins.append(f1)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Histogram
        counts, _, _ = ax1.hist(attrs, bins=bin_edges, alpha=0.4, label=f'{attribute} Histogram'.title(), color='gray')
        if log_y:
            ax1.set_xlabel(f"Log10({attribute})", color='gray')
        else:
            ax1.set_xlabel(f'{attribute}'.title())
        ax1.set_ylabel("Count", color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # Metrics on second axis
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, precision_bins, label='Precision', color='blue', marker='o')
        ax2.plot(bin_centers, recall_bins, label='Recall', color='red', marker='s')
        ax2.plot(bin_centers, f1_bins, label='F1 Score', color='green', marker='^')
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.05)

        # Legends and layout
        fig.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title("Precision, Recall, and F1 Score vs SNR")
        plt.tight_layout()

        save_path = os.path.join(self.figure_folder, f"PR_vs_{attribute}-{self.basename}.{plot_format}")
        plt.savefig(save_path)

class Multi_Dataset_Study():
    def __init__(self, paths:list, color_legend:list, second_characteristic:list, save_path:str, save_type:str="pdf", second_char_type="line",color_gradient=("#40B9DB", "#FFA21F"), dataset_sizes:list=None):
        self.paths = paths
        self.color_legend = color_legend
        self.shape_legend = second_characteristic
        self.line_legend = second_characteristic
        self.opacity_legend = second_characteristic
        self.dataset_studies = []
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.save_type = save_type
        self.basename = os.path.basename(save_path)
        self.second_char_type = second_char_type
        self.dataset_sizes = dataset_sizes

        for e in paths:
            dataset_study:Single_Dataset_Study = Single_Dataset_Study.load(e)
            self.dataset_studies.append(dataset_study)

        temp_dict = {}
        color_list = []
        for color in self.color_legend:
            if not color in temp_dict:
                color_list.append(color)
                temp_dict[color] =  1
        temp_dict = {}
        line_list = []
        for line in self.line_legend:
            if not line in temp_dict:
                line_list.append(line)
                temp_dict[line] =  1
        temp_dict = {}
        shape_list = []
        for shape in self.shape_legend:
            if not shape in temp_dict:
                shape_list.append(shape)
                temp_dict[shape] =  1
        temp_dict = {}
        opacity_list = []
        for opacity in self.opacity_legend:
            if not opacity in temp_dict:
                opacity_list.append(opacity)
                temp_dict[opacity] =  1

        self.color_list = color_list
        self.shape_list = shape_list
        self.line_list = line_list
        self.opacity_list = opacity_list

        self.matplotlib_markers = ['o',  's',  '^',  '<',  '>',  'D',  '*',  'x', '+']
        self.matplotlib_lines = ["-", "--", "-.", ":"]
        self.matplotlib_opacity = [v for v in np.linspace(0.2, 1.0, len(self.opacity_list))]
        if color_gradient is None:
            self.matplotlib_colors  = [
            (0.1216, 0.4667, 0.7059),
            (1.0000, 0.4980, 0.0549),
            (0.1725, 0.6275, 0.1725),
            (0.8392, 0.1529, 0.1569),
            (0.5804, 0.4039, 0.7412),
            (0.5490, 0.3373, 0.2941),
            (0.8902, 0.4667, 0.7608),
            (0.4980, 0.4980, 0.4980),
            (0.7373, 0.7412, 0.1333),
            (0.0902, 0.7451, 0.8118),
            'red',
            'blue',
            'green',
            'cyan',
            'magenta',
            'yellow',
            'black',
            'orange',
            'purple',
            'darkgreen'
        ]
        else:
            self.matplotlib_colors =generate_color_gradient(color_gradient[0], color_gradient[1], len(self.color_list))



                # Build legends
        self.color_handles = [
            Line2D([0], [0], color=self.matplotlib_colors[c], markersize=10, label=self.color_list[c])
            for c in range(len(self.color_list))
        ]
        self.shape_handles = [
            Line2D([0], [0], marker=self.matplotlib_markers[m], color='k', linestyle='None', markersize=10, label=self.shape_list[m])
            for m in range(len(self.shape_list))
        ]
        self.line_handles = [
            Line2D([0], [0], color="black", linestyle=self.matplotlib_lines[ls], linewidth=3, label=self.line_list[ls])
            for ls in range(len(self.line_list))
        ]
        self.opacity_handles = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=3, label=self.opacity_list[ls], alpha=self.matplotlib_opacity[ls])
            for ls in range(len(self.opacity_list))
        ]

    def combine_metric_plots(self, metrics:list):
        plt.clf() 
        plt.close('all')
        plt.figure(figsize=(7,6)) 
        save_path = os.path.join(self.save_path, f"combined_metrics.{self.save_type}")
        used_shapes = {}
        used_colors = {}

        large_range= False

        min_value = 1

        for shape_index, df in enumerate(self.dataframes):
            for color_index, metric in enumerate(metrics):
                color = self.matplotlib_colors[color_index]
                shape = self.matplotlib_markers[shape_index]
                
                df = df.sort_values(by="date")
                df[metric]
                try: df[metric]
                except KeyError: continue
                if isinstance(df[metric], pd.Series): continue
                if metric != "date":
                    if df[metric].max() >1:
                        large_range=True
                bruh = self.color_legend[shape_index]
                if "Panoptic" in self.color_legend[shape_index] and not 'F1' in metric:
                    continue

                min_value = min(min_value, df[metric].min() )
                days_since_start = (df["date"]-df["date"].min()).dt.days
                plt.plot(days_since_start, df[metric], marker=shape, linestyle="-", color=color, markersize=8, markerfacecolor=color, markeredgecolor='none', alpha=.75)

                used_colors[color_index] = Line2D([-1], [-1], color=color, lw=1, label=metric)
                used_shapes[shape_index] = Line2D([-1], [-1], marker=shape, color='black', linestyle='None', label=self.color_legend[shape_index])
    
        # Draw both legends separately
        key = []
        for k,value in used_colors.items():
            key.append(value)
        for k,value in used_shapes.items():
            key.append(value)
        plt.legend(handles=key, loc='lower right')

        if large_range:
            plt.ylim(bottom=0)
        else:
            plt.ylim(bottom=min_value, top=1)
        # plt.xticks([])
        # plt.xlim(bottom=0)
        # plt.xticks(df["date"].tolist(), rotation=45)
        plt.xlabel("Days Elapsed Since Training")
        plt.ylabel(f"Metric")
        # plt.title(f"{metric.replace("_"," ")}")
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.close()

    def plot_metrics_over_time(self):
        for key in self.dataframes[0].columns:
            self.plot_metric(key)
        
    def plot_metric(self, metric:str):
        plt.clf() 
        plt.close('all')
        plt.figure(figsize=(5,4,)) 
        save_path = os.path.join(self.save_path, f"{metric}.{self.save_type}")
        used_shapes = {}
        used_colors = {}

        large_range= False

        for index, df in enumerate(self.dataframes):
            color_label = self.color_legend[index]
            shape_label = self.shape_legend[index]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]
            
            df = df.sort_values(by="date")
            try: df[metric]
            except KeyError: continue
            if metric != "date":
                if df[metric].max() >1:
                    large_range=True

            days_since_start = (df["date"]-df["date"].min()).dt.days
            plt.plot(days_since_start, df[metric], marker=shape, linestyle="-", color=color, markersize=8, markerfacecolor=color, markeredgecolor='none', alpha=.75)

            used_colors[color_index] = Line2D([-1], [-1], color=color, lw=1, label=color_label)
            used_shapes[shape_index] = Line2D([-1], [-1], marker=shape, color='black', linestyle='None', label=shape_label)

    
        # Draw both legends separately
        key = []
        for k,value in used_colors.items():
            key.append(value)
        for k,value in used_shapes.items():
            key.append(value)
        plt.legend(handles=key, loc='lower right')

        if large_range:
            plt.ylim(bottom=0)
        else:
            plt.ylim(bottom=0, top=1)
        # plt.xticks([])
        # plt.xlim(bottom=0)
        # plt.xticks(df["date"].tolist(), rotation=45)
        plt.xlabel("Days Elapsed Since Training")
        plt.ylabel(f"{metric.replace("_"," ")}")
        # plt.title(f"{metric.replace("_"," ")}")
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.close()

    def print_metric_avgs(self):
        for index, df in enumerate(self.dataframes):
            print()
            print(f"Model: {self.color_legend[index]}")
            print(f"Dataset: {self.shape_legend[index]}")
            self.print_metric_avg(df)
            
    def print_metric_avg(self, dataframe:pd.DataFrame):
        for key in dataframe.columns:
            if key == "date": 
                continue
            if isinstance(dataframe[key], pd.Series): 
                continue
            bruh = dataframe[key]
            average = dataframe[key].mean()
            std = dataframe[key].std()
            minn = dataframe[key].min()
            maxx = dataframe[key].max()
            print(f"\t{key}")
            print(f"\t\tAVG: {average}")
            print(f"\t\tSTD: {std}")
            print(f"\t\tMIN: {minn}")
            print(f"\t\tMAX: {maxx}")

    def print_f1star(self, tfit:float):
        overleaf_string = []
        for j,df in enumerate(self.dataset_studies):
            fit_index = df.cumulative_metrics["PR_Curve_Fit"].index(tfit)
            recall = df.cumulative_metrics["PR_Curve_Recall"][:,fit_index]
            precision = df.cumulative_metrics["PR_Curve_Precision"][:,fit_index]
            best_p = 0
            best_r = 0
            best_f1 = 0
            for r,p in zip(recall, precision):
                temp_f1 = 2*r*p/(r+p)
                if temp_f1 > best_f1:
                    best_p = p
                    best_r = r
                    best_f1 = temp_f1
            print(f"{df.basename}, F1* = {best_f1:.2f}, P* = {best_p:.2f}, R* = {best_r:.2f} ")
            overleaf_string.append(f"{df.basename}&{tfit}&{best_f1:.2f}&{best_p:.2f}&{best_r:.2f}\\\\")
        print()
        for overleaf_row in overleaf_string:
            print(overleaf_row)

    def plot_combined_per_attribute_PR(self, attribute:str, n_bins=20, curve:str="recall", log_x:bool=False,  save_type:str="pdf", fig_size:tuple=(8,6)):
        fig, ax1 = plt.subplots(figsize=fig_size)

        attribute_name = attribute.replace("local_","")
        attribute_name = attribute.replace("snr","SNR")

        for j,df in enumerate(self.dataset_studies):
            color_label = self.color_legend[j]
            shape_label = self.shape_legend[j]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]

            ax1.set_xlabel(f"{attribute_name}")
            attrs = np.array([np.average([j[attribute] for j in entry]) for entry in df.cumulative_metrics["original_tgt_attributes"]])
            # attrs = np.array(df.cumulative_metrics[attribute])
            if log_x:
                ax1.set_xlabel(f"Log({attribute_name})")
                attrs[attrs < 0] = 0 
                attrs = np.log1p(attrs)
            tp_list = np.array(df.cumulative_metrics["True_Positives"])
            fp_list = np.array(df.cumulative_metrics["False_Positives"])
            fn_list = np.array(df.cumulative_metrics["False_Negatives"])


            # Bin edges and bin indices
            bin_edges = np.linspace(attrs.min(), attrs.max(), n_bins + 1)
            bin_indices = np.digitize(attrs, bins=bin_edges, right=False) - 1

            # Store metrics per bin
            precision_bins = []
            recall_bins = []
            f1_bins = []
            bin_centers = []

            for i in range(n_bins):
                bin_mask = bin_indices == i
                tp = tp_list[bin_mask].sum()
                fp = fp_list[bin_mask].sum()
                fn = fn_list[bin_mask].sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                precision_bins.append(precision)
                recall_bins.append(recall)
                f1_bins.append(f1)
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            counts, _, _ = ax1.hist(attrs, bins=bin_edges, alpha=0.4, color='gray')
            ax1.set_ylabel("Count", color='gray')
            ax1.tick_params(axis='y', labelcolor='gray')
            # Metrics on second axis
            ax2 = ax1.twinx()
            if "precision" in curve.lower():
                plt.title("Precision")
                ax2.plot(bin_centers, precision_bins, label=f'{color_label}', color=color, marker=shape)
            if "recall" in curve.lower():
                plt.title("Recall")
                ax2.plot(bin_centers, recall_bins, label=f'{color_label}', color=color, marker=shape)
            if "f1" in curve.lower():
                plt.title("F1 Score vs SNR")
                ax2.plot(bin_centers, f1_bins, label=f'{color_label}', color=color, marker=shape)
            ax2.set_ylabel("Score")
            ax2.set_ylim(0, 1.05)

        # Legends and layout
        fig.legend(loc='lower right', bbox_to_anchor=(1, 0), bbox_transform=ax1.transAxes)
        # fig.legend(loc="lower right", bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(self.save_path, f"{curve}_vs_{attribute}-{self.basename}.{save_type}")
        plt.savefig(save_path)

    def plot_combined_PR_Curves(self, threshold_fit = 1, threshold_confidence = 0.5, split_legend=False, save_type:str="pdf", fig_size:tuple=(8,6), shape_size=1):
        fig, ax1 = plt.subplots(figsize=fig_size)

        # for j,df in enumerate(self.dataset_studies):
        #     df.plot_PR_Curves(self.save_path)

        for j,df in enumerate(self.dataset_studies):
            color_label = self.color_legend[j]
            shape_label = self.shape_legend[j]
            line_label = self.line_legend[j]
            opacity_label = self.opacity_legend[j]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            line_index = self.line_list.index(line_label)
            opacity_index = self.opacity_list.index(opacity_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]
            line = self.matplotlib_lines[line_index]
            opacity = self.matplotlib_opacity[opacity_index]

            confidence_index = df.cumulative_metrics["PR_Curve_Confidence"].index(threshold_confidence)

            recall = df.cumulative_metrics["PR_Curve_Recall"][confidence_index,:]
            precision = df.cumulative_metrics["PR_Curve_Precision"][confidence_index,:]
                
            # plt.plot(recall, precision, 'o-', label=f'{color_label}', color=color, marker=shape)
            if self.second_char_type == "shape":
                plt.plot(recall, precision, '-', label=f'{color_label}-{shape_label}', marker=shape, color=color, markersize=shape_size) 
            if self.second_char_type == "line":
                plt.plot(recall, precision, linestyle=line, label=f'{color_label}-{line_label}', color=color)  
            if self.second_char_type == "opacity":
                plt.plot(recall, precision, linestyle="-", label=f'{color_label}-{opacity_label}', color=color, alpha = opacity) 


        if split_legend:
            if self.second_char_type == "shape":
                if len(self.shape_handles)<= 1:
                    all_handles = self.color_handles
                else:
                    all_handles = self.color_handles + self.shape_handles 
            if self.second_char_type == "line":
                if len(self.line_handles)<= 1:
                    all_handles = self.color_handles
                else:
                    all_handles = self.color_handles + self.line_handles 
            if self.second_char_type == "opacity":
                if len(self.opacity_handles)<= 1:
                    all_handles = self.color_handles
                else:
                    all_handles = self.color_handles + self.opacity_handles 
            legend1 = plt.legend(handles=all_handles, loc="upper left")
            plt.gca().add_artist(legend1)
        else:
            plt.legend(loc='upper left')


        # Labels and title
        save_path = os.path.join(self.save_path, f"PR_Curve_Confidence-{self.basename}.{save_type}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve Tc = {threshold_confidence}')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        # plt.legend(loc='upper left')
        plt.savefig(save_path)
        plt.close()

        plt.figure(figsize=fig_size) 
        for j,df in enumerate(self.dataset_studies):
            color_label = self.color_legend[j]
            shape_label = self.shape_legend[j]
            line_label = self.line_legend[j]
            opacity_label = self.opacity_legend[j]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            line_index = self.line_list.index(line_label)
            opacity_index = self.opacity_list.index(opacity_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]
            line = self.matplotlib_lines[line_index]
            opacity = self.matplotlib_opacity[opacity_index]


            fit_index = df.cumulative_metrics["PR_Curve_Fit"].index(threshold_fit)

            recall = df.cumulative_metrics["PR_Curve_Recall"][:,fit_index]
            precision = df.cumulative_metrics["PR_Curve_Precision"][:,fit_index]
                
            # plt.plot(recall, precision, 'o-', label=f'{color_label}', color=color, marker=shape)
            if self.second_char_type == "shape":
                plt.plot(recall, precision, '-', label=f'{color_label}-{shape_label}', marker=shape, color=color, markersize=shape_size) 
            if self.second_char_type == "line":
                plt.plot(recall, precision, linestyle=line, label=f'{color_label}-{line_label}', color=color)  
            if self.second_char_type == "opacity":
                plt.plot(recall, precision, linestyle="-", label=f'{color_label}-{opacity_label}', color=color, alpha = opacity) 

        if split_legend:
            if self.second_char_type == "shape":
                if len(self.shape_handles)<= 1:
                    all_handles = self.color_handles
                else:
                    all_handles = self.color_handles + self.shape_handles 
            if self.second_char_type == "line":
                if len(self.line_handles)<= 1:
                    all_handles = self.color_handles
                else:
                    all_handles = self.color_handles + self.line_handles 
            if self.second_char_type == "opacity":
                if len(self.opacity_handles)<= 1:
                    all_handles = self.color_handles
                else:
                    all_handles = self.color_handles + self.opacity_handles 
            legend1 = plt.legend(handles=all_handles, loc="upper left")
            plt.gca().add_artist(legend1)
        else:
            plt.legend(loc='upper left')
        # Labels and title
        save_path = os.path.join(self.save_path, f"PR_Curve_Fit-{self.basename}.{save_type}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve Tf = {threshold_fit}')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.savefig(save_path)
        plt.close()

    def calculate_scaling_law(self, tfit=1, tconf=0.5, save_type:str="pdf", fig_size:tuple=(8,6), shape_size=1, shape_color="#0199F1", fit_color="#F38B0D"):
        plt.figure(figsize=fig_size) 
        precisions = []
        recalls = []
        f1s = []
        dataset_sizes = []
        titles = ["Precision", "Recall", "F1"]

        for j,df in enumerate(self.dataset_studies):
            confidence_index = df.cumulative_metrics["PR_Curve_Confidence"].index(tconf)
            fit_index = df.cumulative_metrics["PR_Curve_Fit"].index(tfit)
            recall = df.cumulative_metrics["PR_Curve_Recall"][confidence_index,fit_index]
            precision = df.cumulative_metrics["PR_Curve_Precision"][confidence_index,fit_index]
            F1 = 2*recall*precision/(recall+precision)
            dataset_size = self.dataset_sizes[j]

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(F1)
            dataset_sizes.append(dataset_size)

        for k,l in enumerate([precisions, recalls, f1s]):
            initial_guess = [0.5, 1.0, 0.1]  # [M_inf, A, alpha]
            popt, pcov = curve_fit(scaling_law, dataset_sizes, l, p0=initial_guess, maxfev=10000)
            M_inf, A, alpha = popt

            N_fit = np.logspace(np.log10(min(dataset_sizes)), np.log10(max(dataset_sizes)*10), 200)
            M_fit = scaling_law(N_fit, *popt)

            plt.title(f"{titles[k]} Scaling Law Tf = {tfit}, Tc = {tconf}")
            plt.plot(N_fit, M_fit, label=f"Fit: M_inf={M_inf:.2f}, α={alpha:.2f}", color=fit_color)
            plt.scatter(dataset_sizes, l, marker="D", color=shape_color)
            plt.xscale("log")
            plt.xlabel("Dataset size (N)")
            plt.ylabel("Performance")
            plt.legend()
            
            print(f"{titles[k]} Scaling Law: M_inf={M_inf}, alpha={alpha}")

            save_path = os.path.join(self.save_path, f"ScalingLaw-{titles[k]}-{self.basename}.{save_type}")
            plt.savefig(save_path)
            plt.close()


        





class Dataset_study():
    def __init__(self, out_put_directory:str, testing_sets:list, title:str):
        self.dataframe = pd.DataFrame()
        self.path_to_date = {}
        self.testing_sets = testing_sets
        self.basename = "N/A"
        self.index = 0
        for path in testing_sets:
            match = re.search(r'\d{4}-\d{2}-\d{2}', path)
            if match is not None:
                self.path_to_date[path] = match.group()
                self.basename = os.path.basename(path.split(match.group())[0])
            else:
                date_string = "2024-08-19"
                self.path_to_date[path] = datetime.strptime(date_string, "%Y-%m-%d")
                self.basename = "NoDateDataset"

        self.output_directory = out_put_directory
        self.overal__metrics_path = os.path.join(self.output_directory, "overal_metrics.txt")
        self.figure_folder = os.path.join(self.output_directory, "figures")
        self.pickle_path = os.path.join(self.output_directory, f"{self.basename}_{title}.pkl")
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.figure_folder, exist_ok=True)
        self.cumulative_metrics = {}

    def save(self):
        """Save the object as a pickle file."""
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename:str):
        """Load an object from a pickle file."""
        with open(filename, 'rb') as f:
            object = pickle.load(f)
            object.pickle_path = filename
            object.output_directory = os.path.dirname(filename)
            object.figure_folder = os.path.join(object.output_directory, "figures")
            return object
        
    def __iter__(self):
        return self  # the iterator is the object itself

    def __next__(self):
        if self.index >= len(self.testing_sets):
            self.index=0
            raise StopIteration
        item = self.testing_sets[self.index]
        self.index += 1
        return item
    
    def __len__(self):
        return len(self.testing_sets)
    
    def get_date(self, path:str):
        return self.path_to_date[path]

    def add_metric(self, dictionary:dict):
        assert "date" in dictionary
        temp_dataframe = pd.DataFrame([dictionary])
        temp_dataframe["date"] = pd.to_datetime(temp_dataframe["date"])
        self.dataframe = pd.concat([self.dataframe,temp_dataframe ], ignore_index=True)
        self.dataframe = self.dataframe.sort_values(by="date")

    def print(self):
        print(self.dataframe)

    def plot_metric(self, metric:str):
        self.dataframe = self.dataframe.sort_values(by="date")
        save_path = os.path.join(self.figure_folder, f"{metric}-{self.basename}.png")
        mean = np.mean(self.dataframe[metric])
        std = np.std(self.dataframe[metric])
        plt.clf() 
        plt.close('all')
        plt.figure(figsize=(8,9)) 
        plt.axhline(mean, color='red', linestyle='-', label=f'Mean: {mean}',  alpha=0.5)          # Solid red line
        plt.axhline(mean + std, color='red', linestyle='--', label=f'STD: {std}',  alpha=0.5)  # Dashed red line
        plt.axhline(mean - std, color='red', linestyle='--', alpha=0.5)  # Dashed red line
        plt.plot(self.dataframe["date"], self.dataframe[metric], color="teal", label=f"{metric}")

        if metric != "date":
            if np.max(self.dataframe[metric]) < 1.0:
                plt.ylim(0,1)

        plt.legend(loc='lower right')
        plt.xticks(self.dataframe["date"].tolist(), rotation=45)
        plt.xlabel("Dataset Date")
        plt.ylabel(f"{metric}")
        plt.title(f"{self.basename}: {metric} vs Date")
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
        plt.close()
        
    def plot_all_metrics(self):
        for key in self.dataframe.columns:
            if "PR_Curve" in key: continue
            self.plot_metric(key)
        self.plot_PR_Curves()

    def plot_PR_Curves(self):
        # {"PR_Curve_Precision": precisions, "PR_Curve_Recall":recalls, "PR_Curve_F1":f1s, "PR_Curve_Fit":fit_thresholds, "PR_Curve_Confidence":conf_thresholds}

        plt.figure(figsize=(6,4)) 
        for conf_index, confidence_thresh in enumerate(self.dataframe["PR_Curve_Confidence"][0]):
            recall = self.dataframe["PR_Curve_Recall"][0][conf_index,:]
            precision = self.dataframe["PR_Curve_Precision"][0][conf_index,:]
            
            plt.plot(recall, precision, 'o-', label=f'Tc: {confidence_thresh}')

        # Labels and title
        save_path = os.path.join(self.figure_folder, f"PR_Curve_Confidence-{self.basename}.pdf")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.legend(loc='lower left')
        plt.savefig(save_path)
        plt.close()

        plt.figure(figsize=(8,6)) 
        for fit_index, fit_thresh in enumerate(self.cumulative_metrics["PR_Curve_Fit"]):
            recall = self.cumulative_metrics["PR_Curve_Recall"][:,fit_index]
            precision = self.cumulative_metrics["PR_Curve_Precision"][:,fit_index]
            
            plt.plot(recall, precision, 'o-', label=f'Tf: {fit_thresh}') 

        # Labels and title
        save_path = os.path.join(self.figure_folder, f"PR_Curve_Fit-{self.basename}.pdf")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.legend(loc='lower left')
        plt.savefig(save_path)
        plt.close()

    def plot_per_attribute_PR(self, attribute:str, n_bins:int=20):
        attrs = np.array(self.cumulative_metrics[attribute])
        tp_list = np.array(self.cumulative_metrics["True_Positives"])
        fp_list = np.array(self.cumulative_metrics["False_Positives"])
        fn_list = np.array(self.cumulative_metrics["False_Negatives"])


        # Bin edges and bin indices
        bin_edges = np.linspace(attrs.min(), attrs.max(), n_bins + 1)
        bin_indices = np.digitize(attrs, bins=bin_edges, right=False) - 1

        # Store metrics per bin
        precision_bins = []
        recall_bins = []
        f1_bins = []
        bin_centers = []

        for i in range(n_bins):
            bin_mask = bin_indices == i
            tp = tp_list[bin_mask].sum()
            fp = fp_list[bin_mask].sum()
            fn = fn_list[bin_mask].sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precision_bins.append(precision)
            recall_bins.append(recall)
            f1_bins.append(f1)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Histogram
        counts, _, _ = ax1.hist(attrs, bins=bin_edges, alpha=0.4, label='SNR Histogram', color='gray')
        ax1.set_xlabel("SNR")
        ax1.set_ylabel("Count", color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # Metrics on second axis
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, precision_bins, label='Precision', color='blue', marker='o')
        ax2.plot(bin_centers, recall_bins, label='Recall', color='red', marker='s')
        ax2.plot(bin_centers, f1_bins, label='F1 Score', color='green', marker='^')
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.05)

        # Legends and layout
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title("Precision, Recall, and F1 Score vs SNR")
        plt.tight_layout()

        save_path = os.path.join(self.figure_folder, f"PR_vs_{attribute}-{self.basename}.pdf")
        plt.savefig(save_path)

class CompiledExperiments():
    def __init__(self, paths:list, color_legend:list, shape_legend:list, save_path:str, save_type:str="pdf"):
        self.paths = paths
        self.color_legend = color_legend
        self.shape_legend = shape_legend
        self.dataframes = []
        self.studies = []
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.save_type = save_type
        self.basename = os.path.basename(save_path)

        self.matplotlib_colors  = [
            (0.1216, 0.4667, 0.7059),
            (1.0000, 0.4980, 0.0549),
            (0.1725, 0.6275, 0.1725),
            (0.8392, 0.1529, 0.1569),
            (0.5804, 0.4039, 0.7412),
            (0.5490, 0.3373, 0.2941),
            (0.8902, 0.4667, 0.7608),
            (0.4980, 0.4980, 0.4980),
            (0.7373, 0.7412, 0.1333),
            (0.0902, 0.7451, 0.8118),
            'red',
            'blue',
            'green',
            'cyan',
            'magenta',
            'yellow',
            'black',
            'orange',
            'purple',
            'darkgreen'
        ]
 

        # self.matplotlib_colors = ['red','blue','green','cyan','magenta','yellow','black','orange','purple','darkgreen']
        self.matplotlib_markers = ['o',  's',  '^',  '<',  '>',  'D',  '*',  'x', '+']
        # self.matplotlib_markers = ['s',  '^',  '<',  '>',  'D',  '*',  'x', '+']
        # self.matplotlib_markers = ['^',  '0<',  '>',  'D',  '*',  'x', '+']


        temp_dict = {}
        color_list = []
        for color in self.color_legend:
            if not color in temp_dict:
                color_list.append(color)
                temp_dict[color] =  1
        temp_dict = {}
        shape_list = []
        for shape in self.shape_legend:
            if not shape in temp_dict:
                shape_list.append(shape)
                temp_dict[shape] =  1

        self.color_list = color_list
        self.shape_list = shape_list


        for file_path in paths:
            files = os.listdir(file_path)
            filename = [file for file in files if file.endswith('.pkl')]
            if len(filename) > 0:
                filename = filename[0]
            else: continue
            study = Dataset_study.load(os.path.join(file_path, filename))
            self.studies.append(study)
            self.dataframes.append(study.dataframe)

    def combine_metric_plots(self, metrics:list):
        plt.clf() 
        plt.close('all')
        plt.figure(figsize=(7,6)) 
        save_path = os.path.join(self.save_path, f"combined_metrics.{self.save_type}")
        used_shapes = {}
        used_colors = {}

        large_range= False

        min_value = 1

        for shape_index, df in enumerate(self.dataframes):
            for color_index, metric in enumerate(metrics):
                color = self.matplotlib_colors[color_index]
                shape = self.matplotlib_markers[shape_index]
                
                df = df.sort_values(by="date")
                df[metric]
                try: df[metric]
                except KeyError: continue
                if isinstance(df[metric], pd.Series): continue
                if metric != "date":
                    if df[metric].max() >1:
                        large_range=True
                bruh = self.color_legend[shape_index]
                if "Panoptic" in self.color_legend[shape_index] and not 'F1' in metric:
                    continue

                min_value = min(min_value, df[metric].min() )
                days_since_start = (df["date"]-df["date"].min()).dt.days
                plt.plot(days_since_start, df[metric], marker=shape, linestyle="-", color=color, markersize=8, markerfacecolor=color, markeredgecolor='none', alpha=.75)

                used_colors[color_index] = Line2D([-1], [-1], color=color, lw=1, label=metric)
                used_shapes[shape_index] = Line2D([-1], [-1], marker=shape, color='black', linestyle='None', label=self.color_legend[shape_index])
    
        # Draw both legends separately
        key = []
        for k,value in used_colors.items():
            key.append(value)
        for k,value in used_shapes.items():
            key.append(value)
        plt.legend(handles=key, loc='lower right')

        if large_range:
            plt.ylim(bottom=0)
        else:
            plt.ylim(bottom=min_value, top=1)
        # plt.xticks([])
        # plt.xlim(bottom=0)
        # plt.xticks(df["date"].tolist(), rotation=45)
        plt.xlabel("Days Elapsed Since Training")
        plt.ylabel(f"Metric")
        # plt.title(f"{metric.replace("_"," ")}")
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.close()

    def plot_metrics_over_time(self):
        for key in self.dataframes[0].columns:
            self.plot_metric(key)
        
    def plot_metric(self, metric:str):
        plt.clf() 
        plt.close('all')
        plt.figure(figsize=(5,4,)) 
        save_path = os.path.join(self.save_path, f"{metric}.{self.save_type}")
        used_shapes = {}
        used_colors = {}

        large_range= False

        for index, df in enumerate(self.dataframes):
            color_label = self.color_legend[index]
            shape_label = self.shape_legend[index]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]
            
            df = df.sort_values(by="date")
            try: df[metric]
            except KeyError: continue
            if metric != "date":
                if df[metric].max() >1:
                    large_range=True

            days_since_start = (df["date"]-df["date"].min()).dt.days
            plt.plot(days_since_start, df[metric], marker=shape, linestyle="-", color=color, markersize=8, markerfacecolor=color, markeredgecolor='none', alpha=.75)

            used_colors[color_index] = Line2D([-1], [-1], color=color, lw=1, label=color_label)
            used_shapes[shape_index] = Line2D([-1], [-1], marker=shape, color='black', linestyle='None', label=shape_label)

    
        # Draw both legends separately
        key = []
        for k,value in used_colors.items():
            key.append(value)
        for k,value in used_shapes.items():
            key.append(value)
        plt.legend(handles=key, loc='lower right')

        if large_range:
            plt.ylim(bottom=0)
        else:
            plt.ylim(bottom=0, top=1)
        # plt.xticks([])
        # plt.xlim(bottom=0)
        # plt.xticks(df["date"].tolist(), rotation=45)
        plt.xlabel("Days Elapsed Since Training")
        plt.ylabel(f"{metric.replace("_"," ")}")
        # plt.title(f"{metric.replace("_"," ")}")
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.close()

    def print_metric_avgs(self):
        for index, df in enumerate(self.dataframes):
            print()
            print(f"Model: {self.color_legend[index]}")
            print(f"Dataset: {self.shape_legend[index]}")
            self.print_metric_avg(df)
            
    def print_metric_avg(self, dataframe:pd.DataFrame):
        for key in dataframe.columns:
            if key == "date": 
                continue
            if isinstance(dataframe[key], pd.Series): 
                continue
            bruh = dataframe[key]
            average = dataframe[key].mean()
            std = dataframe[key].std()
            minn = dataframe[key].min()
            maxx = dataframe[key].max()
            print(f"\t{key}")
            print(f"\t\tAVG: {average}")
            print(f"\t\tSTD: {std}")
            print(f"\t\tMIN: {minn}")
            print(f"\t\tMAX: {maxx}")

    def plot_combined_per_attribute_PR(self, attribute:str, n_bins=20, curve:str="precision", log_x:bool=False):
        fig, ax1 = plt.subplots(figsize=(8, 6))

        for j,df in enumerate(self.studies):
            color_label = self.color_legend[j]
            shape_label = self.shape_legend[j]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]

            ax1.set_xlabel("SNR")
            attrs = np.array(df.cumulative_metrics[attribute])
            if log_x:
                ax1.set_xlabel("Log SNR")
                attrs = np.array(df.cumulative_metrics[attribute])
                attrs[attrs < 0] = 0 
                attrs = np.log1p(attrs)
            tp_list = np.array(df.cumulative_metrics["True_Positives"])
            fp_list = np.array(df.cumulative_metrics["False_Positives"])
            fn_list = np.array(df.cumulative_metrics["False_Negatives"])


            # Bin edges and bin indices
            bin_edges = np.linspace(attrs.min(), attrs.max(), n_bins + 1)
            bin_indices = np.digitize(attrs, bins=bin_edges, right=False) - 1

            # Store metrics per bin
            precision_bins = []
            recall_bins = []
            f1_bins = []
            bin_centers = []

            for i in range(n_bins):
                bin_mask = bin_indices == i
                tp = tp_list[bin_mask].sum()
                fp = fp_list[bin_mask].sum()
                fn = fn_list[bin_mask].sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                precision_bins.append(precision)
                recall_bins.append(recall)
                f1_bins.append(f1)
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            counts, _, _ = ax1.hist(attrs, bins=bin_edges, alpha=0.4, color='gray')
            ax1.set_ylabel("Count", color='gray')
            ax1.tick_params(axis='y', labelcolor='gray')
            # Metrics on second axis
            ax2 = ax1.twinx()
            if "precision" in curve.lower():
                plt.title("Precision")
                ax2.plot(bin_centers, precision_bins, label=f'{color_label}', color=color, marker=shape)
            if "recall" in curve.lower():
                plt.title("Recall")
                ax2.plot(bin_centers, recall_bins, label=f'{color_label}', color=color, marker=shape)
            if "f1" in curve.lower():
                plt.title("F1 Score vs SNR")
                ax2.plot(bin_centers, f1_bins, label=f'{color_label}', color=color, marker=shape)
            ax2.set_ylabel("Score")
            ax2.set_ylim(0, 1.05)

        # Legends and layout
        fig.legend(loc='lower right', bbox_to_anchor=(1, 0), bbox_transform=ax1.transAxes)
        # fig.legend(loc="lower right", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        save_path = os.path.join(self.save_path, f"{curve}_vs_{attribute}-{self.basename}.png")
        plt.savefig(save_path)

    def plot_combined_PR_Curves(self, threshold_fit = 1, threshold_confidence = 0.5):
        fig, ax1 = plt.subplots(figsize=(8, 6))

        for j,df in enumerate(self.studies):
            color_label = self.color_legend[j]
            shape_label = self.shape_legend[j]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]

            confidence_index = df.cumulative_metrics["PR_Curve_Confidence"].index(threshold_confidence)

            recall = df.cumulative_metrics["PR_Curve_Recall"][confidence_index,:]
            precision = df.cumulative_metrics["PR_Curve_Precision"][confidence_index,:]
                
            plt.plot(recall, precision, 'o-', label=f'{color_label}', color=color)

        # Labels and title
        save_path = os.path.join(self.save_path, f"PR_Curve_Confidence-{self.basename}.png")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.legend(loc='lower left')
        plt.savefig(save_path)
        plt.close()

        plt.figure(figsize=(8,6)) 

        for j,df in enumerate(self.studies):
            color_label = self.color_legend[j]
            shape_label = self.shape_legend[j]
            color_index = self.color_list.index(color_label)
            shape_index = self.shape_list.index(shape_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]

            fit_index = df.cumulative_metrics["PR_Curve_Fit"].index(threshold_fit)

            recall = df.cumulative_metrics["PR_Curve_Recall"][:,fit_index]
            precision = df.cumulative_metrics["PR_Curve_Precision"][:,fit_index]
                
            plt.plot(recall, precision, 'o-', label=f'{color_label}') 

        # Labels and title
        save_path = os.path.join(self.save_path, f"PR_Curve_Fit-{self.basename}.png")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve')
        plt.grid(True)

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.legend(loc='lower left')
        plt.savefig(save_path)
        plt.close()