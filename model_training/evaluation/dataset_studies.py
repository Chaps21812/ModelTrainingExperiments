import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import os
import mlflow
import numpy as np
from matplotlib.lines import Line2D
from datetime import datetime


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