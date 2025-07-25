import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import os
import mlflow
import numpy as np
from matplotlib.lines import Line2D


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
            self.plot_metric(key)

    def save_cumulative_metrics(self):
        with open(self.overal__metrics_path, 'w') as f:
            for key, value in self.cumulative_metrics.items():
                f.write(f"{key}: {value}\n")

class CompiledExperiments():
    def __init__(self, paths:list, model_names:list, datasets:list, save_path:str):
        self.paths = paths
        self.model_names = model_names
        self.datasets = datasets
        self.dataframes = []
        self.save_path = save_path
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
            (0.0902, 0.7451, 0.8118)
        ]
 

        # self.matplotlib_colors = ['red','blue','green','cyan','magenta','yellow','black','orange','purple','darkgreen']
        self.matplotlib_markers = ['o',  's',  '^',  '<',  '>',  'D',  '*',  'x', '+']

        self.model_set = list(set(self.model_names))
        self.dataset_set = list(set(self.datasets))


        for file_path in paths:
            files = os.listdir(file_path)
            filename = [file for file in files if file.endswith('.pkl')]
            if len(filename) > 0:
                filename = filename[0]
            else: continue
            study = Dataset_study(save_path,[""],"")
            study = study.load(os.path.join(file_path, filename))
            self.dataframes.append(study.dataframe)

    def plot_metrics_over_time(self):
        for key in self.dataframes[0].columns:
            self.plot_metric(key)
        
    def plot_metric(self, metric:str):
        plt.clf() 
        plt.close('all')
        plt.figure(figsize=(7,6)) 
        save_path = os.path.join(self.save_path, f"{metric}.pdf")
        used_shapes = {}
        used_colors = {}

        large_range= False

        for index, df in enumerate(self.dataframes):
            color_label = self.model_names[index]
            shape_label = self.datasets[index]
            color_index = self.model_set.index(color_label)
            shape_index = self.dataset_set.index(shape_label)
            color = self.matplotlib_colors[color_index]
            shape = self.matplotlib_markers[shape_index]
            
            df = df.sort_values(by="date")
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
        plt.legend(handles=key, loc='upper right')

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
            print(f"Model: {self.model_names[index]}")
            print(f"Dataset: {self.datasets[index]}")
            self.print_metric_avg(df)
            
    def print_metric_avg(self, dataframe:pd.DataFrame):
        for key in dataframe.columns:
            if key == "date": continue
            average = dataframe[key].mean()
            std = dataframe[key].std()
            minn = dataframe[key].min()
            maxx = dataframe[key].max()
            print(f"\t{key}")
            print(f"\t\tAVG: {average}")
            print(f"\t\tSTD: {std}")
            print(f"\t\tMIN: {minn}")
            print(f"\t\tMAX: {maxx}")

