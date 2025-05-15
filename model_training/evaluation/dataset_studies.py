import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import os
import mlflow


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
        self.figure_folder = os.path.join(self.output_directory, "figures")
        self.pickle_path = os.path.join(self.output_directory, f"{self.basename}_{title}.pkl")
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.figure_folder, exist_ok=True)


    def save(self):
        """Save the object as a pickle file."""
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename:str):
        """Load an object from a pickle file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
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
        plt.clf() 
        plt.close('all')
        plt.figure(figsize=(10, 12)) 
        plt.plot(self.dataframe["date"], self.dataframe[metric], color="teal")
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


    