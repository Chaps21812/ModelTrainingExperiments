from astropy.io import fits
import json
import os
from evaluation.dataset_studies import Dataset_study, CompiledExperiments, Single_Dataset_Study

dataset_study= "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments/Evaluation_ALLData_E_2025-11-17_05:33"
experiments = Single_Dataset_Study.load(dataset_study)
experiments.plot_PR_Curves()
experiments.plot_per_attribute_PR("local_snr", 20, log_y=True)