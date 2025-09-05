from evaluation.dataset_studies import Single_Dataset_Study, Multi_Dataset_Study
S1 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments/Evaluation_Sim1Real_2025-08-25_10:58"
S2 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments/Evaluation_Sim2Real_2025-08-25_10:58"
R1 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments/Evaluation_CL1Real_2025-08-25_10:57"
R2 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments/Evaluation_CL2Real_2025-08-25_10:57"
R5 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments/Evaluation_CL5Real_2025-08-25_10:58"

color_legend=["1 Step", "2 Step", "1 Step", "2 Step", "5 Step"]
shape_legend=["Satsim", "Satsim", "Real", "Real", "Real"]
experiments = [S1, S2, R1, R2, R5]

color_legend=["1 Step", "2 Step", "5 Step"]
shape_legend=[ "Real", "Real", "Real"]
experiments = [ R1, R2, R5]

save_path = '/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Plots/CL_Plots_WACV'
multi_study = Multi_Dataset_Study(experiments, color_legend, shape_legend=shape_legend,save_path=save_path )
# multi_study.plot_combined_PR_Curves(threshold_fit=2)
multi_study.plot_combined_per_attribute_PR("local_snr", curve="precision")
multi_study.plot_combined_per_attribute_PR("local_snr", curve="recall")
multi_study.plot_combined_per_attribute_PR("local_snr", curve="f1")