##############################################
# IMPORT PACKAGES
##############################################
print("Importing packages...")
# Generic
import os
import argparse
import pickle as pkl
import pathlib
from pathlib import Path  

# Custom-made, project-specific packages
from aerobotpackages import train_load_transformer_model, download_from_GDrive

##############################################
# DOWNLOAD FILES
##############################################
print("Checking the presence of necessary data for inference, download if necessary.")
AeroBOT_root_dir = Path.cwd()
trans_data_path = AeroBOT_root_dir.joinpath('data', 'transformed')
model_data_path = AeroBOT_root_dir.joinpath('data', 'models')

download_from_GDrive(trans_data_path, 
                    filename = 'df_test_for_Anomaly_prediction.pkl', 
                    url = 'https://drive.google.com/uc?id=1HZSxIfwGqg38yByiXxS4EjyLgeHI6yOv')
download_from_GDrive(model_data_path, 
                    filename = 'model_11_3_3.zip', 
                    url = 'https://drive.google.com/uc?id=1qUGjWwJ9vLutgw_P-1Ax2Hl3DSm-7XGX')
# Files for streamlit demo
download_from_GDrive(trans_data_path, 
                    filename = 'model_results_diffBLM_bestmodel_20221207.csv', 
                    url = 'https://drive.google.com/uc?id=1Fv__EH0Z6gUSirpKzEmrDUsKkbBtxMEb')    
download_from_GDrive(trans_data_path, 
                    filename = 'baseline_vs_best_BERT_20221207.csv', 
                    url = 'https://drive.google.com/uc?id=1-5faJU71SZ_LbehLMwKddO_aeWk-e-rk')  
os.chdir(AeroBOT_root_dir) # make sure we return to the root directory of the repo

##############################################
# Manage the arguments passed in the cmd line
##############################################
parser = argparse.ArgumentParser(description="HELP for AeroBOT.py:")
parser.add_argument('limit', 
                    type=int, 
                    default=10, 
                    help='(int) How many entries of the DataFrame to process. \
                    The higher this argument, the longer the inference will last. \
                    Default: 10')
args = parser.parse_args()

##############################################
# Load the FINAL TEST data (10805 entries)
##############################################
os.chdir(trans_data_path)  
print("Loading pd.DataFrame...")
with open("df_test_for_Anomaly_prediction.pkl", "rb") as f:
    loaded_data = pkl.load(f)
df = loaded_data
os.chdir(AeroBOT_root_dir)
print("A Dataframe with", len(df), "entries has been loaded.")
print(f'Only the {args.limit} first entries will be used.')

##############################################
# Retrieve the labels of the target variable
##############################################
Anomaly_RootLabels_columns = []
for col in df.columns:
  if 'Anomaly_' in str(col):
      Anomaly_RootLabels_columns.append(col)

##############################################
# Infer on final test set
##############################################
experiment_name = '11_3_3'
train_load_transformer_model(
          dir_name = str(model_data_path) + pathlib.os.sep, # on MAC-OS: pathlib.os.sep = '/'
          experiment_name = '11_3_3',
          df = df[:args.limit],
          anomalies = Anomaly_RootLabels_columns, 
          train_mode = False, 
          num_epochs = 20,
          load_model = True, 
          save_and_overwrite_model = True) # in inference mode, this only exports .pkl files
          
###### THE END; THANKS FOR WATCHING! #######