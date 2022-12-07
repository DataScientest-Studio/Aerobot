##############################################
# IMPORT PACKAGES
##############################################
print("Importing packages...")
# Generic
import os
import gdown
import zipfile
import argparse
import pickle as pkl
import pathlib
from pathlib import Path  

# Custom-made, project-specific packages
from aerobotpackages import train_load_transformer_model

##############################################
# DOWNLOAD FILES
##############################################
print("Checking the presence of necessary data for inference, download if necessary.")
AeroBOT_root_dir = Path.cwd()
trans_data_path = AeroBOT_root_dir.joinpath('data', 'transformed')
model_data_path = AeroBOT_root_dir.joinpath('data', 'models')

def download_from_GDrive(path, filename, url, zipped=False):
    """
    Inputs
    -------
    - path: path where to look for the file and if not present, download. Use OS-independent paths with the help of the pathlib library
    - file_to_download (str): filename of the file to download
    - url (str): Google Drive URL of the file to download. Example: 'https://drive.google.com/uc?id=1HZSxIfwGqg38yByiXxS4EjyLgeHI6yOv'
    
    If filename is .zip, will unzip it into a folder named [filename].

    Return
    -------
    None
    """
    if path.joinpath('11_3_3').exists(): # the model has been already unzipped
        print('check this:', path.joinpath('11_3_3').exists())
        return None 

    if not path.joinpath(filename).exists(): # no files are present
        os.chdir(path) 
        print("Downloading file '{}'...".format(filename))
        gdown.download(url, filename, quiet=False)

    else:
        print("File '{}' is already present.".format(filename))
        if filename[-4:] == '.zip':
            os.chdir(model_data_path) 
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                print('Unzipping the file...')
                zip_ref.extractall('') # if '', extracts into pwd     
            print("Deleting zip file...")
            os.remove(filename)
            print("Zip file was deleted successfully.")
    os.chdir(AeroBOT_root_dir)
    
    return None

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
