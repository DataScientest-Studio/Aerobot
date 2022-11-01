##############################################
# IMPORT PACKAGES
##############################################
# Generic
import os
import gdown
import zipfile
import argparse
import pickle as pkl

# Custom-made, project-specific packages
from aerobotpackages import train_load_transformer_model
#from aerobotpackages import  y_prob_to_y_pred, y_multilabel_to_binary, convert_clf_rep_to_df_multilabel_BERT_kw_args

##############################################
# DOWNLOAD FILES
##############################################
print("Checking the presence of necessary data for inference, download if necessary.")
if not os.path.exists('./data/transformed/df_test_for_Anomaly_prediction.pkl'):
    # Download final test set data
    os.chdir("./data/transformed/") 
    url = 'https://drive.google.com/uc?id=1HZSxIfwGqg38yByiXxS4EjyLgeHI6yOv'
    output = 'df_test_for_Anomaly_prediction.pkl'
    print('Downloading final test set data...')
    gdown.download(url, output, quiet=False)
    # Move up 2 directories, to the location of the present script
    os.chdir('..')
    os.chdir('..')

else: 
    print('df_test_for_Anomaly_prediction.pkl is already present.')

if not os.path.exists('./data/models/11_3_3/'):
    # Download transformer model
    os.chdir("./data/models/") 
    url = 'https://drive.google.com/uc?id=1qUGjWwJ9vLutgw_P-1Ax2Hl3DSm-7XGX'
    output = 'model_11_3_3.zip'
    print('Downloading transformer model zip file...')
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile('model_11_3_3.zip', 'r') as zip_ref:
        print('Unzipping the model file...')
        zip_ref.extractall('') # if '', extracts into pwd

    print("Deleting zip file...")
    os.remove('model_11_3_3.zip')
    print("Zip file successfully deleted.")

    # Move up 2 directories, to the location of the present script
    os.chdir('..')
    os.chdir('..')
else: 
    print('Transformer model 11_3_3 is already present.')

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
os.chdir("./data/transformed/") 
print("Loading pd.DataFrame...")
with open("df_test_for_Anomaly_prediction.pkl", "rb") as f:
    loaded_data = pkl.load(f)

df = loaded_data
os.chdir("..")
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
# Define variables
PWD = os.getcwd()
dir_name = PWD + '/models/'
experiment_name = '11_3_3'

# Call the function
train_load_transformer_model(dir_name = dir_name,
          experiment_name = experiment_name,
          df = df[:args.limit],
          anomalies = Anomaly_RootLabels_columns, 
          train_mode = False, 
          num_epochs = 20,
          load_model = True, 
          save_and_overwrite_model = True) # in inference mode, this only exports .pkl files
          
###### THE END; THANKS FOR WATCHING! #######