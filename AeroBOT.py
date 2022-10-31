import os

# Manage the arguments passed in the cmd line
import argparse
parser = argparse.ArgumentParser(description="HELP for AeroBOT.py:")
parser.add_argument('limit', 
                    type=int, 
                    default=10, 
                    help='(int) How many entries of the DataFrame to process. \
                    The higher this argument, the longer the inference will last. \
                    Default: 10')
args = parser.parse_args()

from aerobotpackages import train_load_transformer_model
from aerobotpackages import  y_prob_to_y_pred, y_multilabel_to_binary, convert_clf_rep_to_df_multilabel_BERT_kw_args

# Load the FINAL TEST data (10805 entries)
import pickle as pkl
os.chdir("./data/transformed/") 
with open("df_test_for_Anomaly_prediction.pkl", "rb") as f:
    loaded_data = pkl.load(f)

df = loaded_data
os.chdir("..")
print("\n A Dataframe with", len(df), "entries has been loaded.")
print(f'Only the {args.limit} first entries will be used.')

# Retrieve the list of Anomaly label columns
Anomaly_RootLabels_columns = []
for col in df.columns:
  if 'Anomaly_' in str(col):
      Anomaly_RootLabels_columns.append(col)

# Define variables
PWD = os.getcwd()
dir_name = PWD + '/models_and_outputs/2022_10_23_11_3_3_BERT_classes_reproduce_7_3_9_3/'
experiment_name = '11_3_3'

# Call the function
train_load_transformer_model(dir_name = dir_name,
          experiment_name = experiment_name,
          df = df[:args.limit],
          anomalies = Anomaly_RootLabels_columns, 
          train_mode = False, 
          num_epochs = 20,
          load_model = True, 
          save_and_overwrite_model = False)