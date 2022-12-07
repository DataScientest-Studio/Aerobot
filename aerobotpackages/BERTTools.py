# project-specific functions and classes created in the 'Aerobot' project by Ioannis STASINOPOULOS
# This file contains functions and classes for the BERT implementation

#####################################################################################################
# Import generic packages
#####################################################################################################
import numpy as np
import seaborn as sns
import math # for math.pi etc.
import time # time code execution
from pathlib import Path  

#######################
# Pandas
#######################
import pandas as pd
# Set pandas settings to show all data when using .head(), .columns etc.
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.colheader_justify","left") # left-justify the print output of pandas

### Display full columnwidth
# Set pandas settings to display full text columns
#pd.options.display.max_colwidth = None
# Restore pandas settings to display standard colwidth
pd.reset_option('display.max_colwidth')

import itertools # Pour créer des iterateurs

# Package to show the progression of pandas operations
from tqdm import tqdm
# from tqdm.auto import tqdm  # for notebooks

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()
# simply use .progress_apply() instead of .apply() on your pd.DataFram

######################
# PLOTTING
######################
import matplotlib.pyplot as plt
# Define global plot parameters for better readability and consistency among plots
# A complete list of the rcParams keys can be retrieved via plt.rcParams.keys() function
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 23
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 23
plt.rc('legend', fontsize=23)    # legend fontsize

###############################
# ML preprocessing and models
###############################
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble # random forest
from sklearn.svm import SVC

# EVALUATION tools from sklearn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, average_precision_score, precision_recall_curve, PrecisionRecallDisplay

###############################
# Deep Learning
###############################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, Flatten, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras import callbacks

###############################
# Other
###############################
import pickle as pkl # Saving data externally

  
#####################################################################################################
# MAIN CLASSES AND FUNCTIONS FOR BERT IMPLEMENTATION
#####################################################################################################
  
class DataPrepMultilabelBERT():
    '''
    Prepare data for multilabel classification using BERT.
    An object of this class will return 3 'tf.data.Dataset' datasets, when called:
    tf_train_dataset, tf_validation_dataset, tf_test_dataset.

    Inference only
    ---------------
    If the 'train_mode = True' option is passed when calling an object of this
    class, *dummy* train and validation datasets are built, containing only the
    first two entries of df (see attributes at instantiation). In this case,
    df should equal the test set, on which inference should be performed.
    The test set returned contains the full input data 'df'.
    For more details, read the DocString of the build_dummy_datasets() method.

    Methods
    ----------
    - __init__(), like every class
    - get_text_and_labels()
    - train_validation_test_split()
    - build_dummy_datasets()
    - convert_to_HF_dataset()
    - tokenize_the_BERT_way()
    - convert_to_TensorFlow_dataset()
    - __call__()
      Reminder: The __call__() method is called by executing 'object()',
      where 'object' is an instance of the class.

    Attributes at class object instantiation
    -----------------------------------------
    Attributes that are created within __call__(). They are the outputs of the
    class's own methods.
    - BERT_model_name : string, "google/bert_uncased_L-12_H-768_A-12" for BERT BASE,
      or one of the Small BERTs: "google/bert_uncased_L-12_H-128_A-2"
      or #google/bert_uncased_L-2_H-128_A-2" (BERT tiny)
      https://huggingface.co/google/bert_uncased_L-12_H-768_A-12
    - df: pd.DataFrame, containing
        - ACN number (unique code for each ASRS database entry)
        - narratives
        - one-hot encoded labels for the narratives, e.g. Anomaly
    - anomalies: list of anomaly labels, e.g. as obtained from df.columns
    - text_input: string, which column to use as the text,
      e.g. 'Narrative', 'Narrative_PP_stemmed_str', ...
    - max_length: int, default = 200, length of tokenized text
    - batch_size: int, default = 32, batch size of the tf.data.Dataset created
    - random_state : for the train_validation_test_split() method of the class,
      see corresponding Docstring
    - first_split_prop : for the train_validation_test_split() method of the class,
      see corresponding Docstring
    - second_split_prop : for the train_validation_test_split() method of the class,
      see corresponding Docstring

    Attributes created by the class's __call__() method
    ------------------------------------------------
    - text_and_labels: output of get_text_and_labels() method
    - df_train, df_validation, df_test: outputs of either
      - train_validation_test_split() method, or
      - build_dummy_datasets() method (when in inference mode)
    - HF_dataset, train_dataset_HF, validation_dataset_HF, test_dataset_HF
      outputs of convert_to_HF_dataset() method
    - tokenized_dataset, train_set_len: outputs of tokenize_the_BERT_way() method
    - tf_train_dataset, tf_validation_dataset, tf_test_dataset: outputs of
      convert_to_TensorFlow_datasets() method
    '''

    def __init__(self,
                 BERT_model_name,
                 df,
                 anomalies,
                 text_input='Narrative',
                 max_length=200,
                 batch_size=32,
                 random_state=12,
                 first_split_prop=0.2,
                 second_split_prop=0.2):

        self.BERT_model_name = BERT_model_name
        self.df = df
        self.anomalies = anomalies
        self.text_input = text_input
        self.max_length = max_length
        self.batch_size = batch_size
        self.random_state = random_state
        self.first_split_prop = first_split_prop
        self.second_split_prop = second_split_prop

        # Instantiate a data_collator used by the method 'convert_to_TensorFlow_datasets()'
        from transformers import DefaultDataCollator
        self.data_collator = DefaultDataCollator(return_tensors="tf")

    def get_text_and_labels(self, df, anomalies, text_input):
        '''
        Return a DataFrame containing a list of binary multilabels and a text per row.
        Inputs
        ------
        - df: pd.DataFrame
        - anomalies: list of Anomaly label names
        - text_input: name of the column of df that contains the texts, e.g. 'Narrative'
        '''
        print("Creating multilabels...")
        text_and_labels = df[anomalies]
        text_and_labels['labels'] = text_and_labels.apply(lambda r: tuple(r), axis=1).apply(np.array)
        # /!\ has to be 'labels' and not 'label' or other. HuggingFace expects this.
        # With 'label', I got AssertError in the conversion to tf.dataset
        text_and_labels = text_and_labels.drop(columns=self.anomalies)
        text_and_labels['text'] = df[self.text_input]
        text_and_labels = text_and_labels.reset_index().drop(columns=['ACN'])
        print("Example of text and corresponding multilabel:\n")
        print(text_and_labels.iloc[0])
        print("\n")
        print("get_text_and_labels() done")
        print(30 * "*", "\n")
        return text_and_labels

    def train_validation_test_split(self,
                                    text_and_labels,
                                    first_split_prop,
                                    second_split_prop,
                                    random_state):
        '''
        Split arrays or matrices into random train, validation and test subsets.
        First separate test set from df, then separate validation set.

        Inputs
        ----------
        - text_and_labels : pd.DataFrame containing only the columns 'text' and 'label'
        - first_split_prop : float, default = 0.2
          Proportion to use in the first split.
        - second_split_prop : float, default = 0.2
          Proportion to use in the second split.
        - random_state : default = 12

        Return
        ----------
        pd.DataFrames
        - df_train
        - df_validation
        - df_test
        '''
        print("Splitting dataset...")
        # Train-test split
        df_train_plus_val, df_test, = train_test_split(text_and_labels,
                                                       test_size=first_split_prop,
                                                       random_state=random_state)
        # Train-validation split
        df_train, df_validation, = train_test_split(df_train_plus_val,
                                                    test_size=second_split_prop,
                                                    random_state=random_state)

        print("train set length:", len(df_train))
        print("validation set length:", len(df_validation))
        print("test set length:", len(df_test))

        print("\n")
        print("train_validation_test_split() done")
        print(30 * "*", "\n")
        return df_train, df_validation, df_test

    def build_dummy_datasets(self,
                             text_and_labels):
        '''
        Build dummy train and validation datasets, containing only the first two
        entries of text_and_labels, which should correspond to the test set.

        The test set returned contains the full input data 'text_and_labels'.

        Why create dummy datasets? In order to modify the data preparation pipeline
        as little as possible, because it builds complex data structures
        (i.e. HuggingFace dictionary, tf.Data.dataset).

        Inputs
        ----------
        - text_and_labels : pd.DataFrame containing only the columns 'text' and 'label'

        Return
        ----------
        pd.DataFrames
        - df_train_dummy
        - df_validation_dummy
        - df_test
        '''
        print("Building dummy train and validation datasets: they contain only the first two entries of the test set.")
        df_train_dummy = pd.DataFrame(text_and_labels.iloc[:2])
        df_validation_dummy = pd.DataFrame(text_and_labels.iloc[:2])
        df_test = text_and_labels

        print("dummy train set length:", len(df_train_dummy))
        print("dummy validation set length:", len(df_validation_dummy))
        print("test set length:", len(df_test))

        print("\n")
        print("build_dummy_datasets() done")
        print(30 * "*", "\n")
        return df_train_dummy, df_validation_dummy, df_test

    def convert_to_HF_dataset(self, df_train, df_validation, df_test):
        '''
        Convert pd.DataFrames to 🤗 datasets and combining them into a 🤗 DatasetDict.

        Input
        -------
        pd.DataFrames
        - df_train
        - df_validation
        - df_test

        Return
        -------
        - HF_dataset: 🤗 DatasetDict
        - Its three constituent 🤗 datasets:
          - train_dataset_HF
          - validation_dataset_HF
          - test_dataset_HF
        '''
        print("Combining pd.DataFrames into a HuggingFace dataset...")
        # IMPORT 🤗 PACKAGES
        # when putting outside this function, it did not recognize 'Dataset'
        from datasets import Dataset, DatasetDict

        # Convert into 🤗 datasets
        train_dataset_HF = Dataset.from_dict(df_train)
        validation_dataset_HF = Dataset.from_dict(df_validation)
        test_dataset_HF = Dataset.from_dict(df_test)
        # use from_dict() and not from_pandas()
        # otherwise you get an extra key, smth litke '__index col__'
        print("\n Structure of Hugging Face dataset (train):", train_dataset_HF)

        # 'merge' the three 🤗 datasets into a single 🤗 DatasetDict
        HF_dataset = DatasetDict({"train": train_dataset_HF,
                                  "validation": validation_dataset_HF,
                                  "test": test_dataset_HF})
        print("\n Structure of the complete Hugging Face dataset:\n", HF_dataset)
        print("\n First entry of the train dataset:\n", HF_dataset["train"][0])
        print("\n")
        print("convert_to_HF_dataset() done")
        print(30 * "*", "\n")
        return HF_dataset, train_dataset_HF, validation_dataset_HF, test_dataset_HF

    def tokenize_the_BERT_way(self, HF_dataset, BERT_model_name, max_length):
        '''
        Tokenize a 🤗 dataset with the appropriate BERT tokenizer, downloaded from 🤗

        Inputs
        ------
        - HF_dataset: 🤗 DatasetDict
        - BERT_model_name (string), as defined on the 🤗 website
        - max_length (int), desired length of sequence

        Return
        ------
        - tokenized_dataset, a 🤗 DatasetDict
        - train_set_len (int): length of the train set
        '''
        # instantiate tokenizer
        # IMPORT 🤗 PACKAGES
        from transformers import AutoTokenizer
        # TOKENIZE using a pre-trained tokenizer from HuggingFace

        tokenizer = AutoTokenizer.from_pretrained(BERT_model_name)
        max_length = max_length

        # Define a function in order to use .map() below
        def tokenize_function(examples):
            return tokenizer(examples["text"],
                             padding='max_length', max_length=max_length,  # same value as we used for WordEmbedding
                             truncation=True
                             # return_tensors="tf"
                             )

        # Map the tokenization function onto our dataset
        # it is ok to apply it also to the test set, since it is a *pretrained* tokenizer,
        # i.e. it will not train on our data
        print("Performing tokenization in batches on: train, validation, test sets...")

        # Auxialiary variable, see below
        pre_tokenizer_columns = set(HF_dataset["train"].features)

        tokenized_dataset = HF_dataset.map(tokenize_function, batched=True)
        # The tokenizer uses its own batch_size, not the class attribute

        # Display the additional columns created by the tokenizer
        # They are necessary inputs for BERT
        tokenizer_columns = list(set(tokenized_dataset["train"].features) - pre_tokenizer_columns)
        print("Columns added by tokenizer:", tokenizer_columns)
        print("\n Structure of the complete tokenized Hugging Face dataset:\n", tokenized_dataset)
        print("\n")
        print("tokenize_the_BERT_way() done")
        print(30 * "*", "\n")
        # return length of tokenized_dataset["train"] to feed it to AdamW optimizer
        train_set_len = len(tokenized_dataset["train"])

        return tokenized_dataset, train_set_len

    def convert_to_TensorFlow_datasets(self, tokenized_dataset_to_convert, key="test"):
        """
        Convert our datasets to tf.data.Dataset, which Keras understands natively.

        Inputs
        -------
        - tokenized_dataset_to_convert: a 🤗 DatasetDict
        - key (str): key of the 🤗 DatasetDict "train", "validation", "test"

        Return
        -------
        - tf_dataset: tf.data.Dataset
        """
        # Two ways to do the conversion:
        # (i) use the slightly more low-level Dataset.to_tf_dataset() method
        # (ii) use Model.prepare_tf_dataset().
        # The Model method can inspect the model to determine which column names it
        # can use as input, which means you don't need to specify them yourself.
        # Unless our samples are all the same length, we will also need to pass
        # a tokenizer or collate_fn so that the tf.data.Dataset knows
        # how to pad and combine samples into a batch.
        tf_dataset = tokenized_dataset_to_convert[key].to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            label_cols=["labels"],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.batch_size)

        return tf_dataset

    def __call__(self, train_mode=True):
        """
        Function that makes class instances being callable.
        This function calls all methods of the class and
        (i)  passes the output of one function as input of the next one
        (ii) writes the method outputs as additional class attributes. This
        is useful for fetching some of these outputs later.

        Inputs
        ------
        - train_mode (bool): Whether to prepare the data for training or for
          inference only, e.g. when you have only a test set to pass.
          Default = True

        Return
        ------
        - Outputs and in the same time, attributes of the class
          - self.tf_train_dataset
          - self.tf_validation_dataset
          - self.tf_test_dataset

        Note: If train_mode = False, 'tf_train_dataset' and
        'tf_validation_dataset' are dummy datasets, containing only the first and
        second entry of the test set, respectively.
        The test set contains the full input data. This is done, in order to
        change the conversion pipeline as little as possible, as it builds complex
        data structures i.e. HuggingFace dictionary, tf.Data.dataset.
        """
        self.text_and_labels = self.get_text_and_labels(self.df,
                                                        self.anomalies,
                                                        self.text_input)
        if train_mode == True:
            self.df_train, self.df_validation, self.df_test = self.train_validation_test_split(self.text_and_labels,
                                                                                               self.first_split_prop,
                                                                                               self.second_split_prop,
                                                                                               self.random_state)
        elif train_mode == False:  # inference only
            self.df_train, self.df_validation, self.df_test = self.build_dummy_datasets(self.text_and_labels)

        self.HF_dataset, self.train_dataset_HF, self.validation_dataset_HF, self.test_dataset_HF = self.convert_to_HF_dataset(
            self.df_train,
            self.df_validation,
            self.df_test)
        self.tokenized_dataset, self.train_set_len = self.tokenize_the_BERT_way(self.HF_dataset,
                                                                                self.BERT_model_name,
                                                                                self.max_length)

        print("Converting tokenized_dataset into tf.data.Dataset datasets...")
        self.tf_train_dataset = self.convert_to_TensorFlow_datasets(tokenized_dataset_to_convert=self.tokenized_dataset,
                                                                    key="train")
        self.tf_validation_dataset = self.convert_to_TensorFlow_datasets(
            tokenized_dataset_to_convert=self.tokenized_dataset, key="validation")
        self.tf_test_dataset = self.convert_to_TensorFlow_datasets(tokenized_dataset_to_convert=self.tokenized_dataset,
                                                                   key="test")

        print("\n Structure of the train tf.Data.dataset:\n", self.tf_train_dataset)
        print("\n")
        print(
            "Content of one batch: the narratives have been converted into 'input_ids', 'token_type_ids', 'attention_mask' (see below)")
        print("The multilabels are shown last ('array'). \n")
        print(next(self.tf_train_dataset.as_numpy_iterator()))
        print("\n")
        print("convert_to_TensorFlow_datasets() done")
        print(30 * "*", "\n")

        return self.tf_train_dataset, self.tf_validation_dataset, self.tf_test_dataset

class GetBERTModel():
  '''
  Attributes
  ----------
  BERT_model_name : string, Important: has to be the same model that was passed
  to instantiate a DataPrepMultilabelBERT object!
  i.e. "google/bert_uncased_L-12_H-768_A-12" for BERT BASE, 
  or one of the Small BERTs: "google/bert_uncased_L-12_H-128_A-2" 
  or #google/bert_uncased_L-2_H-128_A-2" (BERT tiny).
  Therefore, it is best to pass the attribute of the 'DataPrepMultilabelBERT' 
  object as input. 
  https://huggingface.co/google/bert_uncased_L-12_H-768_A-12

  num_classes: int, in our multilabel case: num_classes = len(anomalies)

  trainable_layers: list of int, START at ZERO! [0, ..., 11]
    layers of BERT that should be trainable. 
    'trainable_layers = None' if all layers should be frozen.

  Methods
  ----------
  - __init__(), like every class
  - prepare_config()
  - download_weights()
  - make_layers_trainable()
  - __call__() 
    Reminder: The __call__() method is called by executing 'object()', 
    where 'object' is an instance of the class.

  Attributes created by the class's __call__() method
  ------------------------------------------------
  Attributes that are created within __call__(). They are the outputs of the 
  class's own methods.
  - config: output of prepare_config()
  - downloaded_model: output of download_weights() and then of make_layers_trainable()
  '''
  def __init__(self, 
               BERT_model_name, 
               num_classes,
               trainable_layers):

    self.BERT_model_name = BERT_model_name
    self.num_classes = num_classes
    self.trainable_layers = trainable_layers
  

  def prepare_config(self, BERT_model_name, num_classes):
    """
    Get the BertConfig of the model from Hugging Face and set customize it
    if necessary, e.g. in order to get the hidden_states outputs.
    
    BertConfig is the configuration class to store the configuration 
    of a BertModel (Pytorch) or a TFBertModel (TensorFlow). 
    
    It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. 

    Instantiating a configuration with the defaults will yield a similar 
    configuration to that of the BERT bert-base-uncased architecture.

    Configuration objects inherit from PretrainedConfig and can be used 
    to control the model outputs. 

    Read the documentation from PretrainedConfig for more information.
    https://huggingface.co/docs/transformers/model_doc/bert

    Inputs
    -------
    - BERT_model_name
    - num_classes

    Return
    -------
    - 'config': configuration to pass to the HuggigFace 
    TFAutoModel.from_pretrained() method that gets the model (weights) from the
    HuggingFace library.
    """
    from transformers import AutoConfig
    # AutoXXX functions bring more flexibility regarding model checkpoints
    # https://github.com/huggingface/transformers/issues/5587

    config = AutoConfig.from_pretrained(BERT_model_name)
    # Loading a model from its configuration file does not load the model weights. 
    # It only affects the model’s configuration. 
    # We use from_pretrained() below, to load the model weights.

    config.output_hidden_states = True # we want access to the hidden states' outputs
    config.num_labels = num_classes
    
    print('Configuration: \n\n', config)
    print("prepare_config() done")
    print(30*"*", "\n")
    return config

  
  def download_weights(self, config, BERT_model_name):
    '''
    Download the model (the weights) and pass our configuration,
    as defined in prepare_config().
    Inputs
    -------
    - config: output of the prepare_config() method
    - BERT_model_name

    Return 
    ---------
    - downloaded_model: downloaded model from HuggingFace 🤗 library, 
      configured according to 'config'.
    '''
    from transformers import TFAutoModel
    # Download the Transformers BERT model
    print("Downloading BERT model from Hugging Face...")
    downloaded_model = TFAutoModel.from_pretrained(BERT_model_name, 
                                                    config = config, 
                                                    from_pt = True) # From PyTorch
    return downloaded_model


  def make_layers_trainable(self, downloaded_model, trainable_layers):
    '''
    Make the desired layers of the model trainable.
    Inputs
    -----
    - downloaded_model: output of download_weights() method
    - trainable_layers: list of int, START at ZERO! [0, ..., 11]
      Layers of BERT that should be trainable. 
      'trainable_layers = None' if all layers should be frozen.

    Return
    -----
    - downloaded_model with desired layer training setting
    '''
    # This gives the whole bert base model and sets all layers' trainable attribute at once
    # See this post: https://discuss.huggingface.co/t/fine-tune-bert-models/1554/6
    for layer in downloaded_model.layers: # this is the main layer
       
        if trainable_layers != None:
          layer.trainable = True # this makes ALL layers trainable

        else:
          layer.trainable = False

    # Individually make the desired layers (un)trainable
    # see https://stackoverflow.com/questions/71336067/how-to-freeze-some-layers-of-bert-in-fine-tuning-in-tf2-keras
    nb_layers = len(downloaded_model.bert.encoder.layer)
    print("\n Number of layers present:", nb_layers, "(12 in the case of BERT BASE).\n")

    if trainable_layers != None:
      all_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
      frozen_layers  = [i for i in all_layers if i not in trainable_layers] 

      for layer_ID in frozen_layers:
        downloaded_model.bert.encoder.layer[layer_ID].trainable = False # freeze layer

    # Show  the status of all layers
    for i in range(nb_layers):
      print(f"Layer {i} trainable:", downloaded_model.bert.encoder.layer[i].trainable)

    print("\n")
    print("make_layers_trainable() done")
    print(30*"*", "\n")
    
    # return the updated model
    return downloaded_model

  def __call__(self):
    self.config = self.prepare_config(self.BERT_model_name, self.num_classes)
    self.downloaded_model = self.download_weights(self.config, self.BERT_model_name)
    self.downloaded_model = self.make_layers_trainable(self.downloaded_model, self.trainable_layers)

    return self.downloaded_model

class ConcatSlice(tf.keras.layers.Layer):
    '''
    Custom, nontrainable model layer, inheriting from tf.keras.Layer. 
    Defines which model outputs to keep, e.g. last_hidden_state, hidden_layer[i] etc.
    and whether to concatenate or flatten them.

    Attributes 
    --------
    - layer_to_get : string 'last_hidden_state' or integer -2, -3, etc. or 'concat'
      Which layers' output(s) to use. 
      '-2' means the second-to-last hidden state, 'concat' will produce a richer 
      embedding for each word in the sequence. Default = 'last_hidden_state'
    
    - emb_to_use = (string) 'CLS' or 'flatten', use the embedding of the CLS token
      (start of the sequence) or flatten the output, default = 'CLS'

    - layers_to_concat : list of int, from 1 to 12. Which of BERT's layers 
      # to use for concatenation. Applies only when layer_to_get = 'concat', 
      default = [9, 10, 11, 12]

    Methods
    --------
    - __init__()
    - get_config()
    - use_CLS_or_flatten()
    - concat_BERT_layers()
    - __call__()

    '''    
    def __init__(self, 
                 layer_to_get = 'last_hidden_state',
                 emb_to_use = 'CLS',
                 layers_to_concat = [9, 10, 11, 12], 
                 **kwargs):
      
      # Initialize the parent class, i.e. tf.keras.layers.Layer
      # OLD CODE
      #super(ConcatSlice, self).__init__()
      super(ConcatSlice, self).__init__(**kwargs)
      
      # Class attributes
      self.layer_to_get = layer_to_get
      self.emb_to_use = emb_to_use
      self.layers_to_concat = layers_to_concat

      # OLD CODE
      #super(ConcatSlice, self).__init__(**kwargs)

      # Print information to the user
      if self.layer_to_get == 'last_hidden_state':
        self.layers_to_concat = None # important for the creation of the classif
        # report in pd.DataFrame format
        print("Will get the output(s) of layer(s):", self.layer_to_get)

      elif type(self.layer_to_get) == int:
        # In this case, self.layer_to_get has to be -2 or -3 etc.
        self.layers_to_concat = None # important for the creation of the classif
        # report in pd.DataFrame format
        print(f"Will get the output(s) of layer(s): [hidden_state][{self.layer_to_get}]")
      
      elif self.layer_to_get == 'concat': 
        print("Will get the output(s) of layer(s):", self.layer_to_get)
        print(f"Will concatenate the outputs of BERT layers", self.layers_to_concat, ".\n")
      
      print("Setting used (CLS or flatten?):", emb_to_use, "\n")

    def get_config(self):
      """
      In order to save/load a model with custom-defined layers, 
      or a subclassed model, you should overwrite the get_config 
      and optionally from_config methods of the parent class.
      Returns the config of the layer.

      A layer config is a Python dictionary (serializable, 
      i.e. can be written into .json or .pkl) containing the configuration 
      of a layer. 
      The same layer can be reinstantiated later (without its trained weights) 
      from this configuration.
      see https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
      """
      config = super(ConcatSlice, self).get_config().copy()
      config.update({"layer_to_get": self.layer_to_get,
                     "emb_to_use": self.emb_to_use,
                     "layers_to_concat": self.layers_to_concat})
      return config
    
    def use_CLS_or_flatten(self, x, setting):
      """
      Slice or Flatten() the input 'x', depending on the chosen 'setting'
      """
      #print("setting used:", setting, "\n")
      if setting == 'CLS':
        # keep the embedding of the CLS token only
        x = x[:, 0, :]

      elif setting == 'flatten':
        # flatten the output using keras layer Flatten()
        x = Flatten()(x)
        # expected dimension: embedding size * max_length,
        # e.g. 768 * 200 = 153,600 if using bert_base

      return x

    def concat_BERT_layers(self, inputs, layers_to_concat):
      """
      Concatenante the 'layers_to_concat' layers of BERT
      """
      list_for_concat = []
      for i in layers_to_concat:
        list_for_concat.append(inputs['hidden_states'][i])
      
      return tf.keras.layers.Concatenate(axis = -1)(list_for_concat)

    def call(self, inputs):
      if self.layer_to_get == 'last_hidden_state':
        x = inputs['last_hidden_state']
        x = self.use_CLS_or_flatten(x, setting = self.emb_to_use)
      
      elif type(self.layer_to_get) == int:
        # In this case, self.layer_to_get has to be -2 or -3 etc.
        x = inputs['hidden_states'][self.layer_to_get]
        x = self.use_CLS_or_flatten(x, setting = self.emb_to_use)

      elif self.layer_to_get == 'concat': 
        x = self.concat_BERT_layers(inputs, self.layers_to_concat)
        # expected dimension: embedding size * layers_to_concat,
        # e.g. 768 * 4 = 3072 if using bert_base and concatenating the last 4 layers
        x = self.use_CLS_or_flatten(x, setting = self.emb_to_use)

      #print("Shape of the input to the first dense layer:\n", x.shape, "\n")

      return x
    
class ClassifTransformerModelML(tf.keras.Model):
    '''
    Multilabel, multiclass transformer(BERT)-based model for text classification

    Attributes 
    --------
    - num_classes : (int) number of classes to classify to.
    - max_length : (int) sequence length, e.g. 200. Better pass the 
      .max_length attribute of the Class 'DataPrepMultilabelBERT' object created previously
    - transformer_model : transformer model returned when calling a GetBERTModel instance
    
    Additional attributes created upon instantiation
    -------------------------------------------------
    - input_ids
    - concat_slice
    - dense1
    - dense2
    - transformer_model._saved_model_inputs_spec

    Methods
    -------
    - __init__()
    - get_config()
    - call()
    - summary()

    Notes
    --------
    We are 'subclassing' the 'tf.keras.Model' class (see documentation of tf.keras.Model), 
    i.e. we define our layers in __init__() and implement the model's forward pass in call().
    '''
    def __init__(self, num_classes, max_length, transformer_model, **kwargs):

        tf.keras.backend.clear_session() # ensure that no model is present in the memory

        # Initialize the parent class, i.e. tf.keras.Model
        # OLD code
        #super(ClassifTransformerModelML, self).__init__()
        super(ClassifTransformerModelML, self).__init__(**kwargs)

        ## Attributes specific to our model
        self.max_length = max_length
        self.input_ids = Input(shape=(self.max_length,), name='input_ids', dtype='int32')
        self.num_classes = num_classes
        # Define model layers
        self.transformer_model = transformer_model
        # Custom layer
        self.concat_slice = ConcatSlice() 
        # () will call the default values, i.e. returns the 'CLS' embedding
        # of 'last_hidden_state'.
        
        # Classification head 
        self.dense1 = Dense(units = 32, activation='relu', name='dense1')
        self.dense2 = Dense(units = self.num_classes, activation='sigmoid', name='dense2')

        #########################
        # RESET THE INPUT SPECS
        #########################
        # Keras saves the input specs on the first call of the model. 
        # When loading a pretrained model with transformers using the 
        # 'from_pretrained' class method of TFPretrainedModel, the networks 
        # is first fed dummy inputs. So the saved models expect their
        # input tensors to be of sequence length 5 (that is the length of the 
        # dummy inputs). 
        # To change that behaviour, reset the input specs before saving to 
        # a saved model like this.
        # Ioannis thinks its crucial to reset these specs before the first call 
        # of the pretrained model, that is why this is done inside __init__().
        # see https://github.com/keras-team/keras/issues/14345#issuecomment-1118569356

        # Create dummy tensors with the shape of our actual features
        dummy_array = list(range(1, self.max_length + 1))
        dummy_tensor = tf.constant([dummy_array], dtype = tf.int64) # it has to be int64
        # Dummy model input
        features = {"input_ids": dummy_tensor, 
                    "attention_mask": dummy_tensor, 
                    "token_type_ids": dummy_tensor}
        # Set the save spec using the dummy input and write them as class attributes           
        self.transformer_model._saved_model_inputs_spec = None
        self.transformer_model._set_save_spec(features)
        #print("saved_model_inputs_spec have been overwritten to:")
        #print(self.transformer_model._saved_model_inputs_spec)
        print("\n")
        ###########################

    def get_config(self):
        """
        In order to save/load a model with custom-defined layers, 
        or a subclassed model, you should overwrite the get_config 
        and optionally from_config methods.
        Returns the config of the layer.

        A layer config is a Python dictionary (serializable, 
        i.e. can be written into .json or .pkl) containing the configuration 
        of a layer. 
        The same layer can be reinstantiated later (without its trained weights) 
        from this configuration.
        see https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        """
        config = super(ClassifTransformerModelML, self).get_config().copy()
        config.update({"num_classes": self.num_classes, 
                        "max_length": self.max_length, 
                        "transformer_model": self.transformer_model#,
                        # "infer": self.infer,
                        # "save_transformer": self.save_transformer
                       })
        # print("CONFIG:")
        # print(config, "\n")
        return config
    
    def call(self, inputs):
        #print("The call() method of a ClassifTransformerModelML class object has been called.\n")
        # Connect the layers the functional way
        x = self.transformer_model(inputs)
        x = self.concat_slice(x) 
        x = self.dense1(x)
        x = self.dense2(x)
        #print("Shape of final dense layer's output:\n", x.shape, "\n")

        return x

    def summary(self, print_fn=None):
        """
        Print the model's summary
        """
        # Instantiate an object of the class tf.keras.Model
        model = Model(inputs=[self.input_ids], 
                      outputs=self.call(self.input_ids),
                      name='Generic model instance')
        return model.summary(print_fn = print_fn)

def compile_transformer(model, 
                        batch_size,
                        train_set_len,
                        num_epochs = 10,
                        metric = 'binary_accuracy',
                        optimizer_type = 'AdamW'):
  """
  Inputs
  -------
  - model: instance of 'ClassifTransformerModelML' class
  - batch_size
  - train_set_len (int): length of the train set; see the outputs of the 
    __call__() method of DataPrepMultilabelBERT
  - num_epochs (int): number of epochs, default = 10
  - metric: default = 'binary_accuracy' (calculates how often predictions 
    match binary labels. This is the default for a multilabel classification)
  - optimizer_type: pass 'AdamW' or 'Adam', default = 'AdamW'

  Return
  -------
  - model: compiled model
  - optimizer
  - num_epochs: for further use by .train()
  - loss
  - metric
  """
  ########################  
  # Create optimizer
  ########################
  if optimizer_type == 'Adam':
    optimizer = keras.optimizers.Adam(model,
                                      learning_rate=5e-05,
                                      epsilon=1e-08,
                                      decay=0.01,
                                      clipnorm=1.0)

  elif optimizer_type == 'AdamW': 
    # The create_optimizer function in the Transformers library creates an AdamW 
    # optimizer with weight and learning rate decay. This performs very well for 
    # training most transformer networks - we recommend using it as your default 
    # unless you have a good reason not to! Note, however, that because it decays 
    # the learning rate over the course of training, it needs to know how 
    # many batches it will see during training.
    # See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.create_optimizer
    from transformers import create_optimizer
    batches_per_epoch = train_set_len // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

    optimizer, schedule = create_optimizer(init_lr = 2e-5,
                                          # The initial learning rate for the schedule after the warmup 
                                          # (so this will be the learning rate at the end of the warmup)
                                          num_warmup_steps=0,
                                          num_train_steps=total_train_steps)

  ########################  
  # Set loss and metrics
  ########################
  loss = tf.keras.losses.BinaryCrossentropy(from_logits = False) 
  # value in [-inf, inf] when from_logits=True or a probability (i.e, value in [0., 1.] when from_logits = False).
  metric = tf.keras.metrics.BinaryAccuracy(metric) 

  ########################  
  # Compile the model
  ########################
  # Defining a variable, e.g. 'compiled_model' and assigning it the output of
  # model.compile returns a NoneType object, because the compile() method does 
  # not return a model, but only updates its values (?)
  # Hence, we need to return model, which is not any more the same as the input
  # that was received.
  compiled_model = model.compile(optimizer = optimizer,
                                loss = loss, 
                                metrics = metric)
  
  return model, optimizer, num_epochs, loss, metric

# Define CALLBACKS
def get_callbacks(include_tensorboard_CB = False, **kwargs):
  """
  Define callbacks for model training:
  - tf.keras.callbacks.TerminateOnNaN 
  - tf.keras.callbacks.EarlyStopping 

  Inputs
  -------
  - include_tensorboard_CB (bool), whether to include the tensorboard_callback, 
    default = False

  *Optional keyword arguments:
  - experiment_dir: (str) directory; where to save the training logs used by TensorBoard.
    e.g. /content/drive/MyDrive/data/saved models/Yannis/BERT/h5_tests/test1
    No slash at the end.

  Return
  -------
  - list of callbacks
  """
  TON = callbacks.TerminateOnNaN() # Callback that terminates training when a NaN loss is encountered

  early_stopping = callbacks.EarlyStopping(monitor = 'val_binary_accuracy', # 'binary_accuracy' calculates how often predictions match binary labels.
                                          min_delta=0.003, # original: 0.005
                                          patience = 3, 
                                          mode = 'max', 
                                          restore_best_weights = True, 
                                          verbose = 1)
  if include_tensorboard_CB == True:
    print("To run TensorBoard, refer to aerobotpackages/BERTTools.py for the necessary bash commands.")
#     # Load the TensorBoard notebook extension
#     import datetime
#     %reload_ext tensorboard

#     # Clear any logs from previous runs
#     %rm -rf ./logs/

#     # 'Unpack' the optional keyword arguments
#     experiment_dir = kwargs['log_dir_name']

#     # Define full path for writing logs
#     log_dir = experiment_dir + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     print("Writing logs in:", experiment_dir + "/logs/fit/")
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
#                                                           histogram_freq=1)
    
#     return [TON, early_stopping, tensorboard_callback]
  
  else:
    return [TON, early_stopping]

def train_transformer(model, train_dataset, validation_data, my_callbacks, num_epochs):
  """
  Input
  ------
  - 'model': compiled transformer model
  - 'my_callbacks': list of callbacks, see output of get_callbacks(). Do not 
    mix up with 'callbacks' which is the name of the loaded Keras package.
  - 'num_epochs' (int): number of epochs to train

  Return
  -------
  - trained_model with updated weights
  - training_history: for plotting the training process
  - exec_time: execution time in minutes

  Note
  -----
  The call() method of a ClassifTransformerModelML class object is called twice,
  once on the training, once on the validation data.
  """
  # Time the function execution
  start_time = time.time()
  print(7*'-', f"Execution started...", 7*'-')

  training_history = model.fit(x = train_dataset,
                              validation_data = validation_data,
                              epochs = num_epochs, 
                              callbacks = my_callbacks,
                              verbose = 1) # use verbose 1 to show the progress bar

  # Calculate and print time duration
  print(7*'-', f"Training finished!", 7*'-')
  end_time = time.time()
  exec_time = np.round((end_time - start_time)/60,1)
  print(f"--- It took {exec_time} minutes --- \n\n")

  # Plot the train history
  plot_train_history(training_history, 'binary_accuracy', '')
  plot_train_history(training_history, 'loss', '')

  # Return the 'model', which has been trained by now.
  # Since the fit() method does not return a model instance, we cannot
  # do this: trained_model = model.fit(...)
  return model, training_history, exec_time

def infer(model, test_dataset, tokenized_dataset, batch_size, threshold, anomalies):
    """
    - Calculate scores on the metrics defined during model.compile()
    - Get multilabel predictions y_pred_proba, i.e. 'num_classes' 
      probabilities for each data entry
    - Convert probabilities y_pred_proba into y_pred, i.e. binary (0,1) 
      multilabels
    - Construct the classification report

    Inputs
    -------
    - test_dataset (tf.data.Dataset): best is to use the attribute of a 
      'DataPrepMultilabelBERT' class instance
    - tokenized_dataset (HuggingFace dataset). Best is to use the attribute 
      of a 'DataPrepMultilabelBERT' class instance
    - batch_size (int): best is to use the attribute of a 
      'DataPrepMultilabelBERT' class instance
    - threshold for the probability --> binary conversion
    - anomalies (list of str): Used to label the classification report.
      Best use the .anomalies attribute of a 'DataPrepMultilabelBERT' class 
      object

    Return
    -------
    - evaluation_scores: output of tensforflow method 'model.evaluate()'
    - y_pred_proba: output of tensforflow method 'model.predict()'. 
      Multilabel probabilities.
    - y_pred: binary (0,1) multilabels
    - y_test: binary (0,1) multilabels
    - clf_rep: classification report in dictionary format

    Notes
    ------
    Ioannis initially created this function as a 'ClassifTransformerModelML'
    class method. Because the latter is a custom model, the method was
    'untraced' and therefore was not part of *reloaded* models. 
    This is why the infer() function is now defined outside the class. 
    """
    print("Evaluation scores on the test set (usually: loss and accuracy):")
    evaluation_scores = model.evaluate(test_dataset)
    print(evaluation_scores, '\n')
    
    print("Predicting multilabel probabilities y_pred_proba...")
    y_pred_proba = model.predict(test_dataset, #tokenized_dataset["test"]['input_ids'], 
                                batch_size = batch_size, 
                                verbose = 1)
    print("Shape of y_pred_proba:", y_pred_proba.shape, '\n')
    print("Example of entry in y_pred_proba:", y_pred_proba[0], '\n')

    print("Converting probabilities into binary (0,1) multilabel 'y_pred' using threshold =", threshold)
    y_pred = y_prob_to_y_pred_ML(y_pred_proba, threshold = threshold)
    print("Example of entry in y_pred:", y_pred[0])

    print("\n Getting y_test from tokenized test dataset in HuggingFace dataset format to build the classification report.\n")
    y_test = tokenized_dataset["test"]['labels']

    # Classification report
    clf_rep = classification_report(y_test, y_pred, output_dict = True)
    print(f"\n\n Classification Report: \n {classification_report(y_test, y_pred, target_names = anomalies)}\n")

    return evaluation_scores, y_pred_proba, y_pred, y_test, clf_rep

def save_transformer(model, experiment_dir):
  """
  Saves the model using model.save(), an alias of tf.keras.models.save_model()
  method

  Inputs
  ------
  - experiment_dir (str): directory of the experiment. The data will be saved here.

  Return
  ------
  - None; this function saves data externally

  Notes on execution
  -------------------
  - The experiment_dir is created automatically, if not already existing.
  - If you are about to overwrite a saved model, you will get a prompt.
    You could change this setting to overwritting by default, e.g. if you let
    a model train for hours and want to make sure it is saved (in case you are 
    not there to type the 'y' inside the promt). 
  - It is normal to see 7x the msg.
    'The call() method of a ClassifTransformerModelML class object 
    has been called.'; It remains unclear why the object is called 7x during save.
  - The file takes some time to appear in the experiment_dir 
    (at least if it's a Google Drive folder).
  - Executing for a model containing BERT, will probably print something like 
    'WARNING:absl:Found untraced functions such as embeddings_layer_call_fn, etc.'
    Apparently there are ~80 functions that 'will not be *directly* callable after loading.'
    Ioannis thinks that this is not a problem, since the loaded model can be 
    successfully compiled and trained.

  Notes on available formats for saving models
  ---------------------------------------------
  There are two formats you can use to save an entire model to disk: 
  - the TensorFlow SavedModel format
  - the older Keras H5 format. 
  
  **The recommended format is TensorFlow SavedModel.**

  SavedModel is the more comprehensive save format that saves 
  - the model architecture, 
  - weights, 
  - the traced Tensorflow subgraphs of the call functions. 
  
  This enables Keras to restore both built-in layers as well as custom objects.
  When saving in TensorFlow SavedModel format, a folder is created containing 
  the files/folders:
  - assets  
  - keras_metadata.pb  
  - saved_model.pb  
  - variables

  On the other hand, HDF5 is a single file containing 
  - the model's architecture, 
  - weights values (which were learned during training), 
  - compile() information (if compile() was called)
  - the optimizer and its state, if any (this enables you to restart training 
  where you left)
  See https://www.tensorflow.org/guide/keras/save_and_serialize
  
  HDF5 is a light-weight alternative to TensorFlow SavedModel with imitations
  (see https://www.tensorflow.org/guide/keras/save_and_serialize#limitations)

  /!\ In any case: DO NOT USE .pkl (pickle) file format for DEEP LEARNING models !

  Notes on implementation
  ------
  Ioannis initially created this function as a 'ClassifTransformerModelML'
  class method. Because the latter is a custom model, the method was
  'untraced' and therefore was not part of *reloaded* models. 
  This is why the infer() function is now defined outside the class. 
  """
  # Define options for saving to SavedModel
  # We save as much info as possible
  tf_save_model_opts = tf.saved_model.SaveOptions(namespace_whitelist=None,
                                                  save_debug_info = True, # default is False
                                                  function_aliases=None,
                                                  experimental_io_device=None,
                                                  experimental_variable_policy=None,
                                                  experimental_custom_gradients=True)

  # model.save() is an alias for tf.keras.models.save_model()
  # Difference between tf.saved_model.save and tf.keras.model.save: none essentially
  model.save(
      filepath = experiment_dir,
      overwrite=True, #False: ask the user with a manual prompt
      include_optimizer=True, # save optimizer's state together
      save_format=None, # Either 'tf' for Tensorflow SavedModel or 'h5' HDF5, defaults to 'tf' in TF 2.X, and 'h5' in TF 1.X.
      signatures=None, # Signatures to save with the SavedModel. Applicable to the 'tf' format only
      options = tf_save_model_opts,
      save_traces=True # when save_traces=False, all custom objects must have defined get_config/from_config methods. When loading, the custom objects must be passed to the custom_objects argument. save_traces=False reduces the disk space used by the SavedModel and saving time.
  )
  print("Model successfully saved in:\n", experiment_dir)
    
def load_saved_transformer(filepath, 
                           compile=True):
  """
  Inputs
  -------
  - filepath (str): directory where to load the model from
  - compile (bool): whether to compile the model after loading, default = True.
    Works only if the original model was compiled, and saved along with 
    the optimizer. 

  Return
  -------
  - loaded transformer model

  Notes
  ------
  Use the 'custom_object' parameter of keras.models.load_model() while loading 
  the model if you've custom layer-like stuff.
  Alternatively, the @tf.keras.utils.register_keras_serializable() decorator can
  be used in the CustomLayer or CumstomModel class definitions,
  see https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
  """
  print(f"Will load the model from directory \n {filepath}\n")
  from tensorflow.keras.models import load_model
  print("Loading model...")
  loaded_model = keras.models.load_model(filepath = filepath, 
                                        custom_objects={"ClassifTransformerModelML": ClassifTransformerModelML, 
                                                        "ConcatSlice": ConcatSlice},
                                        compile = compile 
                                        )
  print("Model successfully loaded.")
  return loaded_model

def train_load_transformer_model(dir_name, experiment_name, 
                                 df,
                                 num_epochs, 
                                 anomalies,
                                 threshold = 0.5, 
                                 train_mode=False, 
                                 load_model=False, 
                                 save_and_overwrite_model=False):
  """
  Inputs
  --------
  - dir_name (str): directory where to save the model, training logs, y_pred, 
  classification report etc.
  - experiment_name (str): experiment name that will be used to name the .pkl files.
    A folder entitled [experiment_name] is created within the directory [dir_name] 
    and contains the files that constitute the TensorFlow SavedModel format. 
    In other words, this folder contains the model with all its assets.
  - df (pd.DataFrame): input data containing at least a column 'Narrative' with 
    the texts and columns of Anomalies in one-hot format
  - num_epochs (int): number of epochs to train
  - anomalies (list of str), anomaly root labels 
  - threshold (float) threshold for probability to boolean conversion 
    during prediciton, default = 0.5
  - train_mode (bool), whether to train a model. Inference only if set to False.
    Default = False
  - load_model (bool), whether to load an existing model. Default = False
  - save_and_overwrite_model (bool), whether to save the trained model. 
    If train_mode == False and load_model = True, 'save_and_overwrite_model' 
    decides whether the clf_rep is saved or not.

    /!\ this will overwrite an existing model located in the same directory!
    Default = False)

  If both train_mode = True and load_model = True, it will train a model, 
  load it from the file and retrain it.

  If train_mode = False and load_model = True, y_pred, clf_rep etc. will be saved
  with 'from_reloaded' included in their filename, so that they can be distinguished
  from the original files, generated during the initial training of the model.

  Return
  ------- 
  None; this function saves data externally 
  """
#   experiment_dir = dir_name + experiment_name
  experiment_dir = Path(dir_name).joinpath(Path(experiment_name))

  if train_mode == True:

    ############################
    # Preprocess for training
    preprocess = DataPrepMultilabelBERT(BERT_model_name = "google/bert_uncased_L-12_H-768_A-12", 
                                        df = df, 
                                        anomalies = anomalies)

    tf_train_dataset, tf_validation_dataset, tf_test_dataset = preprocess(train_mode = train_mode)

    #############################
    # Donwload the BERT model
    get_bert_model = GetBERTModel(BERT_model_name = preprocess.BERT_model_name,
                                num_classes = len(preprocess.anomalies),
                                trainable_layers = [8,9,10,11] # None
                                )
    bert_model = get_bert_model()

    # Build our classification model, including the BERT layer
    transformer_model = ClassifTransformerModelML(num_classes = get_bert_model.num_classes,
                                                  max_length = preprocess.max_length,
                                                  transformer_model = bert_model)
    # get_bert_model.num_classes: attribute of the Class 'GetBertModel' object created above
    transformer_model.summary()
    # if __name__ == '__main__':
    #   transformer_model.summary()
    # If the source file is executed as the main program, 
    # the interpreter sets the __name__ variable to have a value “__main__”. 
    # If this file is being imported from another module, 
    # __name__ will be set to the module’s name.
    # https://www.geeksforgeeks.org/__name__-a-special-variable-in-python/#:~:text=__name__%20is%20one,set%20to%20the%20module's%20name.

    ########################
    # Compile the model
    compiled_transformer_model, optimizer, num_epochs, loss, metric = compile_transformer(transformer_model,
                                                    batch_size = preprocess.batch_size, # use the attribute of the object created above
                                                    train_set_len = preprocess.train_set_len,
                                                    num_epochs = num_epochs, 
                                                    optimizer_type = 'AdamW')

    # If you want to use TensorBoard, define directory where to save the training logs to
    my_callbacks = get_callbacks(include_tensorboard_CB = False, 
                                log_dir_name = experiment_dir)
    #######################
    # Train the model
    trained_transformer_model, training_history, exec_time = train_transformer(compiled_transformer_model, 
                                                                            preprocess.tf_train_dataset,
                                                                            preprocess.tf_validation_dataset,
                                                                            my_callbacks, 
                                                                            num_epochs = num_epochs)


#     ######################
#     # Launch TensorBoard
#     ######################
#     # Change the pwd, so that TensorBoard may locate the logs
#     %cd $experiment_dir

#     #Start TensorBoard through the command line or within a notebook experience. 
#     # The two interfaces are generally the same. In notebooks, use the %tensorboard line magic. 
#     # On the command line, run the same command without "%"
#     %tensorboard --logdir logs/fit
#     # Takes some time to launch

#     # For Comparing different executions of your model see 
#     # https://github.com/tensorflow/tensorboard/blob/master/README.md#runs-comparing-different-executions-of-your-model

#     # # Kill tensorboard (use the appropriate process id)
#     # !kill 2022

    evaluation_scores, y_pred_proba, y_pred, y_test, clf_rep = infer(trained_transformer_model,
                                                                  preprocess.tf_test_dataset,
                                                                  preprocess.tokenized_dataset,
                                                                  preprocess.batch_size,
                                                                  threshold = threshold, 
                                                                  anomalies = preprocess.anomalies)

    # Convert the classification report in form of a pd.DataFrame, adding 
    # additional info
    clf_rep_df = convert_clf_rep_to_df_multilabel_BERT_kw_args(clf_rep, 
                                                        preprocess.anomalies, 
                                                        preprocessing = 'original',
                                                        classifier = 'BERT_BASE',
                                                        undersampling = 0,
                                                        UNfrozen_layers = '9,10,11,12',
                                                        concat_layers = 'None',
                                                        comments = 'last_hidden_state_CLS',
                                                        experiment_ID = experiment_name
                                                        )
    print("\n Showing clf_rep_df.head():\n")
    print(clf_rep_df.head())

    if save_and_overwrite_model == True:
      #################
      # SAVE THE MODEL
      save_transformer(trained_transformer_model, experiment_dir)

      ###############################
      # SAVE THE MULTILABEL OUTPUTS
      save_ML_outputs(dir_name, experiment_name, 
                  y_pred_proba = y_pred_proba, 
                  y_pred = y_pred, 
                  y_test = y_test, 
                  clf_rep = clf_rep,
                  clf_rep_df = clf_rep_df)

      ######################################
      # WRITE EXPERIMENT INFO TO .txt file
      model_attr_to_save = ['BERT_model_name', 'trainable_layers', 
                            'num_classes', 'anomalies', 'batch_size',
                            'max_length', 'emb_to_use', 'layer_to_get', 
                            'layers_to_concat', 'use_CLS_or_flatten']
      class_objects = [preprocess, 
                       get_bert_model, 
                       transformer_model, 
                       transformer_model.concat_slice]

      save_exp_info_to_txt(dir_name = dir_name, 
                      experiment_name = experiment_name, 
                      model_attr_to_save = model_attr_to_save, 
                      model = transformer_model,
                      class_objects = class_objects,
                      num_epochs = num_epochs, 
                      loss = loss, 
                      optimizer = optimizer, 
                      metric = metric,
                      callbacks = my_callbacks,
                      threshold = threshold,
                      execution_time = exec_time)


  if train_mode == True & load_model == True:
    # Delete the freshly trained model from the memory, to see if it is really freshly loaded
    del trained_transformer_model
    # Load a saved model
    loaded_model = load_saved_transformer(filepath = experiment_dir)
    # Fit the loaded model
    loaded_model_retrained, training_history, exec_time = train_transformer(loaded_model,
                                                                            preprocess.tf_train_dataset,
                                                                            preprocess.tf_validation_dataset,
                                                                            my_callbacks)
    # Infer using the retrained loaded model
    evaluation_scores, y_pred_proba, y_pred, y_test, clf_rep = infer(loaded_model_retrained,
                                                                    preprocess.tf_test_dataset,
                                                                    preprocess.tokenized_dataset,
                                                                    preprocess.batch_size,
                                                                    threshold = 0.5, 
                                                                    anomalies = preprocess.anomalies)

  elif train_mode == False:
    ##################################
    # Preprocess for inference only
    preprocess = DataPrepMultilabelBERT(BERT_model_name = "google/bert_uncased_L-12_H-768_A-12", 
                                        df = df, 
                                        anomalies = anomalies)

    tf_train_dataset, tf_validation_dataset, tf_test_dataset = preprocess(train_mode = train_mode)

    if load_model == True:
      # Load a saved model
      loaded_model = load_saved_transformer(filepath = experiment_dir)

      # Infer using loaded model
      # The train and validation datasets are dummy in this case, because 
      # preprocess was called with train_mode = False above.
      evaluation_scores, y_pred_proba, y_pred, y_test, clf_rep = infer(loaded_model,
                                                                    preprocess.tf_test_dataset,
                                                                    preprocess.tokenized_dataset,
                                                                    preprocess.batch_size,
                                                                    threshold = 0.5, 
                                                                    anomalies = preprocess.anomalies)
      
      # Convert the classification report in form of a pd.DataFrame, adding 
      # additional info
      clf_rep_df = convert_clf_rep_to_df_multilabel_BERT_kw_args(clf_rep, 
                                                              preprocess.anomalies, 
                                                              preprocessing = 'original',
                                                              classifier = 'BERT_BASE',
                                                              undersampling = 0,
                                                              UNfrozen_layers = '9,10,11,12',
                                                              concat_layers = 'None',
                                                              comments = 'inference_on_FINAL_test_set',
                                                              experiment_ID = experiment_name
                                                              )
      
      if save_and_overwrite_model == True:
        ###############################
        # SAVE THE MULTILABEL OUTPUTS
        save_ML_outputs(dir_name, experiment_name + '_infer_FINAL_test_set', 
                    y_pred_proba = y_pred_proba, 
                    y_pred = y_pred, 
                    y_test = y_test, 
                    clf_rep = clf_rep,
                    clf_rep_df = clf_rep_df)

      
#####################################################################################################
# AUXILIARY FUNCTIONS
#####################################################################################################

def plot_train_history(training_history, metric, anomaly_name):
  """
  Generete plots to monitor the train process
  Inputs: 
  - 'training_history'; use training_history = model.train(...)
  - 'metric' to plot; string e.g. 'accuracy', 'loss'
  - 'anomaly_name' e.g. 'Anomaly_Conflict'. This is used for the plot title
  """
  fig = plt.figure(figsize = (10,4))
  #plt.title(f"{anomaly_name} train history - {metric.upper()}", fontsize = 20)
  train_acc = training_history.history[metric]
  val_acc = training_history.history['val_' + metric] # e.g. 'val_accuracy'

  plt.plot(train_acc, label = f'Training {metric}')
  plt.plot(val_acc, label = f'Validation {metric}')
  plt.xlabel('epochs')
  plt.ylabel(f'{metric}')
  plt.legend()
  plt.show();

def y_prob_to_y_pred_ML(y_pred_proba, threshold = 0.5):
  """
  Converts probabilities into 0's and 1's. We are still in the MULTILABEL context.
  Input: MULTILABEL predictions (probabilities whose sum for each sample may exceed > 1) coming directly from the model
  Using a user-defined threshold, return a MULTILABEL prediction vector 'y_pred' containing 0's and 1's
  """
  y_pred=[]
  for sample in y_pred_proba:
    y_pred.append([1 if i>= threshold else 0 for i in sample])
  y_pred = np.array(y_pred)

  return y_pred

def y_prob_to_y_pred(y_pred_proba, threshold = 0.5):
  """
  Converts probabilities into 0's and 1's. We are still in the BINARY context.
  Input: monolabel predictions (probabilities), dimension = #samples
  Using a user-defined threshold, return a prediction vector 'y_pred' with 
  dimension = #samples, containing a '0' or a '1' for each sample
  """
  y_pred=[]
  for value in y_pred_proba:

    if value >= threshold:
      y_pred.append(1)
    else :
      y_pred.append(0)

  y_pred = np.array(y_pred)

  return y_pred

def create_dir_if_not_exists(path):
  """
  Check if the directory 'path' exists and create if necessary.
  """
  import os
  if not os.path.exists(path):
    # Create a new directory because it does not exist
    os.makedirs(path)
    print(f"New directory created:\n {path} \n")
    
def save_ML_outputs(dir_name, experiment_name, **kwargs):
    """
    Save multilabel classification outputs (y_pred_proba, y_pred, clf_rep) and target
    variable y_test.
    Uses **kwargs, so that the user may save only a part of the variables.

    Inputs
    -------
    - 'dir_name' (str): root directory, with slash at the end
    - 'experiment_name' (str): subdirectory, appended to 'dir_name'. Is also
      part of the .pkl file's name.
    - additional keyword arguments:
      - y_pred_proba
      - y_pred
      - y_test
      - 'clf_rep' (classification report in 
      dictionary format)
      - 'clf_rep_df' (classification report in 
      pd.DataFrame format)
    """
    create_dir_if_not_exists(dir_name)

    # 'Unpack' the optional keyword arguments
    # kwargs behaves like a dictionary
    for key, val in zip(kwargs.keys(), kwargs.values()):
      print("Saving", str(key), "...")
      filename_pkl = experiment_name + '_' + str(key) +  '.pkl'
      path_and_filename_pkl = dir_name + filename_pkl
      pkl.dump(val, open(path_and_filename_pkl, 'wb'))

    print("\n")
    print("Multilabel results were successfully saved in \n", dir_name)

def convert_clf_rep_to_df_multilabel_BERT_kw_args(clf_rep, anomalies, **kwargs):
  '''
  Return the classification report in form of a pd.DataFrame.
  Tailored for extracting MULTILABEL BERT experiment results.
  The DataFrame contains dditional columns containing experiment information.
  Improvement: Their construction should be automatised as much as possible.
  
  Inputs
  -------
  - multilabel classification report in dictionary format
  (does not contain '0' and '1' keys)
  - anomalies (list of str): Used to label the classification report.
    Best use the .anomalies attribute of a 'DataPrepMultilabelBERT' class 
    object.
  - additional keyword arguments. Pass the ones you wish and even additional ones.
    They are all automatically unpacked, a column is created in the DataFrame, with 
    col_name = str(keyword_of_the_argument) and value = kwarg_value.
    
    Here is a nonexhaustive list of possible kwargs in the context of our 
    transformers:
    - classifier (str), e.g. 'BERT_BASE' or 'DistilBERT' 
    - preprocessing (str), e.g. 'original' or 'raw_stem' or 'PP'. 'original' in AeroBot
      means no preprocessing at all, not even tokenization or stemming
    - undersampling (int), 0 or 1 if undersampling was applied

    - UNfrozen_layers (str), e.g. '9_10_11_12' if the last 4 layers were trained, 
      str(None) if all layers were frozen. 
      You can pass 'str(model.trainable_layers)'.
      We use str() in case the value is 'None', other wise it is just empty.
      Layers run from 1 to 12!
    - concat_layers (str), whether / which layers were concatenated, e.g. str(None) 
      or '8_9_10_11'
      You can pass 'str(trained_transformer_model.concat_slice.layers_to_concat)'
      We use str() in case the value is 'None', other wise it is just empty.
      Layers run from 1 to 12!
    - comments (str), misc. comment, e.g. 'last_hidden_state_CLS_random_state_222'
      or 'Flatten layer X' or 'max_length_345' or 'last_hidden_state_CLS'
    - experiment_ID (str), e.g. '7_3_9_4'
    - padding (str), padding setting, e.g. 'pre' or 'post'
    - truncating (str), truncation setting, e.g. 'pre' or 'post'
    - maxlen (int), max_length value (length of tokenized sequence)   

  Return
  -------
  - classification report in form of a pd.DataFrame with additional columns 
    containing experiment information
  '''
  # write classification report dictionnary into pd.DataFrame
  metrics = pd.DataFrame(clf_rep)

  # The rest of the code is basically kind of 'transposing' the format 
  # and adding extra columns with parameter values

  # Rename columns with anomaly names
  # Crete dictionary with correspondance among label indices and anomaly names
  anomaly_labels = dict(zip(metrics.columns[0:len(anomalies)], anomalies))
  metrics = metrics.rename(columns = anomaly_labels)

  #####################################################################
  # Create DataFrame in the right format for the plotting of results
  #####################################################################
  clf_rep_df = pd.DataFrame()
  for anomaly in metrics.columns[0:len(anomalies)]:

    temp_df = pd.DataFrame(index = metrics.index) # create temporary DataFrame with the 4 metrics as index
    temp_df['values'] = metrics.filter(items = [anomaly]).values # write the 4 values for the selected anomaly
    temp_df['anomaly'] = anomaly # fill in the column with the selected anomaly label
    clf_rep_df = pd.concat([clf_rep_df, temp_df])

  clf_rep_df = clf_rep_df.reset_index().rename(columns = {'index': 'metric'})

  # Fill in additionnal columns with metadata by 'unpacking' the 
  # keyword arguments
  for key, val in zip(kwargs.keys(), kwargs.values()):
    clf_rep_df[str(key)] = val        # 'BERT_BASE' or 'DistilBERT'

  print("Classification report successfully converted into DataFrame of length:", len(clf_rep_df)) #should be 56 = 14 anomalies * 4 metrics
  return clf_rep_df    

def save_exp_info_to_txt(dir_name, experiment_name, model_attr_to_save, 
                         class_objects, model, include_model_summary = True,
                         **kwargs):
    """
    Save experiment information to a .txt file for future reference.

    Inputs
    -------
    - 'dir_name' (str): root directory, with slash at the end
    - 'experiment_name' (str): subdirectory, appended to 'dir_name'. Is also
      part of the .txt file's name.
    - model_attr_to_save (list of strings): model attributes to save, e.g. 
      'BERT_model_name', 'trainable_layers', 'num_classes', 'anomalies', 
      'batch_size','max_length', ...
    - class_objects (list of class objects): class objects, where to look for the 
      'model_attr_to_save'
    - model (transformer model object): used to generate the model.summary() and
      write it to the file  
    - include_model_summary (bool) whether to include the model summary in the 
      .txt file. Default = True
    - additional keyword arguments that should be saved in the .txt file

    Return
    -------
    None; creates a .txt file in directory 'dir_name'.
    """
    create_dir_if_not_exists(dir_name)
    
    ###################################################################
    # PREPARE THE DATA TO BE SAVED INTO A DICTIONARY 'dict_for_export'
    ###################################################################
    # Initialize empty dictionary
    dict_for_export = dict({})

    # Loop through the class objects
    for obj in class_objects:
      # Get the object's attributes in form of a dictionary
      d = obj.__dict__
      # Select only the elements whose key is in 'model_attr_to_save'
      param_dict = {k:d[k] for k in model_attr_to_save if k in d}
      # 'Append' to the dictionary 'dict_for_export'
      dict_for_export.update(param_dict) # if two keys are the same (e.g. 'max_length'), it overwrites the value

    # Append additional elements that are not class attributes, passed via the 
    # kwargs
    for key, val in zip(kwargs.keys(), kwargs.values()):
      dict_for_export.update({str(key):val})

    ###########################
    # WRITE DATA TO .txt file
    ###########################
    # Write dictionary to .txt file
    filename = dir_name + experiment_name + '_exp_info.txt'
    with open(filename, 'w') as f:
        print(dict_for_export, file = f)

    if include_model_summary == True:
      # Append the model summary to the .txt file
      with open(filename, 'a') as f: # 'a' stands for append; prevents overwritting
      # print(dict_for_export, file = f)
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    print("Experiment information successfully written to .txt file located in:\n", dir_name)

def get_cmap(n, name='tab20'):
    '''Used for plot. Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)       
      
def y_multilabel_to_binary(y_test, y_pred_proba, cls_idx):
  """
  For a given class with index 'cls_idx', convert true labels and predicted 
  probabilities from multilabel to binary classification format.
  Used to plot Receiver-Operating Curves (ROCs)

  Inputs:
  - y_test: ndarray, shape (#samples, # classes) containing 0 or 1, e.g. array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    Each row may contain several '1' (multilabel)
  - y_pred_proba: ndarray, shape (#samples, # classes) containing probabilities for each class, as return from our model. 
    The probabilities for a given sample may sum to >1 in multilabel problems.
  - cls_idx: class index (int). Pick the class of interest.

  Return
  - y_test_cls: true labels for the given class
  - probs_cls: probabilities for the given class
  """
  # Initialize variables
  y_test_cls = []
  probs_cls = []

  for i in range(len(y_test)): # Loop through the input test-set
    # Pick the values corresponding to the class of interest
    y_test_cls.append(y_test[i][cls_idx])
    probs_cls.append(y_pred_proba[i][cls_idx])

  y_test_cls = np.array(y_test_cls)
  probs_cls = np.array(probs_cls)

  return y_test_cls, probs_cls

def find_opt_threshold_PR(precision, recall, thresholds):
  """
  Find the optimal threshold in a given Precision Recall (PR) Curve
  by determining the threshold that yields the min distance to the top-right
  point (1,1) of the PR curve.

  Returns the threshold value
  """
  dist = [] # distance from the point (precision = 1; recall = 1)
  for p, r in zip(precision, recall):
    dist.append(np.sqrt( (1 - p)**2 + (1 - r)**2 ))

  dist = np.array(dist)

  # Find the threshold yielding the min distance
  min_dist_idx = np.argmin(dist)
  min_dist = dist[min_dist_idx]
  optimum_threshold = thresholds[min_dist_idx]
  optimum_precision = precision[min_dist_idx]
  optimum_recall = recall[min_dist_idx]
  
  return optimum_threshold, optimum_precision, optimum_recall

def get_list_of_opt_thresholds(anomalies, y_test, y_pred_proba):
  """
  Calculate the optimal thresholds for each anomaly label using 
  find_opt_threshold_PR(). Append optimal threshold, precision and recall values
  to a list.

  Inputs
  -------
  - anomalies (list of str)
  - y_test (list of int)
  - y_pred_proba (nd.array) each element contains [num_classes] probabilities

  Return
  ------
  - opt_thresholds (list of float)
  - opt_precisions (list of float)
  - opt_recalls (list of float)
  """
  from sklearn.metrics import precision_recall_curve

  # Iterate through classes and find the optimal threshold and corresponding precision, recall
  opt_thresholds = []
  opt_precisions = []
  opt_recalls = []

  for anomaly, cls_idx in zip(anomalies, range(len(anomalies))):

    # Compute precision-recall pairs for different probability thresholds.
    y_test_cls, probs_cls = y_multilabel_to_binary(y_test, y_pred_proba, cls_idx)

    # Calculate precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_test_cls, probs_cls)
    
    # Get optimal thresholds
    optimum_threshold, optimum_precision, optimum_recall = find_opt_threshold_PR(precision, recall, thresholds)
    
    opt_thresholds.append(optimum_threshold)
    opt_precisions.append(optimum_precision)
    opt_recalls.append(optimum_recall)

  return opt_thresholds, opt_precisions, opt_recalls

def plot_PR_curve_opt_thresh(anomalies, y_test, y_pred_proba, opt_thresholds, opt_precisions, opt_recalls):
  """
  Plot together the Precision-Recall curves for all anomaly labels. 
  'optimum_threshold' are plotted as stars on the curves at the coordinates (optimum_recall, optimum_precision).
  The AUC and average precision (AP) scores are calculated for each label.
  Includes iso-f1-score lines in the background.
  
  Inputs
  -------
  - anomalies (list of str)
  - y_test (list of int)
  - y_pred_proba (nd.array) each element contains [num_classes] probabilities
  - opt_thresholds (list of float): optimal thresholds for all anomaly labels
  - opt_precisions (list of float): optimal precisions for all anomaly labels
  - opt_recalls (list of float):  optimal recalls for all anomaly labels
  
  Return
  -------
  None; prints a plot.
  """
  from sklearn import metrics

  # Create a colormap
  num_classes = len(anomalies)
  cmap = get_cmap(num_classes)

  # Instantiate figure
  fig = plt.figure(figsize = (17,15))

  lines = []  # Instantiate list of plot lines (i.e. curves) 
  labels = [] # Instantiate list of plot labels

  ###########################################################################
  ### PLOT iso-f1 lines #######
  f_scores = np.linspace(0.2, 0.8, num=4)
  for f_score in f_scores:
      x = np.linspace(0.01, 1)
      y = f_score * x / (2 * x - f_score)
      l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha = 0.4, zorder = 2)
      plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02), size=28, alpha = 0.4)

  # Store in our lists for now, we call them later
  lines.append(l)
  labels.append('iso-f1 curves')

  # Plot diagonal
  l, = plt.plot([0,1], [0,1], '--k', alpha = 0.5)
  lines.append(l)
  labels.append('y = x')

  ###########################################################################
  ### PLOT precision-recall lines #####
  for anomaly, cls_idx in zip(anomalies, range(len(anomalies))):

    # Compute precision-recall pairs for different probability thresholds.
    y_test_cls, probs_cls = y_multilabel_to_binary(y_test, y_pred_proba, cls_idx)

    # Calculate precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_test_cls, probs_cls)
  
    # calculate precision-recall AUC (Area Under Curve)
    auc = np.round(metrics.auc(recall, precision), 3)

    # calculate the average precision (AP)
    average_precision = metrics.average_precision_score(y_test_cls, probs_cls)
    average_precision = np.round(average_precision, 3)

    color = cmap(cls_idx) # use the same cmap as in the ROC curve
    l, = plt.plot(recall, precision, c = color, zorder=3) 
    # ax.plot() returns a tuple which contains only one element. 
    # If you assign it without the comma, you just assign the tuple.
    # using the comma, you unpack the tuple and get its element

    lines.append(l)
    labels.append(f'{anomalies[cls_idx][8:]}, AUC: {auc}, AP: {average_precision}') # slice the string, to ommit 'Anomaly_'

    # Plot optimum thresholds
    x = opt_recalls[cls_idx]
    y = opt_precisions[cls_idx]
    s = plt.scatter(x, y, marker = '*', s = 200, color = color)

    plt.annotate('{0:0.2f}'.format(opt_thresholds[cls_idx]), xy = (x + 0.005, y + 0.005), color = color, size=30, alpha = 1)
    
  lines.append(s)
  labels.append('optimal thresholds')
    
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.title('Precision-recall Curve with optimal thresholds')
  plt.xlabel('recall')
  plt.ylabel('precision')

  plt.legend(lines, labels, bbox_to_anchor=(1, 0.8)); # here we use our lists of lines and labels
#################################################################################################   
