# project-specific functions and classes created in the 'Aerobot' project by Ioannis STASINOPOULOS
# This file contains functions and classes for the BERT implementation

#######################
# Import packages
#######################
import numpy as np

###########################################
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
