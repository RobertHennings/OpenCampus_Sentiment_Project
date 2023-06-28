import pandas as pd
import requests
import time
import glob
import numpy as np
import glob
import pandas as pd
import nltk
import re
import random
import matplotlib.pyplot as plt


def read_in_all_files_from_dir(directory_path: str, startDate: str, endDate: str) -> pd.DataFrame:
    """Reads in all .csv files from the specfied directory and appends the contents to
       one big dataframe, files are saved daily and hold the information for every single
       security, we later split these files into the stock specific timeseries data

    Args:
        directory_path (str): Directory where all the single .csv files are located
        startDate (str): Start from where onto read the files in
        endDate (str): End until which to read the files in

    Returns:
        pd.DataFrame: master dataframe that holds the information per day for the
                      every security
    """
    date_range = pd.date_range(start=startDate, end=endDate, freq="B")
    # Still failures in files: 20100505, 20100506, 20100507, 20100514
    date_range = date_range.drop(pd.Timestamp("2010-05-05"))
    date_range = date_range.drop(pd.Timestamp("2010-05-06"))
    date_range = date_range.drop(pd.Timestamp("2010-05-07"))
    date_range = date_range.drop(pd.Timestamp("2010-05-14"))


    all_files = list(glob.os.listdir(directory_path))
    master_df_1 = pd.DataFrame()
    master_df_2 = pd.DataFrame()

    for d in date_range:
        if d < pd.Timestamp("2011-08-04"):
            try:
                master_df_1 = master_df_1.append(pd.read_csv(directory_path + "//" + d.strftime('%Y%m%d') + ".csv", index_col=0))
            except:
                print(f"{d.strftime('%Y%m%d')} not available")
        else:
            try:
                master_df_2 = master_df_2.append(pd.read_csv(directory_path + "//" + d.strftime('%Y%m%d') + ".csv", index_col=0))
            except:
                print(f"{d.strftime('%Y%m%d')} not available")
    master_df_all = pd.concat([master_df_1, master_df_2])

    return master_df_all


def read_text_files_FinPhrase(file_path: str, file_name: str, encoding: str, save: bool, save_path: str) -> list:
    """Reads in .txt files from the Fin Phrase dataset

    Args:
        file_path (str): Directory where the FinPhrase .txt files are located
        file_name (str): Specific file name to read in
        encoding (str): Specify encoding for simpler handling
        save (bool): Formatted Data can be saved directly for ease of use later
        save_path (str): Directory where the formatted data should be saved

    Returns:
        list: Returns the text data as list of str elements, list of str text labels 
    """    
    f = open(file_path + "//" + file_name, "r", encoding = encoding)
    lines = f.readlines()
    f.close()
    print(f"File: {file_name} succesfully read in")
    text_data = []
    text_label = []

    for l in lines:
        text_data.append(l.split("@")[0])
        text_label.append(l.split("@")[1].replace("\n", ""))

    if save:
        pd.Series(text_data.to_csv(save_path + "//" + "formatted_text_data_" + file_name.split(".txt")[0] + ".csv"))
        pd.Series(text_data.to_csv(save_path + "//" + "formatted_text_labels_" + file_name.split(".txt")[0] + ".csv"))
    print(f"Formatted File: {file_name} succesfully saved in: {save_path}")
    
    print(f"File: {file_name} succesfully formatted")
    return text_data, text_label

# also include removing numbers more stricly
def remove_stopwords(text_data: list, stopwords: list) -> list:
    stop_words = [word.lower() for word in stopwords]
    stop_words.append("")
    word_tokens = []

    for sent in text_data:
        sent = re.sub(r"[^a-zA-Z0-9 ]", '', sent)
        sent = sent.strip()
        split_sent = sent.split(" ")

        word_tokens_sent = [word for word in split_sent if word.lower() not in stop_words]
        word_tokens.append(word_tokens_sent)
        # for token in word_tokens_sent:
        #     word_tokens.append(token)
    return word_tokens


def train_test_split(features: list, labels: list, train_size: float, set_seed: bool, seed: int) -> list:
    """Custom train test split for textual data stored in a list

    Args:
        features (list): List of Features, single str sentences
        labels (list): List of Labels, single str sentiments (positive, neutral, negative)
        train_size (float): Size of the training dataset
        set_seed (bool): Seed for reproducability
        seed (int): Seed number

    Returns:
        list: 4 lists of the sliced data for training and testing
    """    
    if set_seed:
        np.random.seed(seed)

    ind = np.random.randint(low=0, high=len(features)-1, size=int(train_size * len(features)))

    train_features = []
    train_labels = []

    test_features = []
    test_labels = []

    for i in range(len(labels)):
        if i in ind.tolist():
            train_features.append(features[i])
            train_labels.append(labels[i])
        else:
            test_features.append(features[i])
            test_labels.append(labels[i])

    return train_features, train_labels, test_features, test_labels


def plot_metrics(history: pd.DataFrame) -> plt.plot:
    """Plot the different stored metrics from the model training for evaluation

    Args:
        history (pd.DataFrame): Hist dataframe that was stored in the Logs directory for every model to analyze
    """
    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label='Training Loss', color="#9b0a7d")
    plt.plot(history['val_loss'], label='Validation Loss', color="black")
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history['accuracy'], label='Training Accuracy', color="#9b0a7d")
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color="black")
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Training
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label='Training Loss', color="#9b0a7d")
    plt.plot(history['accuracy'], label='Training Accuracy', color="black")
    plt.title('Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/ Accuracy')
    plt.legend()
    plt.show()

    # Validation
    plt.figure(figsize=(8, 6))
    plt.plot(history['val_loss'], label='Validation Loss', color="#9b0a7d")
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color="black")
    plt.title('Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/ Accuracy')
    plt.legend()
    plt.show()

    # Additional metrics
    for metric_name in history:
        if metric_name not in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
            plt.figure(figsize=(8, 6))
            plt.plot(history[metric_name], label=f'Training {metric_name}')
            plt.plot(history[f'val_{metric_name}'], label=f'Validation {metric_name}')
            plt.title(metric_name)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
            plt.show()


def classify_sentiment(model, text: str, tokenizer_model):
    """Testing of a trained model together with a provided text sample

    Args:
        model (tensorlfow model object): Trained Model to be tested with the text sample
        text (str): Text Sample that should be classified
        tokenizer_model (_type_): Model Name to tokenize the given text sample with

    Returns:
        _type_: Sentiment label of the Text, Probability of reult
    """
    # Tokenize the text
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors='tf')

    # Make prediction
    prediction = model.predict(encoded_input['input_ids'])[0][0]
    
    # Process prediction
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    confidence = np.round(prediction * 100, 2)
    
    return sentiment, confidence


