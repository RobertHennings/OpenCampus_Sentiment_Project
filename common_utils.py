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

def read_in_all_files_from_dir(directory_path: str, startDate: str, endDate: str) -> pd.DataFrame:
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


def read_text_files_FinPhrase(file_path: str, file_name: str, encoding: str, save: bool, save_path: str):
    f = open(file_path + "//" + file_name, "r", encoding = encoding)
    lines = f.readlines()
    f.close()

    text_data = []
    text_label = []

    for l in lines:
        text_data.append(l.split("@")[0])
        text_label.append(l.split("@")[1].replace("\n", ""))

    if save:
        pd.Series(text_data.to_csv(save_path + "//" + "formatted_text_data_" + file_name.split(".txt")[0] + ".csv"))
        pd.Series(text_data.to_csv(save_path + "//" + "formatted_text_labels_" + file_name.split(".txt")[0] + ".csv"))

    return text_data, text_label


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


def train_test_split(features: list, labels: list, train_size: float, set_seed: bool, seed: int):
    if set_seed:
        np.random.seed(seed)

    ind = np.random.randint(low=0, high=len(features)-1, size=int(train_size * len(features)))

    train_features = []
    train_labels = []

    test_features = []
    test_labels = []

    for i in range(len(ind)):
        if i in ind.tolist():
            train_features.append(features[i])
            train_labels.append(labels[i])
        else:
            test_features.append(features[i])
            test_labels.append(labels[i])

    return train_features, train_labels, test_features, test_labels
