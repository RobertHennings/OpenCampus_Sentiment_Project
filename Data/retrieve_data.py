# This file should combine all the data retrieving processes
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# glob.os.getcwd()
glob.os.chdir("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data")
# US Market
# First function loads all the single txt files and stores them into one big dataframe
from get_data_short import get_single_txt_filesFinra
# Second function seprates the masterdataframe into a single df holding info for only one specified ticker
from get_data_short import get_single_ticker_ts

glob.os.chdir("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//")
from common_utils import read_in_all_files_from_dir
from common_utils import safe_single_ts

from common_utils import read_text_files_FinPhrase
from common_utils import remove_stopwords
from common_utils import train_test_split

# Month/Date/Year
startDate = "08/04/2011"
endDate = "06/10/2023"
# endDate = "01/06/2010"
datasetName = "FNSQshvol"
save_local = True
save_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Raw_Data//USMarket//Short_Volume"
override_files = True

master = get_single_txt_filesFinra(startDate, endDate, datasetName, save_local, save_path, override_files)

# Load all saved .csv files from directory and filter for single stock timeseries
directory_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data/Raw_Data//USMarket//Short_Volume"
# Month/Date/Year
startDate = "01/04/2010"
endDate = "08/04/2011"

short_data = read_in_all_files_from_dir(directory_path, startDate, endDate)

# Now create the single stock specific timeseries data and save it
# Find all unique stock ticker
save_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Formatted_Data//Short_Volume"
unique_ticker_list = short_data.Symbol.unique()
master_df = short_data.copy()

def safe_single_ts(save_path: str, unique_ticker_list: list, master_df: pd.DataFrame):
    for tick in unique_ticker_list:
        get_single_ticker_ts(master_df, "Symbol", tick).to_csv(save_path + "//" + tick + ".csv")

safe_single_ts(save_path, unique_ticker_list, master_df)


# Next load the text data with its labels and train a model with it
# glob.os.chdir("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//")
file_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Raw_Data//Text_Classification//FinancialPhraseBank-v1.0"
file_name = "Sentences_50Agree.txt"
encoding = "ISO-8859-1"
save = False
save_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Formatted_Data//Text_Classification"


text_data, text_labels = read_text_files_FinPhrase(file_path, file_name, encoding, save, save_path)

# Plot Histogram to get a broad overview of the class distribution
plt.hist(text_labels, density=False, color="#9b0a7d")
plt.title("Distribution of Data classes")
plt.xlabel("Data Classes")
plt.ylabel("Absolute Number")
plt.show()

round((pd.Series(text_labels).value_counts() / len(text_labels)) * 100, 1)

# Word Cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create separate lists for each label class
positive_sentences = [text_data[i] for i, text_labels in enumerate(text_labels) if text_labels == "positive"]
neutral_sentences = [text_data[i] for i, text_labels in enumerate(text_labels) if text_labels == "neutral"]
negative_sentences = [text_data[i] for i, text_labels in enumerate(text_labels) if text_labels == "negative"]

# Function to generate word clouds
def generate_word_cloud(sentences):
    # Concatenate all sentences into a single string
    text = ' '.join(sentences)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Generate word clouds for each label class
generate_word_cloud(positive_sentences)
generate_word_cloud(neutral_sentences)
generate_word_cloud(negative_sentences)






# Next encode the labels into numbers and extract the stop words from the single sentences
replace_dict = {"negative":0,
                "neutral":1,
                "positive":2}

text_labels = pd.Series(text_labels).replace(replace_dict).values

# Remove all the stop words in every single sentence
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stopwords = stopwords.words('english')

text_data_stops_removed = remove_stopwords(text_data, stopwords)


len(text_labels) == len(text_data_stops_removed)

features = text_data_stops_removed
labels = list(text_labels)
train_size = 0.60
seed = 256
set_seed = False

train_X, train_y, test_X, test_y = train_test_split(features, labels, train_size, set_seed, seed)
