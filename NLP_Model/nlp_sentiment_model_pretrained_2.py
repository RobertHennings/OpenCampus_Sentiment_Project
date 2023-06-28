from transformers import pipeline
import pandas as pd
from nltk.corpus import stopwords
import glob
text = "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported"
# Negative
sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline(text)[0]

# Print the results
print(f'Sentiment: {result["label"]}, with a Score of: {result["score"]}')

# Test the model on a broader scale and compute some error metrics
glob.os.chdir("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//")
from common_utils import remove_stopwords
from common_utils import train_test_split
from common_utils import read_text_files_FinPhrase

# Next load the text data with its labels and train a model with it
# glob.os.chdir("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//")
file_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Raw_Data//Text_Classification//FinancialPhraseBank-v1.0"
file_name = "Sentences_50Agree.txt"
encoding = "ISO-8859-1"
save = False
save_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Formatted_Data//Text_Classification"


text_data, text_labels = read_text_files_FinPhrase(file_path, file_name, encoding, save, save_path)

# Next encode the labels into numbers and extract the stop words from the single sentences
# Remove all the stop words in every single sentence
stopwords = stopwords.words('english')

text_data_stops_removed = remove_stopwords(text_data, stopwords)


len(text_labels) == len(text_data_stops_removed)

features = text_data_stops_removed
labels = list(text_labels)
train_size = 0.50
seed = 256
set_seed = False

train_X, train_y, test_X, test_y = train_test_split(features, labels, train_size, set_seed, seed)


def compute_err(test_X, test_y, model):
    result_test_score = []
    result_test_label = []

    err = []
    for t in test_X:
        result_test_score.append(model(t)[0]["score"])
        result_test_label.append(model(t)[0]["label"].lower())

    for er, tr in zip(result_test_label, test_y):
        if er != tr:
            err.append(1)
        else:
            err.append(0)
    return result_test_score, result_test_label, err

model = pipeline("sentiment-analysis")

result_test_score, result_test_label, err = compute_err(test_X[:40], test_y[:40], model)

print(f"Error rate: {(sum(err) / len(err)) *100} %")
