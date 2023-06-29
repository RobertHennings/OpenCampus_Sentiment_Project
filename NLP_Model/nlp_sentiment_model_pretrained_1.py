# Here the NLP Model itself is specified
# Using a pretrained model setup since we are faced with unlabeled texts
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, TFBertModel
import tensorflow as tf
import tensorflow_hub as hub
import glob
import datetime as dt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

def build_sentiment_model():
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = TFBertModel.from_pretrained(model_name)

    # Define the additional neural network layers
    input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    bert_output = bert_model(input_layer)[0]
    pooled_output = tf.keras.layers.GlobalMaxPooling1D()(bert_output)
    dropout_layer = tf.keras.layers.Dropout(0.5)(pooled_output)
    dense_layer_1 = tf.keras.layers.Dense(256, activation='relu')(dropout_layer)
    dense_layer_2 = tf.keras.layers.Dense(128, activation='relu')(dense_layer_1)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer_2)

    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


# model.summary()
def classify_sentiment(model, text):
    # Tokenize the text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors='tf')

    # Make prediction
    prediction = model.predict(encoded_input['input_ids'])[0][0]

    # Process prediction
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    confidence = np.round(prediction * 100, 2)

    return sentiment, confidence

# Example usage:
model = build_sentiment_model()  # Instantiate the model using the function
text = "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported"
# negative
sentiment, confidence = classify_sentiment(model, text)
print(f"Sentiment: {sentiment}, with a confidence of: {confidence}%")

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

features = text_data
labels = list(text_labels)
train_size = 0.50
seed = 256
set_seed = False

train_X, train_y, test_X, test_y = train_test_split(features, labels, train_size, set_seed, seed)


def compute_err(test_X, test_y, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    result_test_score = []
    result_test_label = []

    err = []
    for t in test_X:
        sentiment, confidence = classify_sentiment(model, t)

        result_test_score.append(confidence)
        result_test_label.append(sentiment)

    for er, tr in zip(result_test_label, test_y):
        if er != tr:
            err.append(1)
        else:
            err.append(0)
    return result_test_score, result_test_label, err


result_test_score, result_test_label, err = compute_err(test_X[:40], test_y[:40], model)

print(f"Error rate: {(sum(err) / len(err)) *100} %")