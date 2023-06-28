# Here the NLP Model itself is specified
# Using a pretrained model setup since we are faced with unlabeled texts
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras import regularizers

def nlp_model(seed: int, set_seed: bool, train_data: list, test_data: list,
              max_sequence_len: int, train_labels: list, test_labels: list,
              dropout: float, epochs: int, save_logs: bool, save_path_logs: str,
              save_model_summary: bool):
    # Set the random seed for reproducibility
    if set_seed:
        SEED = seed
        tf.random.set_seed(SEED)

    # Load the Fin dataset and split it into training and testing sets
    train_data = train_data  
    test_data = test_data   

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)

    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_data)
    test_sequences = tokenizer.texts_to_sequences(test_data)

    # Pad sequences to the same length
    MAX_SEQUENCE_LENGTH = max_sequence_len  # Set the desired sequence length
    train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Prepare the labels
    train_labels = train_labels  
    test_labels = test_labels   

    train_labels = tf.constant(train_labels)
    test_labels = tf.constant(test_labels)
    # Set the hyperparameters
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    DROPOUT = dropout

    # Define the neural network model
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True)))
    model.add(Dropout(DROPOUT))
    model.add(Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True)))
    model.add(Dropout(DROPOUT))
    model.add(Bidirectional(LSTM(HIDDEN_DIM)))
    model.add(Dense(OUTPUT_DIM, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    BATCH_SIZE = 64
    EPOCHS = epochs
    # Save the model history
    history = model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_data, test_labels))

    now = dt.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    glob.os.mkdir(save_path_logs + "//" + "Mod_" + now)

    if save_model_summary:
        with open(save_path_logs + "//" + "Mod_" + now + "//" + f"Mod_summary_df_{now}.txt", "a") as s:
            model.summary(print_fn=lambda x: s.write(x + '\n'))
            s.close()

    # Save the model history into tze specified place with timestamp for identification
    if save_logs and EPOCHS >= 7:
        hist_df = pd.DataFrame(history.history)
        hist_df["Mod_Params"] = [MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT, BATCH_SIZE, EPOCHS] + (len(hist_df) - 7) * [np.nan]
        hist_df.to_csv(save_path_logs + "//" + "Mod_" + now + "//" + f"Mod_hist_df_{now}.csv")

    else:
        print(f"Not able to save the model logs, because too few epochs ({EPOCHS}) for appending model parameters to Log Dataframe, at least 7 needed!")

    return model, history




glob.os.chdir("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//")
from common_utils import remove_stopwords
from common_utils import train_test_split
from common_utils import read_text_files_FinPhrase
from common_utils import plot_metrics
from common_utils import classify_sentiment
from common_utils import down_sample_imbalanced_dataset

# Next load the text data with its labels and train a model with it
file_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Raw_Data//Text_Classification//FinancialPhraseBank-v1.0"
file_name = "Sentences_50Agree.txt"
encoding = "ISO-8859-1"
save = False
save_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Data//Formatted_Data//Text_Classification"


text_data, text_labels = read_text_files_FinPhrase(file_path, file_name, encoding, save, save_path)
len(text_labels) == len(text_data)
# Next we are gonna downsize the dataset so that every class is equally represented
data = text_data
labels = text_labels

text_data_downsized, text_labels_downsized = down_sample_imbalanced_dataset(data, labels)
len(text_labels_downsized) == len(text_data_downsized)

# Next encode the labels into numbers and extract the stop words from the single sentences
replace_dict = {"negative":0,
                "neutral":1,
                "positive":2}

text_labels = pd.Series(text_labels_downsized).replace(replace_dict).values

# Remove all the stop words in every single sentence
stopwords = stopwords.words('english')

text_data_stops_removed = remove_stopwords(text_data_downsized, stopwords)


len(text_labels) == len(text_data_stops_removed)

features = text_data_stops_removed
labels = list(text_labels)
train_size = 0.70
seed = 256
set_seed = False

train_X, train_y, test_X, test_y = train_test_split(features, labels, train_size, set_seed, seed)

seed = 3475
set_seed = False
train_data = train_X
test_data = test_X
max_sequence_len = 100
train_labels = train_y
test_labels = test_y
dropout = 0.6
epochs = 20
save_logs = True
save_path_logs = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Logs"
save_model_summary = True

model, hist = nlp_model(seed, set_seed, train_data, test_data, max_sequence_len, train_labels, test_labels, dropout, epochs, save_logs, save_path_logs, save_model_summary)


model.summary()
# Classify new text
text = ["The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported"]
tokenizer_model = 'bert-base-uncased'
textstops_removed = remove_stopwords(text, stopwords)[0]
prob, label = classify_sentiment(model, textstops_removed, tokenizer_model)
print(f'Label: {label} | Probability: {prob:.4f}')

# Plot the metrics
hist = pd.read_csv("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//Logs//Mod_28-06-2023_15:10:26//Mod_hist_df_28-06-2023_15:10:26.csv", usecols=["loss", "accuracy", "val_loss", "val_accuracy"])

plot_metrics(hist)

