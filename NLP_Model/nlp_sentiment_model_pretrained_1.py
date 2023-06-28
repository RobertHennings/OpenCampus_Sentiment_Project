# Here the NLP Model itself is specified
# Using a pretrained model setup since we are faced with unlabeled texts
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
import tensorflow_hub as hub
import glob
import datetime as dt
import pandas as pd
import numpy as np


import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

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
example_text = "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported"
# negative
sentiment, confidence = classify_sentiment(model, example_text)
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence}%")
