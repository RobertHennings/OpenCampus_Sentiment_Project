import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def build_sentiment_model():
    # Load pre-trained FinBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    return model, tokenizer

def classify_sentiment(model, tokenizer, text):
    # Tokenize the text
    encoded_input = tokenizer.encode_plus(
        text,
        truncation=True,
        padding=True,
        return_tensors='tf'
    )

    # Make prediction
    logits = model(encoded_input['input_ids'])[0]
    probabilities = tf.nn.softmax(logits)[0]
    sentiment_label = tf.argmax(probabilities).numpy()

    # Process prediction
    sentiment = "Positive" if sentiment_label == 1 else "Negative"
    confidence = np.round(probabilities[sentiment_label] * 100, 2)

    return sentiment, confidence

# Example usage:
model, tokenizer = build_sentiment_model()  # Instantiate the model and tokenizer

example_text = "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility; contrary to earlier layoffs the company contracted the ranks of its office workers, the daily Postimees reported."
# Negative
example_text = "Net income from life insurance doubled to EUR 6.8 mn from EUR 3.2 mn , and net income from non-life insurance rose to EUR 5.2 mn from EUR 1.5 mn in the corresponding period in 2009 .@positive Net sales increased to EUR193 .3 m from EUR179 .9 m and pretax profit rose by 34.2 % to EUR43 .1 m. ( EUR1 = USD1 .4 )"
# Positive
sentiment, confidence = classify_sentiment(model, tokenizer, example_text)
print(f"Sentiment: {sentiment} with Confidence: {confidence}%")



