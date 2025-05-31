import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


word_index= imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}
model=load_model('imdb_rnn_model.h5')

# step2: Helper function to decode reviews
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text]) 

# function to preprocess user input
def preprocess_input(user_input):
    words = user_input.lower().split()
    # Convert the input text to a sequence of integers
    sequence = [word_index.get(word, 2) + 3 for word in words]
    # Pad the sequence to ensure it has the same length as the training data
    padded_sequence = pad_sequences([sequence], maxlen=500,padding='pre')
    return np.array(padded_sequence)

def predict_sentiment(user_input):
    # Preprocess the user input
    padded_sequence = preprocess_input(user_input)
    
    # Predict sentiment
    prediction = model.predict(padded_sequence)
    
    # Interpret the prediction
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]


# streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative):")
user_input = st.text_area("Review Text", "I love this movie!")

if st.button("Predict Sentiment"):
    preprocessed_input= preprocess_input(user_input)
    prediction =model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
    st.write("Click the button to predict sentiment.")



