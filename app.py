import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load LSTM model and tokenizer once
@st.cache_resource
def load_resources():
    model = load_model('next_word_lstm.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_resources()

# Predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len - 1):]  # trim to fit expected input length
    padded_sequence = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted_probs = model.predict(padded_sequence, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=1)[0]

    # Reverse lookup: index -> word
    word_lookup = {index: word for word, index in tokenizer.word_index.items()}
    return word_lookup.get(predicted_index, "(unknown)")

# Streamlit UI
st.title("📖 Next Word Predictor (LSTM + Early Stopping)")

input_text = st.text_input("Enter a sequence of words:", "Last night of all, When yond same Starre that's")

if st.button("🔮 Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # model expects maxlen-1 input
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.success(f"**Next word:** `{next_word}`")
