import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Constants
MAX_LEN = 150
LABELS = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("toxic_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("💬 Toxic Comment Classifier")
st.markdown("Enter a comment below and this deep learning model will predict the type(s) of toxicity present.")

comment = st.text_area("Enter comment text here", height=150)

if st.button("Classify Comment"):
    if comment.strip() == "":
        st.warning("Please enter a valid comment.")
    else:
        seq = tokenizer.texts_to_sequences([comment])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        preds = model.predict(padded)[0]

        st.subheader("🧪 Prediction Results:")
        results = {label: float(pred) for label, pred in zip(LABELS, preds)}

        for label, score in results.items():
            st.write(f"**{label}:** {'✅' if score > 0.5 else '❌'} ({score:.2f})")
