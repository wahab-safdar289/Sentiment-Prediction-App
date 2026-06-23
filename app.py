import streamlit as st
import joblib
import numpy as np


with open(r"E:\\ML_Projects\Sentiment_Prediction_App\\idf.pkl", "rb") as f:
    tfidf = joblib.load(f)

with open(r"E:\\ML_Projects\Sentiment_Prediction_App\\model.pkl", "rb") as f:
    model = joblib.load(f)

with open(r"E:\\ML_Projects\Sentiment_Prediction_App\\encoder.pkl", "rb") as f:
    encoder = joblib.load(f)


st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Write a movie review and get instant sentiment prediction</div>', unsafe_allow_html=True)


review = st.text_area("Enter your movie review here:")


if st.button("Predict Sentiment "):

    if review.strip() == "":
        st.warning("Please write a review first!")
    else:
        
        review_tfidf = tfidf.transform([review])

        prediction = model.predict(review_tfidf)

        sentiment = encoder.inverse_transform(prediction)[0]

        
        if "pos" in sentiment.lower() or sentiment == 1:
            st.success(f"Sentiment: {sentiment}")
        elif "neg" in sentiment.lower() or sentiment == 0:
            st.error(f"Sentiment: {sentiment}")
        else:
            st.info(f"Sentiment: {sentiment}")