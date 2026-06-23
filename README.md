# Movie Review Sentiment Analysis App

This project is a machine learning-based web application that predicts whether a movie review is positive or negative using natural language processing techniques.

The model uses TF-IDF vectorization to convert text into numerical form and a trained classification model to perform sentiment prediction. The application is built using Streamlit, which provides a simple and interactive user interface.

Project Structure

Sentiment_Prediction_App/

app.py
model.pkl
tfidf_vec.pkl
encoder.pkl
nlp.pkl
requirements.txt
README.txt

Installation & Setup

Clone the repository

git clone https:https://github.com/wahab-safdar289/Sentiment-Prediction-App

Create virtual environment (optional but recommended)

python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py

Requirements.txt

streamlit
numpy
pandas
scikit-learn
joblib

How it works

User enters a movie review
Text is converted using TF-IDF vectorizer
Trained ML model predicts sentiment
Label encoder converts output into readable label
Result is shown on Streamlit interface

Output

Positive review → Positive sentiment
Negative review → Negative sentiment

Notes

Keep all .pkl files in the same folder as app.py
Make sure TF-IDF vectorizer and model are trained on the same pipeline

End of project
