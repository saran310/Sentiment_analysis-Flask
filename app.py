from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/sentiment_model.h5")

# Load the tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define max_len (same as in training)
max_len = 100

# Store user inputs for visualization
user_inputs = {"positive": 0, "negative": 0}

# Function to predict sentiment
def predict_sentiment(review):
    review_seq = tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_seq, maxlen=max_len)
    prediction = model.predict(review_padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

@app.route("/")
def home():
    return render_template("index.html")  # Frontend form

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review = data["review"]
    sentiment = predict_sentiment(review)

    # Store results for visualization
    if sentiment == "Positive":
        user_inputs["positive"] += 1
    else:
        user_inputs["negative"] += 1

    return jsonify({"review": review, "sentiment": sentiment})

@app.route("/visualize")
def visualize():
    labels = list(user_inputs.keys())
    values = list(user_inputs.values())

    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=values, palette="coolwarm")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.title("Sentiment Analysis Results")
    
    plt.savefig("static/sentiment_plot.png")  # Save visualization
    return render_template("visualize.html", image="static/sentiment_plot.png")

if __name__ == "__main__":
    app.run(debug=True)
