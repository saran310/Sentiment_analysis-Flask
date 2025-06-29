# Sentiment Analysis Flask App

This project is a simple web application built with **Flask** and **TensorFlow/Keras** that performs **sentiment analysis** on text input (e.g., product reviews).

---

## 🚀 Features

- Predicts sentiment (Positive/Negative) of user input
- Trained using a deep learning model
- Flask-powered web interface
- Lightweight and easy to run locally

---

## 🧠 Model Info

- Pre-trained using a custom dataset
- Embeddings used: `finefoodembeddings10k.csv` (local file, not included in GitHub repo due to size)

---

## 🛠️ Installation

1. **Clone the repo:**

```bash
git clone https://github.com/saran310/Sentiment_analysis-Flask.git
cd Sentiment_analysis-Flask

python -m venv venv
venv\Scripts\activate   # On Windows
# Or
source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt

python app.py
Visit http://127.0.0.1:5000 in your browser.
