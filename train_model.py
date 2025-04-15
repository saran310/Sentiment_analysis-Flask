import pandas as pd
import re
import nltk
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

nltk.download("stopwords")

# Load dataset
df = pd.read_csv("C:/Users/saran/Downloads/finefoodembeddings10k.csv/finefoodembeddings10k.csv")


# Convert scores into sentiment labels (Positive = 1, Negative = 0)
df["Sentiment"] = df["Score"].apply(lambda x: 1 if x > 3 else 0 if x < 3 else None)
df.dropna(inplace=True)

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

df["Cleaned_Text"] = df["Text"].apply(clean_text)

# Tokenization
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df["Cleaned_Text"])
sequences = tokenizer.texts_to_sequences(df["Cleaned_Text"])
X_padded = pad_sequences(sequences, maxlen=max_len)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, df["Sentiment"], test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save model and tokenizer
model.save("models/sentiment_model.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
