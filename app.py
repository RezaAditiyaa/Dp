import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# **1. Load Model Berdasarkan Dataset**
def load_model(dataset):
    model_files = {
        "X": "Model/modelx.h5",
        "YT": "Model/modelyt.h5",
        "TikTok": "Model/modeltk.h5"
    }
    return tf.keras.models.load_model(model_files[dataset])

# **2. Load Tokenizer Berdasarkan Dataset**
def load_tokenizer(dataset):
    tokenizer_files = {
        "X": "Tokenizer/tokenizerx.pkl",
        "YT": "Tokenizer/tokenizeryt.pkl",
        "TikTok": "Tokenizer/tokenizertk.pkl"
    }
    
    with open(tokenizer_files[dataset], "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer

# **3. Load Data Uji Berdasarkan Dataset**
def load_test_data(dataset, test_size=0.20):
    dataset_files = {
        "X": "Dataset/x.csv",
        "YT": "Dataset/yt.csv",
        "TikTok": "Dataset/tk.csv"
    }
    
    df = pd.read_csv(dataset_files[dataset])

    texts = df["stemmed_text"]
    y = df["polarity"].values  

    # Split data menjadi train dan test
    x_train, x_test, y_train, y_test = train_test_split(texts, y, test_size=test_size, random_state=42)

    return x_test, y_test

# **4. Konversi Teks ke Sequences**
def text_to_sequences(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

# **5. Dapatkan Evaluasi Model**
def get_evaluation_metrics(dataset):
    model = load_model(dataset)
    tokenizer = load_tokenizer(dataset)
    texts, y_true = load_test_data(dataset)

    # Ubah teks ke sequences
    X_test = text_to_sequences(tokenizer, texts)

    # Prediksi dengan model
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)  

    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Hitung classification report
    class_report = classification_report(y_true, y_pred, target_names=["Negatif", "Positif"], output_dict=True)

    return cm, class_report, y_true, y_pred

# **6. Streamlit UI**
st.title("📊 Analisis Sentimen Visi Indonesia Emas 2045")

# Pilihan dataset
option = st.selectbox("📂 Pilih Dataset:", ["X", "YT", "TikTok"])

# Tombol lihat hasil analisis
if st.button("Lihat Hasil"):
    cm, class_report, y_true, y_pred = get_evaluation_metrics(option)

    # **Tampilkan Confusion Matrix dalam Tabel**
    st.subheader("📊 Confusion Matrix")
    st.write(pd.DataFrame(cm, columns=["Pred Negatif", "Pred Positif"], index=["Aktual Negatif", "Aktual Positif"]))

    # **Confusion Matrix Heatmap**
    st.subheader("📊 Confusion Matrix (Heatmap)")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"], ax=ax)
    st.pyplot(fig)

    # **Distribusi Sentimen dalam Pie Chart**
    st.subheader("📊 Distribusi Sentimen")
    sentiment_counts = pd.Series(y_true).value_counts().sort_index()
    labels = ["Negatif", "Positif"]
    colors = ["red", "blue"]

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    ax.axis("equal")
    st.pyplot(fig)

    # **Menampilkan Akurasi, Presisi, Recall, F1-score**
    st.subheader("📊 Evaluasi Model")
    accuracy = class_report["accuracy"]
    precision_neg = class_report["Negatif"]["precision"]
    recall_neg = class_report["Negatif"]["recall"]
    f1_neg = class_report["Negatif"]["f1-score"]
    precision_pos = class_report["Positif"]["precision"]
    recall_pos = class_report["Positif"]["recall"]
    f1_pos = class_report["Positif"]["f1-score"]

    st.write(f"📌 **Akurasi**: {accuracy:.4f}")
    st.write(f"📌 **Presisi (Negatif)**: {precision_neg:.4f} | **Presisi (Positif)**: {precision_pos:.4f}")
    st.write(f"📌 **Recall (Negatif)**: {recall_neg:.4f} | **Recall (Positif)**: {recall_pos:.4f}")
    st.write(f"📌 **F1-score (Negatif)**: {f1_neg:.4f} | **F1-score (Positif)**: {f1_pos:.4f}")

    # Menampilkan jumlah total data
    st.subheader("📊 Statistik Data")
    st.write(f"🔹 **Total Data Uji**: {len(y_true)}")
    st.write(f"🔹 **Total Negatif**: {sum(y_true == 0)}")
    st.write(f"🔹 **Total Positif**: {sum(y_true == 1)}")

st.write("🚀 Pilih dataset dan klik 'Lihat Hasil' untuk melihat analisis sentimen.")

