import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# **1. Load Model Berdasarkan Dataset**
def load_model(dataset):
    model_files = {
        "X": "modelx.keras",
        "YT": "modelyt.keras",
        "TikTok": "modeltk.keras"
    }
    return tf.keras.models.load_model(model_files[dataset])

# **2. Load Tokenizer Berdasarkan Dataset**
def load_tokenizer(dataset):
    tokenizer_files = {
        "X": "tokenizerx.pkl",
        "YT": "tokenizeryt.pkl",
        "TikTok": "tokenizertk.pkl"
    }
    with open(tokenizer_files[dataset], "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# **3. Load Data Uji Berdasarkan Dataset**
def load_test_data(dataset):
    dataset_files = {
        "X": "x.csv",
        "YT": "yt.csv",
        "TikTok": "tk.csv"
    }
    
    df = pd.read_csv(dataset_files[dataset])
    texts = df["lower_text"].astype(str)  
    y = df["polarity"].values  

    _, x_test, _, y_test = train_test_split(texts, y, random_state=42)

    return x_test, y_test, df

# **4. Konversi Teks ke Sequences**
def text_to_sequences(tokenizer, texts, max_length=100):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# **5. Dapatkan Evaluasi Model**
def get_evaluation_metrics(dataset):
    model = load_model(dataset)
    tokenizer = load_tokenizer(dataset)
    texts, y_true, df = load_test_data(dataset)

    X_test = text_to_sequences(tokenizer, texts, max_length=100)
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int) 

    return y_true, y_pred, texts, df

# **6. Fungsi untuk Menampilkan Tabel Klasifikasi Report**
def show_classification_report(dataset):
    report_data = {
        "X": [
            ["Negative", 0.88, 0.88, 0.88, 290],
            ["Positive", 0.88, 0.88, 0.88, 280],
            ["Accuracy", "-", "-", 0.88, 570],
            ["Macro Avg", 0.88, 0.88, 0.88, 570],
            ["Weighted Avg", 0.88, 0.88, 0.88, 570]
        ],
        "YT": [
            ["Negative", 0.86, 0.84, 0.85, 332],
            ["Positive", 0.84, 0.86, 0.85, 322],
            ["Accuracy", "-", "-", 0.85, 654],
            ["Macro Avg", 0.85, 0.85, 0.85, 654],
            ["Weighted Avg", 0.85, 0.85, 0.85, 654]
        ],
        "TikTok": [
            ["Negative", 0.87, 0.91, 0.89, 395],
            ["Positive", 0.75, 0.67, 0.71, 159],
            ["Accuracy", "-", "-", 0.84, 554],
            ["Macro Avg", 0.81, 0.79, 0.80, 554],
            ["Weighted Avg", 0.84, 0.84, 0.84, 554]
        ]
    }

    df_report = pd.DataFrame(
        report_data[dataset],
        columns=["Class", "Precision", "Recall", "F1-Score", "Support"]
    )
    df_report[["Precision", "Recall", "F1-Score"]] = df_report[["Precision", "Recall", "F1-Score"]].applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    st.subheader(f"📋 Klasifikasi Report - {dataset}")
    st.table(df_report)

# **7. Fungsi untuk Menampilkan Confusion Matrix**
def show_confusion_matrix(dataset):
    image_files = {
        "X": "xmatrix.png",
        "YT": "ytmatrix.png",
        "TikTok": "tkmatrix.png"
    }

    st.subheader(f"📊 Confusion Matrix - {dataset}")
    st.image(image_files[dataset], use_container_width=True)  # Menggunakan use_container_width

# **8. Fungsi untuk Membuat Word Cloud**
def show_wordcloud(texts, title):
    text = " ".join(texts)
    wordcloud = WordCloud(
        width=800, height=400, 
        background_color="white", 
        colormap="coolwarm", 
        max_words=200, 
        contour_color="steelblue", 
        contour_width=2
    ).generate(text)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.subheader(title)
    st.pyplot(fig)

# **9. Fungsi untuk Menampilkan Dataset**
def show_full_dataset(df, dataset_name):
    st.subheader(f"📜 Dataset {dataset_name}")

    df["Sentimen"] = df["polarity"].map({1: "Positif", 0: "Negatif"})

    st.dataframe(df[["lower_text", "Sentimen"]], height=300)

# **10. Streamlit UI**
st.title("📊 Analisis Sentimen Visi Indonesia Emas 2045")

# Pilihan dataset (bisa memilih lebih dari satu)
selected_datasets = st.multiselect("📂 Pilih Dataset:", ["X", "YT", "TikTok"], default=["X", "YT", "TikTok"])

# Tombol lihat hasil analisis
if st.button("Lihat Hasil"):
    st.subheader("📊 Hasil Analisis Sentimen")

    columns = st.columns(len(selected_datasets))  # Buat kolom sejajar berdasarkan jumlah dataset yang dipilih
    
    for i, dataset in enumerate(selected_datasets):
        with columns[i]:  
            st.markdown(f"## 📂 {dataset}")

            # **1️⃣ Tampilkan Dataset**
            y_true, y_pred, texts, df = get_evaluation_metrics(dataset)
            show_full_dataset(df, dataset)

            # **2️⃣ Tampilkan Tabel Klasifikasi Report**
            show_classification_report(dataset)

            # **3️⃣ Tampilkan Confusion Matrix**
            show_confusion_matrix(dataset)

    # **4️⃣ Word Cloud Ditampilkan Sejajar**
    st.subheader("☁️ Word Cloud")
    wc_columns = st.columns(len(selected_datasets))

    for i, dataset in enumerate(selected_datasets):
        with wc_columns[i]:
            y_true, _, texts, _ = get_evaluation_metrics(dataset)
            
            st.markdown(f"### {dataset} - Semua Opini")
            show_wordcloud(texts, f"🌍 {dataset} - Semua Opini")

            positive_texts = [text for text, label in zip(texts, y_true) if label == 1]
            negative_texts = [text for text, label in zip(texts, y_true) if label == 0]

            if positive_texts:
                show_wordcloud(positive_texts, f"🌟 {dataset} - Positif")
            if negative_texts:
                show_wordcloud(negative_texts, f"⚠️ {dataset} - Negatif")

st.write("🚀 Pilih dataset dan klik 'Lihat Hasil' untuk melihat analisis sentimen.")
