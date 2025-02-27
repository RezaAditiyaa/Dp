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

# **6. Fungsi untuk Membuat Word Cloud**
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

# **7. Fungsi untuk Menampilkan Dataset**
def show_full_dataset(df, dataset_name):
    st.subheader(f"📜 Dataset {dataset_name}")

    df["Sentimen"] = df["polarity"].map({1: "Positif", 0: "Negatif"})

    st.dataframe(df[["lower_text", "Sentimen"]], height=300)

# **8. Streamlit UI**
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

            # **2️⃣ Statistik Data**
            st.subheader("📊 Statistik Data")
            st.write(f"🔹 **Total Data Uji**: {len(y_true)}")
            st.write(f"🔹 **Total Negatif**: {sum(y_true == 0)}")
            st.write(f"🔹 **Total Positif**: {sum(y_true == 1)}")

            # **3️⃣ Distribusi Sentimen (Pie Chart)**
            st.subheader("📊 Distribusi Sentimen")
            sentiment_counts = pd.Series(y_true).value_counts().sort_index()
            labels = ["Negatif", "Positif"]
            colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(labels))) 

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(
                sentiment_counts,
                labels=labels,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                shadow=True,
                explode=(0.05, 0.05),
                wedgeprops={'edgecolor': 'black'}
            )
            ax.axis("equal")
            st.pyplot(fig)

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
