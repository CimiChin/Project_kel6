# app.py

import streamlit as st

st.set_page_config(
    page_title="Prediksi Permintaan Fashion",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Dashboard Prediksi Permintaan Produk Fashion")

st.sidebar.success("Pilih halaman di atas untuk memulai.")

st.markdown(
    """
    Selamat datang di Dashboard Prediksi Permintaan Produk Fashion.
    
    Dashboard ini terdiri dari tiga halaman utama yang dapat Anda akses melalui menu di sebelah kiri:
    
    ### 1. 📊 EDA dan Karakteristik
    Halaman ini menampilkan analisis data eksplorasi (EDA) dari dataset inventaris ritel. 
    Anda dapat melihat dataset mentah, statistik deskriptif, dan berbagai visualisasi untuk memahami pola data.
    
    ### 2. 🤖 Hasil Pelatihan Model
    Halaman ini menunjukkan hasil dari proses pelatihan model machine learning. 
    Kami menggunakan data historis untuk melatih model klasifikasi yang membedakan antara permintaan 'Tinggi' dan 'Rendah'.
    
    ### 3. 🔮 Formulir Prediksi
    Gunakan formulir interaktif di halaman ini untuk memasukkan data baru dan mendapatkan prediksi permintaan 
    secara real-time menggunakan model **K-Nearest Neighbors (KNN)** dan **Naive Bayes**.
    
    ---
    
    **Dataset**: [Retail Store Inventory Forecasting Dataset on Kaggle](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
    
    **Teknologi yang Digunakan**:
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Plotly
    """
)
