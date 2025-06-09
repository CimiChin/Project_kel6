# pages/1_📊_EDA_dan_Karakteristik.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Halaman 1: Exploratory Data Analysis (EDA)")
st.markdown("Halaman ini menampilkan analisis, statistik, dan visualisasi dari dataset.")

# Fungsi untuk memuat data dengan caching
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    # Hanya fokus pada kategori Fashion/Clothing
    df = df[df['Category'] == 'Clothing'].copy()
    return df

try:
    df = load_data('retail_inventory.csv')

    # Tampilkan Dataset
    st.header("Tampilan Awal Dataset (Kategori: Pakaian)")
    st.dataframe(df.head())

    # Karakteristik Data
    st.header("Karakteristik dan Statistik Data")
    st.write(df.describe())

    st.header("Distribusi Data")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Jumlah Baris Data", value=f"{df.shape[0]:,}")
    with col2:
        st.metric(label="Jumlah Kolom Data", value=df.shape[1])
        
    # Visualisasi Data
    st.header("Visualisasi Data")
    
    # Tren Penjualan Harian
    st.subheader("Tren Penjualan Harian (Units Sold)")
    time_series_fig = px.line(df, x='Date', y='Units Sold', title='Penjualan Harian Seiring Waktu', 
                              labels={'Date': 'Tanggal', 'Units Sold': 'Unit Terjual'})
    st.plotly_chart(time_series_fig, use_container_width=True)

    # Visualisasi lainnya
    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        # Penjualan berdasarkan Kondisi Cuaca
        st.subheader("Penjualan Berdasarkan Kondisi Cuaca")
        weather_fig = px.box(df, x='Weather Condition', y='Units Sold', title='Distribusi Penjualan vs Cuaca',
                              labels={'Weather Condition': 'Kondisi Cuaca', 'Units Sold': 'Unit Terjual'})
        st.plotly_chart(weather_fig, use_container_width=True)
        
    with col_vis2:
        # Pengaruh Hari Libur/Promosi
        st.subheader("Penjualan Selama Promosi/Hari Libur")
        promo_fig = px.pie(df, names='Holiday/Promotion', title='Proporsi Penjualan pada Hari Promosi/Libur', 
                           hole=0.3, labels={'Holiday/Promotion': 'Promosi/Libur'})
        st.plotly_chart(promo_fig, use_container_width=True)

except FileNotFoundError:
    st.error("File `retail_inventory.csv` tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan `app.py`.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
