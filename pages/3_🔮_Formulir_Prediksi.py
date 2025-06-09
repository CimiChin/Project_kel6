# pages/3_🔮_Formulir_Prediksi.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Formulir Prediksi", page_icon="🔮", layout="wide")

st.title("🔮 Halaman 3: Formulir Prediksi Permintaan")
st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi permintaan produk fashion.")

# Fungsi untuk memuat model yang sudah dilatih
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model di path '{path}' tidak ditemukan. Harap jalankan halaman 'Hasil Pelatihan Model' terlebih dahulu.")
        return None

# Muat model
knn_model = load_model('knn_model.joblib')
nb_model = load_model('nb_model.joblib')

# Opsi untuk input form (diasumsikan dari data)
weather_options = ['Clear', 'Cloudy', 'Rainy', 'Stormy']
promo_options = [0, 1] # 0 = Tidak Ada, 1 = Ada

if knn_model and nb_model:
    with st.form("prediction_form"):
        st.header("Masukkan Data untuk Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            inventory = st.number_input("Tingkat Inventaris (Stok Awal)", min_value=0, max_value=1000, value=100, step=10)
            weather = st.selectbox("Kondisi Cuaca", options=weather_options, index=0)
        
        with col2:
            promotion = st.selectbox("Ada Hari Libur/Promosi?", options=promo_options, format_func=lambda x: "Ya" if x == 1 else "Tidak", index=1)
        
        submit_button = st.form_submit_button(label="🚀 Lakukan Prediksi")

    if submit_button:
        # Buat dataframe dari input
        input_data = pd.DataFrame({
            'Inventory Level': [inventory],
            'Weather Condition': [weather],
            'Holiday/Promotion': [promotion]
        })
        
        st.subheader("Hasil Prediksi")
        
        # Lakukan prediksi dengan kedua model
        pred_knn = knn_model.predict(input_data)[0]
        prob_knn = knn_model.predict_proba(input_data)
        
        pred_nb = nb_model.predict(input_data)[0]
        prob_nb = nb_model.predict_proba(input_data)
        
        # Tampilkan hasil
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.info("Prediksi dari Model KNN")
            if pred_knn == "Tinggi":
                st.success(f"📈 Permintaan Diprediksi **{pred_knn}**")
            else:
                st.warning(f"📉 Permintaan Diprediksi **{pred_knn}**")
            
            st.write("Probabilitas:")
            st.dataframe(pd.DataFrame(prob_knn, columns=knn_model.classes_, index=['Prob.']))

        with col_res2:
            st.info("Prediksi dari Model Naive Bayes")
            if pred_nb == "Tinggi":
                st.success(f"📈 Permintaan Diprediksi **{pred_nb}**")
            else:
                st.warning(f"📉 Permintaan Diprediksi **{pred_nb}**")
                
            st.write("Probabilitas:")
            st.dataframe(pd.DataFrame(prob_nb, columns=nb_model.classes_, index=['Prob.']))
else:
    st.warning("Model tidak dapat dimuat. Silakan buka halaman **2_🤖_Hasil_Pelatihan_Model** untuk melatih dan menyimpan model terlebih dahulu.")
