# pages/2_🤖_Hasil_Pelatihan_Model.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

st.set_page_config(page_title="Hasil Model", page_icon="🤖", layout="wide")

st.title("🤖 Halaman 2: Hasil Pelatihan Model")
st.markdown("Halaman ini menjelaskan proses dan menampilkan hasil dari pelatihan model.")

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_and_prep_data(url):
    df = pd.read_csv(url)
    df = df[df['Category'] == 'Clothing'].copy()
    df.dropna(inplace=True)

    # Membuat target variable (klasifikasi)
    # Jika 'Units Sold' > median -> 'Tinggi', else -> 'Rendah'
    median_sold = df['Units Sold'].median()
    df['Demand'] = df['Units Sold'].apply(lambda x: 'Tinggi' if x > median_sold else 'Rendah')
    
    # Memilih fitur untuk model
    features = ['Inventory Level', 'Weather Condition', 'Holiday/Promotion']
    target = 'Demand'
    
    X = df[features]
    y = df[target]
    
    return X, y, median_sold

# Fungsi untuk melatih model dan menyimpan hasilnya
@st.cache_resource
def train_models(X, y):
    # Memisahkan fitur numerik dan kategorikal
    numeric_features = ['Inventory Level']
    categorical_features = ['Weather Condition', 'Holiday/Promotion']

    # Membuat preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Membuat pipeline untuk KNN
    knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', KNeighborsClassifier(n_neighbors=5))])

    # Membuat pipeline untuk Naive Bayes
    nb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', GaussianNB())])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Latih model
    knn_pipeline.fit(X_train, y_train)
    nb_pipeline.fit(X_train, y_train)
    
    # Simpan model yang sudah dilatih
    joblib.dump(knn_pipeline, 'knn_model.joblib')
    joblib.dump(nb_pipeline, 'nb_model.joblib')

    # Prediksi
    y_pred_knn = knn_pipeline.predict(X_test)
    y_pred_nb = nb_pipeline.predict(X_test)

    # Evaluasi
    acc_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    
    acc_nb = accuracy_score(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
    
    return acc_knn, report_knn, acc_nb, report_nb

try:
    X, y, median_val = load_and_prep_data('retail_inventory.csv')

    st.header("1. Persiapan Data")
    st.markdown(f"""
    - **Fitur (Input)**: `Inventory Level`, `Weather Condition`, `Holiday/Promotion`.
    - **Target (Output)**: `Demand` (Permintaan).
    - Variabel target `Demand` dibuat secara biner:
        - **Tinggi**: Jika `Units Sold` > {int(median_val)}
        - **Rendah**: Jika `Units Sold` <= {int(median_val)}
    - Data yang kosong (`NaN`) telah dihapus.
    """)

    st.header("2. Pelatihan dan Evaluasi Model")
    st.info("Model dilatih menggunakan data historis. Proses ini di-cache agar tidak berjalan berulang kali.")

    acc_knn, report_knn, acc_nb, report_nb = train_models(X, y)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.metric("Akurasi", f"{acc_knn:.2%}")
        st.text("Laporan Klasifikasi:")
        st.dataframe(pd.DataFrame(report_knn).transpose())
    
    with col2:
        st.subheader("Naive Bayes")
        st.metric("Akurasi", f"{acc_nb:.2%}")
        st.text("Laporan Klasifikasi:")
        st.dataframe(pd.DataFrame(report_nb).transpose())

    st.success("Model KNN dan Naive Bayes telah berhasil dilatih dan disimpan sebagai `knn_model.joblib` dan `nb_model.joblib` untuk digunakan di halaman prediksi.")

except FileNotFoundError:
    st.error("File `retail_inventory.csv` tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
