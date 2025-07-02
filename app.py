# app.py
import streamlit as st
import joblib
import re

# -------------------------
# 1. Load model & komponen
# -------------------------
model = joblib.load("logistic_regression_model.pkl")  # Ganti dengan nb_model jika pakai Naive Bayes
tfidf = joblib.load("tfidf_vectorizer (1).pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------------
# 2. Utility Function
# -------------------------

def clean_text(text):
    """Membersihkan teks dari simbol dan huruf besar"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def validasi_input(teks, minimal_kata=3):
    """Memastikan teks cukup panjang"""
    return len(teks.strip().split()) >= minimal_kata

def prediksi_status(teks):
    """Melakukan prediksi sentimen dari status"""
    if not validasi_input(teks):
        return "⚠️ Status terlalu pendek untuk diprediksi. Masukkan minimal 3 kata."

    teks_bersih = clean_text(teks)
    vektor = tfidf.transform([teks_bersih])
    pred = model.predict(vektor)
    label = label_encoder.inverse_transform(pred)[0]

    return label

# -------------------------
# 3. Tampilan Streamlit
# -------------------------

st.set_page_config(page_title="Analisis Sentimen Facebook", layout="centered")
st.title("💬 Analisis Sentimen Status Facebook")
st.caption("oleh Siti Nurhayati")

st.write("Masukkan status Facebook kamu yang berkaitan dengan kesehatan mental, lalu tekan tombol prediksi untuk melihat hasilnya.")

status_input = st.text_area("📝 Tulis status Facebook kamu di sini")

if st.button("🔍 Prediksi Sentimen"):
    if status_input.strip() == "":
        st.warning("Silakan isi status terlebih dahulu.")
    else:
        hasil = prediksi_status(status_input)

        if "⚠️" in hasil:
            st.warning(hasil)
        else:
            warna = {
                "POSITIF": "green",
                "NETRAL": "orange",
                "NEGATIF": "red"
            }
            st.markdown(f"<h4 style='color:{warna.get(hasil.upper(), 'black')}'>🔮 Sentimen: {hasil}</h4>", unsafe_allow_html=True)
