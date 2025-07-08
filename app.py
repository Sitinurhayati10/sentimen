
import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import joblib
import random
import re
import io
import plotly.express as px
import time
import hashlib

st.set_page_config("Mental Health Sentiment App", page_icon="💬", layout="centered")

# -----------------------------
# Modern Font + Responsif Layout
# -----------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            font-size: 16px;
        }

        .main {
            padding-left: 1rem;
            padding-right: 1rem;
            transition: all 0.3s ease;
        }

        .sentiment-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f7f7f7;
            margin-top: 10px;
            margin-bottom: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }

        @media screen and (max-width: 600px) {
            .stButton > button {
                width: 100%;
            }
            .stTextInput > div, .stTextArea > div {
                width: 100% !important;
            }
        }

        @media (prefers-color-scheme: dark) {
            .sentiment-box {
                background-color: #1e1e1e;
                color: #f5f5f5;
            }
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Tambahan UI dari wireframe
# -----------------------------
def tampilkan_footer():
    st.markdown("""<hr style='margin-top:3rem;'>
    <center><small>Footer: ©2025 Sentimen Kesehatan Mental App</small></center>
    """, unsafe_allow_html=True)

def tombol_navigasi_hasil():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Analisis Ulang"):
            st.session_state.page = "input"
            st.rerun()
    with col2:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()

# -----------------------------
# Load model, TF-IDF, encoder
# -----------------------------
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer (1).pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Fungsi bantu (sama seperti sebelumnya)
# -----------------------------
# Untuk ringkas, diasumsikan isi fungsi bantu tidak berubah dari versi kamu sebelumnya

# -----------------------------
# Halaman-halaman (login, home, input, hasil, journal)
# -----------------------------
# Diasumsikan tidak berubah kecuali bagian judul & footer/navigasi

# Contoh perubahan:
# 1. Di halaman input:
# st.markdown("<h2>MASUKKAN STATUS ANDA</h2>", unsafe_allow_html=True)
# tampilkan_footer()

# 2. Di halaman hasil:
# st.markdown("<h2>HASIL ANALISIS SENTIMEN ANDA</h2>", unsafe_allow_html=True)
# tombol_navigasi_hasil()
# tampilkan_footer()

# 3. Di halaman lain (home, journal):
# tampilkan_footer()

# Silakan salin fungsi bantu + logika halaman kamu ke dalam struktur ini.
