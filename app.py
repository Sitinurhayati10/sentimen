
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
# Font dan Styling
# -----------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            font-size: 16px;
        }
        .sentiment-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f7f7f7;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# UI Tambahan
# -----------------------------
def tampilkan_footer():
    st.markdown("""<hr><center><small>Footer: ©2025 Sentimen Kesehatan Mental App</small></center>""", unsafe_allow_html=True)

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
# Load model
# -----------------------------
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer (1).pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Fungsi Database dan Utilitas
# -----------------------------
def init_db():
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, nama_lengkap TEXT, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS status (
        id_status INTEGER PRIMARY KEY AUTOINCREMENT,
        isi_status TEXT, label_sentimen TEXT,
        kepercayaan REAL, tanggal_status DATE, id_user TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()
def simpan_user(username, nama_lengkap, password):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?)", (username, nama_lengkap, hash_password(password)))
    conn.commit(); conn.close()

def validasi_login(username, password):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor(); c.execute("SELECT nama_lengkap, password FROM users WHERE username = ?", (username,))
    row = c.fetchone(); conn.close()
    if not row: return "TIDAK_TERDAFTAR"
    elif row[1] != hash_password(password): return "PASSWORD_SALAH"
    else: return row[0]

def simpan_status(id_user, isi_status, label_sentimen, confidence):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute("INSERT INTO status (id_user, isi_status, label_sentimen, kepercayaan, tanggal_status) VALUES (?, ?, ?, ?, ?)",
              (id_user, isi_status, label_sentimen, confidence, datetime.today().date()))
    conn.commit(); conn.close()

def ambil_riwayat(id_user):
    conn = sqlite3.connect('sentimen.db')
    df = pd.read_sql_query("SELECT * FROM status WHERE id_user = ? ORDER BY id_status DESC", conn, params=(id_user,))
    conn.close(); return df

def hapus_semua_riwayat(id_user):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor(); c.execute("DELETE FROM status WHERE id_user = ?", (id_user,))
    conn.commit(); conn.close()

def bersihkan_teks(teks):
    teks = teks.lower(); teks = re.sub(r'[^a-zA-Z\s]', '', teks); return teks.strip()

def validasi_input(teks, minimal_kata=3): return len(teks.strip().split()) >= minimal_kata

def prediksi_sentimen(teks):
    if not validasi_input(teks): return "Terlalu pendek", 0
    teks_bersih = bersihkan_teks(teks)
    fitur = tfidf.transform([teks_bersih])
    pred = model.predict(fitur); prob = model.predict_proba(fitur).max() * 100
    label = label_encoder.inverse_transform(pred)[0]
    return label.upper(), round(prob, 2)

# -----------------------------
# Halaman Login dan Navigasi
# -----------------------------
init_db()
if "page" not in st.session_state: st.session_state.page = "login"

if st.session_state.page != "login":
    with st.sidebar:
        st.title("🧭 Navigasi")
        st.button("🏠 Home", on_click=lambda: st.session_state.update(page="home"))
        st.button("✍️ Input Status", on_click=lambda: st.session_state.update(page="input"))
        st.button("📊 Hasil", on_click=lambda: st.session_state.update(page="hasil"))
        st.button("🚪 Logout", on_click=lambda: st.session_state.clear())

# -----------------------------
# Login Page
# -----------------------------
if st.session_state.page == "login":
    st.title("🔐 Login")
    tab1, tab2 = st.tabs(["Login", "Daftar"])
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Masuk"):
            hasil_login = validasi_login(username, password)
            if hasil_login == "TIDAK_TERDAFTAR": st.warning("Username belum terdaftar.")
            elif hasil_login == "PASSWORD_SALAH": st.error("Password salah.")
            else:
                st.session_state.username = username
                st.session_state.nama = hasil_login
                st.session_state.page = "home"; st.rerun()
    with tab2:
        nama = st.text_input("Nama Lengkap")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Daftar"):
            if nama and user and pw:
                simpan_user(user, nama, pw)
                st.success("Akun berhasil dibuat. Silakan login.")
            else:
                st.warning("Semua kolom wajib diisi.")

# -----------------------------
# Halaman Home
# -----------------------------
elif st.session_state.page == "home":
    st.title(f"Halo, {st.session_state.nama} 👋")
    st.write("Silakan pilih menu di sidebar.")
    tampilkan_footer()

# -----------------------------
# Halaman Input
# -----------------------------
elif st.session_state.page == "input":
    st.markdown("<h2>MASUKKAN STATUS ANDA</h2>", unsafe_allow_html=True)
    status = st.text_area("Apa yang sedang Anda pikirkan hari ini?", height=150)
    if st.button("🔍 Analisis Sekarang"):
        if status.strip():
            label, conf = prediksi_sentimen(status)
            if label == "Terlalu pendek":
                st.warning("⚠️ Status minimal 3 kata.")
            else:
                simpan_status(st.session_state.username, status, label, conf)
                st.session_state.hasil_status = status
                st.session_state.hasil_label = label
                st.session_state.hasil_conf = conf
                st.session_state.page = "hasil"; st.rerun()
        else:
            st.warning("Status tidak boleh kosong.")
    tampilkan_footer()

# -----------------------------
# Halaman Hasil
# -----------------------------
elif st.session_state.page == "hasil":
    st.markdown("<h2>HASIL ANALISIS SENTIMEN ANDA</h2>", unsafe_allow_html=True)
    if "hasil_status" in st.session_state:
        st.markdown(f'<div class="sentiment-box">💬 <b>"{st.session_state.hasil_status}"</b><br>📌 Sentimen: <b>{st.session_state.hasil_label}</b><br>📈 Kepercayaan: <b>{st.session_state.hasil_conf}%</b></div>', unsafe_allow_html=True)
    df = ambil_riwayat(st.session_state.username)
    if not df.empty:
        df['tanggal_status'] = pd.to_datetime(df['tanggal_status'])
        df_plot = df.groupby(['tanggal_status', 'label_sentimen']).size().reset_index(name='jumlah')
        fig = px.bar(df_plot, x='tanggal_status', y='jumlah', color='label_sentimen',
                     color_discrete_map={'POSITIF': 'blue', 'NETRAL': 'gray', 'NEGATIF': 'red'},
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada riwayat.")
    tombol_navigasi_hasil()
    tampilkan_footer()
