import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
import matplotlib.pyplot as plt

# Memuat model dari file
model = joblib.load('crop_growth_model.pkl')

# Konfigurasi MongoDB
client = MongoClient('mongodb+srv://irfnriza:monggoCorn123@cornenvironment.mwtc0sc.mongodb.net/?retryWrites=true&w=majority&appName=CornEnvironment')
db = client['environment']
collection = db['sensor_data']

# Fungsi untuk mengambil data dari MongoDB dan menghitung prediksi
@st.cache_data(ttl=60)
def get_predictions():
    # Mengambil data dari MongoDB
    data = pd.DataFrame(list(collection.find()))

    # Menentukan kolom yang diperlukan
    temperature_column = 'temperature'
    humidity_column = 'humidity'
    soil_moisture_column = 'soil_moisture'

    # Menghitung rata-rata data
    mean_data = data[[temperature_column, humidity_column, soil_moisture_column]].mean().values.reshape(1, -1)

    # Mengambil data terbaru
    latest_data = data[[temperature_column, humidity_column, soil_moisture_column]].iloc[-1].values.reshape(1, -1)

    # Membuat prediksi berdasarkan rata-rata data
    mean_pred = model.predict(mean_data)

    # Membuat prediksi berdasarkan data terbaru
    latest_pred = model.predict(latest_data)
    
    return mean_pred, latest_pred, data

# Fungsi untuk memperbarui data dan prediksi
def update_predictions():
    mean_pred, latest_pred, data = get_predictions()

    # Menampilkan rata-rata data dan prediksi berdasarkan rata-rata data
    st.subheader('Rata-rata Data dan Prediksi')
    col1, col2 = st.columns(2)
    with col1:
        st.write('#### Rata-rata Data')
        st.write(data[['temperature', 'humidity', 'soil_moisture']].mean())
    with col2:
        st.write('#### Prediksi Berdasarkan Rata-rata Data')
        st.write(f'Prediksi Pertumbuhan Jagung: {mean_pred[0]:.2f}')

    # Menampilkan data terbaru dan prediksi berdasarkan data terbaru
    st.subheader('Data Terbaru dan Prediksi')
    col3, col4 = st.columns(2)
    with col3:
        st.write('#### Data Terbaru')
        st.write(data[['temperature', 'humidity', 'soil_moisture']].iloc[-1])
    with col4:
        st.write('#### Prediksi Berdasarkan Data Terbaru')
        st.write(f'Prediksi Pertumbuhan Jagung: {latest_pred[0]:.2f}')

    # Grafik garis yang menampilkan histori suhu, kelembapan, dan kelembapan tanah dalam satu grafik
    st.subheader('Grafik Histori')
    fig, ax = plt.subplots()
    ax.plot(data['temperature'], label='Suhu')
    ax.plot(data['humidity'], label='Kelembapan')
    ax.plot(data['soil_moisture'], label='Kelembapan Tanah')
    ax.set_xlabel('Index')
    ax.set_ylabel('Nilai')
    ax.legend()
    st.pyplot(fig)

# Fungsi untuk menampilkan grafik histori suhu, kelembapan, dan kelembapan tanah dalam grafik terpisah di sidebar
def sidebar_charts(data):
    st.sidebar.subheader('Grafik Histori Terpisah')
    
    st.sidebar.write('### Suhu')
    fig1, ax1 = plt.subplots()
    ax1.plot(data['temperature'], label='Suhu')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Suhu')
    st.sidebar.pyplot(fig1)

    st.sidebar.write('### Kelembapan')
    fig2, ax2 = plt.subplots()
    ax2.plot(data['humidity'], label='Kelembapan')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Kelembapan')
    st.sidebar.pyplot(fig2)

    st.sidebar.write('### Kelembapan Tanah')
    fig3, ax3 = plt.subplots()
    ax3.plot(data['soil_moisture'], label='Kelembapan Tanah')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Kelembapan Tanah')
    st.sidebar.pyplot(fig3)

# Membuat judul untuk aplikasi
st.title('Prediksi Pertumbuhan Jagung')

# Tombol untuk memperbarui prediksi
if st.button("Perbarui Sekarang"):
    # Menggunakan session_state untuk menandai perlu refresh
    st.session_state.refresh = True

# Jika refresh ditandai, lakukan refresh
if st.session_state.get('refresh', False):
    st.session_state.refresh = False
    st.experimental_rerun()
else:
    mean_pred, latest_pred, data = get_predictions()
    update_predictions()
    sidebar_charts(data)
