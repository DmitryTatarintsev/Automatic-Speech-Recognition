import streamlit as st
import numpy as np
import librosa

st.title("Система распознания аварийных вызовов.")

# модель
easy_model = lambda y, sr: "Авария. Вызывайте экстренные службы!" if np.min(list(map(np.min, librosa.feature.chroma_stft(y=y, sr=sr)))) < 0.003046*.02 else "Ложный вызов."

uploaded_file = st.file_uploader("Загрузите аудиофайл, и я распознаю экстренный случай", type=["mp3", "wav"])

if uploaded_file is not None:
    st.subheader("Результат распознания")
    y, sr = librosa.load(uploaded_file)
    st.write("Запись завершена! Файл сохранен как 'recording.wav'")
    st.write(" ")
    st.write("Распознание речи...")
    st.write("Нет модуля в связи с трудностями интеграции в Hugging Face.")
    st.write(" ")
    st.write("Анализ вызова...")
    st.write(easy_model(y, sr))

st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write("Подробнее о проекте и полная функциональная версия приложения здесь - https://github.com/alecseiterr/Icon_Soft/tree/main/Tatarintsev_Dmitry")
st.write("Автор: https://t.me/dtatarintsev")

    
