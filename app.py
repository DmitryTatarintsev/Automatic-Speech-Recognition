import streamlit as st #  создания веб-приложений
import sounddevice # запись и воспроизведение звука
import soundfile # для чтения и записи звуковых файлов
import io # модуль io для работы с потоками ввода-вывода
import numpy as np # для работы с массивами и математическими операциями
import librosa # для анализа и обработки аудио
import os #  модуль os для работы с файловой системой
import speech_recognition # SpeechRecognition для распознавания речи
import tempfile  # временная запись аудио файла

st.title("Система распознания аварийных вызовов")
st.write("Для начала, выберите индекс устройства записи звука из списка доступных устройств на вашем пк в графе 'Настройка средств записи'")
devices = sounddevice.query_devices()
# Вывод списка устройств ввода
input_devices = [device for device in devices if device['max_input_channels'] > 0]
# Вывод стандартного устройства ввода
default_input_device = sounddevice.default.device[0]
default_input_name = devices[default_input_device]['name']
with st.expander("Настройка средств записи"):
    st.write(f"Устройство ввода по умолчанию: {default_input_name}. Индекс {default_input_device}.")
    st.write(" ")
    st.table(input_devices)
    # Выбор устройства ввода пользователем
    input_device_index = st.selectbox(
        "Выберите индекс устройства ввода звука",
        options=list(range(len(input_devices))),  # Список индексов устройств
        index=default_input_device  # Значение по умолчанию (стандартное устройство)
    )
st.write(f"Выбранное устройство: {input_devices[input_device_index]['name']}")
st.write(" ")
st.write(" ")
st.write(" ")

st.write("**Памятка оператора аварийной службы**")

with st.expander("*Важная информация*"):
    st.markdown("- Первый запуск приложения может занимать больше времени.")
    st.markdown("- Могут возникать ошибки записи. Приложение об этом предупредит. Проверьте формат входных данных и убедитесь, что устройство ввода аудиосигнала подключено и работает корректно.")
    st.markdown("- Точность распознавания аварий составляет 9 из 10 случаев. Тем не менее, алгоритм не учитывает все возможные случаи и носит рекомендательный характер.")
    st.markdown("- Иногда возникает аномалия модуля распознания, 'анализ сигнала' может принять разговор оператора с пострадавшим за 'аварийный случай', когда вызов ложный.")

with st.expander("*Ответственность*"):
    st.markdown("- Вся ответственность по-прежнему лежит на операторе.")
    st.markdown("- Пожалуйста, примите это во внимание, на кону человеческие жизни!")

with st.expander("*Принцип работы*"):
    st.markdown("- Временная запись первых 20 секунд.")
    st.markdown("- Распознание речи.")
    st.markdown("- Анализ спектра частот для выявления признаков аварийной ситуации.")
    st.markdown("- Удаление временной записи из 20 секунд.")

st.write(" ")
st.write(" ")
st.write(" ")
st.markdown("Нажмите кнопку 'Начать запись' или загрузите аудио файл, что бы запустить работу.")
st.markdown("В режиме загрузки речевой модуль отсутствует.")

duration = 20  # Запись в течение 20 секунд
sample_rate = 44100  # Частота дискретизации 44100 Гц
channels = 1  # Моно-запись

audio_data = None

# Распозание речи
def recognizeAudio(audio_data):
    recognizer = speech_recognition.Recognizer()
    with io.BytesIO(audio_data) as buffer:
        audio_file = speech_recognition.AudioFile(buffer)
        with audio_file as source:
            audio = recognizer.record(source, duration=duration, offset=0)
        try:
            return recognizer.recognize_google(audio, language='ru')
        except speech_recognition.UnknownValueError:
            return "Речь слишком неразборчивая или ее нет вовсе."
        except speech_recognition.RequestError as e:
            return "Ошибка при запросе к службе распознавания речи; {0}".format(e)

# Модель. Анализ сигнала
def easy_model(y, sr):
    chroma_features = librosa.feature.chroma_stft(y=y, sr=sr)
    min_chroma = np.min(chroma_features)
    threshold = 0.003046 * 0.02
    if min_chroma < threshold: return "Предполагаю, аварийный случай. Вызывайте экстренные службы!"
    else: return "Предполагаю, ложный вызов. Решение за вами."

if st.button("Начать запись"):
    st.write("Запись пошла...")

    audio_data = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, device=input_device_index)
    sounddevice.wait()
    
    # Сохранение записанного звука в формате WAV (временный файл)
    with io.BytesIO() as buffer:
        try:
            soundfile.write(buffer, audio_data, sample_rate, subtype="PCM_24", format="wav")
            # Ожидание завершения записи
            sounddevice.wait()
            st.write("Запись завершена!")
            st.audio(buffer)
            st.write(" ")
            # Распознавание речи
            st.write("Распознавание речи...")
            try:
                speech_text = recognizeAudio(buffer.getvalue())
                st.write("Итог:", speech_text)
                if len(speech_text) == 0: st.write("Речь слишком неразборчивая или ее нет вовсе.")
            except: st.write("Речь слишком неразборчивая или ее нет вовсе.")
            st.write(" ")
    
            # Анализ вызова
            st.write("Анализ вызова...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:  # Используем tempfile
                temp_file.write(buffer.getvalue())  # Запишем байты в временный файл
                y, sr = librosa.load(temp_file.name)  # Используем имя файла
                st.write(easy_model(y, sr))
                temp_file.close()  # Закрываем временный файл
        except Exception as e: st.error(f"Ошибка при записи: {e}")

st.write(" ")
# Пользователь загружает аудио файл
uploaded_file = st.file_uploader("Загрузите аудио файл", type=["wav", "mp3"])

if uploaded_file is not None:
    # Чтение аудио файла
    audio_bytes = uploaded_file.read()
    
    # Создание временного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name

        st.write(" ")
        st.write("Анализ вызова...")
        
        # Загружаем аудио из временного файла
        y, sr = librosa.load(temp_file_path, duration=duration)
        # Выполняем анализ
        result = easy_model(y, sr)
        st.write(result)
        
