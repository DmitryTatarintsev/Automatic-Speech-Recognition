import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import scipy
import os
import speech_recognition as sR     # модуль для распознавания речи
import librosa
from jiwer import wer               # модуль для оценки распознанного текста по методу WER
from pydub import AudioSegment      # модуль конвертирования аудио

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Настройки записи
duration = 20  # Длительность записи в секундах
fs = 44100  # Частота дискретизации
channels = 1  # Количество каналов

# Запись звука с микрофона
print("Система распознания аварийных вызовов.")
print("\nНажмите Enter, чтобы начать запись...")
input()
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
sd.wait()

# Сохранение записанного звука в формате WAV
with io.BytesIO() as buffer:
    # Укажите формат "wav" при записи в buffer
    sf.write(buffer, audio_data, fs, subtype="PCM_24", format="wav") 
    print("Запись завершена! Файл сохранен как 'recording.wav'")
    with open("recording.wav", "wb") as f:
        f.write(buffer.getvalue())
# Распозание речи
def recognizeAudio(filename, duration=None):
    AUDIO_FILE = os.path.join(filename) # задаем путь к аудиофайлу
    r = sR.Recognizer() # создаем объект класса Recognizer
    with sR.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source, duration=duration) # считываем аудиофайл
    return r.recognize_google(audio, language='ru') # запускаем распознавание
# Модель. Анализ сигнала
easy_model = lambda y, sr: "Авария!" if np.min(list(map(np.min, librosa.feature.chroma_stft(y=y, sr=sr)))) < 0.003046*.02 else "Ложный вызов."

print("\nРаспознание речи...")
try: print("Итог:",recognizeAudio('recording.wav'))
except: print("Речь слишком неразборчивая или ее нет вовсе.")

print("\nАнализ вызова...")
y, sr = librosa.load('recording.wav')
print(easy_model(y,sr))