FROM python:3.12

# Создание рабочей директории
WORKDIR /app

# Добавление кода приложения
COPY app.py /app/

# Загрузка и обновление импортов
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install streamlit
RUN pip install sounddevice
RUN pip install soundfile
RUN pip install python-io
RUN pip install numpy
RUN pip install librosa
RUN pip install SpeechRecognition
RUN pip install temp
RUN pip install ffmpeg

# Установка переменных окружения
ENV PORT=8501

CMD ["streamlit", "run", "app.py"]

