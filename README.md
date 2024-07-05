<div>
    <img src="https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/img/header.jpg" alt="альтернативный текст" title="заголовок изображения" width="auto" style="float: left; margin-right: 10px">
</div>

# 🔊📳🆘 ИИ-система обработки звонков про экстренном реагировании на ДТП

<details>
<br> 

> Для эффективного реагирования на экстренные ситуации, такие как ДТП, необходимо разработать модель нейронной сети, способную быстро анализировать аудиозаписи звонков и определять, является ли вызов ложным или истинным. Цель проекта - разработать модель, которая в течение 20 секунд после начала звонка определит характер вызова и выдаст рекомендацию по дальнейшим действиям оператору.

**Интерфейс (Демо версия)**: [РАБОЧАЯ ССЫЛКА](https://huggingface.co/spaces/dmitry212010/emergency_calls_net_demo)

**Статус проекта: завершен.**

**Стэк**: *streamlit, sounddevice, soundfile, io, numpy, librosa, os, speech_recognition, tempfile*.

<details>
<br> 
<h5>⚗️ Описание стэка </h5>

- `streamlit`: *создания веб-приложений*
- `sounddevice`: *запись и воспроизведение звука*
- `soundfile`: *для чтения и записи звуковых файлов*
- `io`: *модуль io для работы с потоками ввода-вывода*
- `numpy`: *для работы с массивами и математическими операциями*
- `librosa`: *для анализа и обработки аудио*
- `os`: *модуль os для работы с файловой системой*
- `speech_recognition`: *SpeechRecognition для распознавания речи*
- `tempfile`: *временная запись аудио файла*

</details>

**Цель:**
- Разработать модель нейронной сети, способную анализировать аудиозаписи звонков и определять экстренные ситуации.
- Обеспечить оперативное определение ложных и истинных вызовов в течение 20 секунд.
- Предоставить оператору рекомендацию по дальнейшим действиям на основе результатов анализа.


**План:** *анализ стека разработки **-->** парсинг датасета **-->** создание демонстрационной версии*

</details>
</details>

<details>
<br> 
<summary><h4>📂 Файлы </h4></summary>

🪄 *Парсинг.*
- - `processing_module.py`: Модуль предобработки данных. (Не требуется)
- - [`data_info.ipynb`](https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/experimental_laboratory/data_info.ipynb): Конструктор таблиц для анализа данных и обучения модели.

<br>

🧬 *Обучение.*
- - [`package_versions.txt`](https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/package_versions.txt): Актуальные версии библиотек.
- - [`requirements.txt`](https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/requirements.txt): Импорты необходимые для проекта.
- - [`train.ipynb`](https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/train.ipynb): Модель анализа аудиозаписей. 

<br>

💻 *Интерфейс.*
- - [`app.py`](https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/app.py): Модуль взаимодействия с оператором. Отвечает за взаимодействие с оператором и передачу информации о вызове и рекомендации. Рабочий многофункциональный интерфейс.
- - [`app-code-cell-demo.py`](https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/app-code-cell-demo.py): Интерфейс взаимодействия через ячейку кода в код ридере типа VisualStudio.
- - [`app-huggingface-demo.py`](https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/experimental_laboratory/app-huggingface-demo.py): Образец интерфейса для HuggingFace.

<details>
<br> 
РАБОЧАЯ ССЫЛКА - https://huggingface.co/spaces/dmitry212010/emergency_calls_net_demo

<div>
    <img src="https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/img/interface.png" alt="альтернативный текст" title="заголовок изображения" width="auto" style="float: left; margin-right: 10px">
</div>

**Версия для запуска из ячейки кода.** app-code-cell-demo.py
https://github.com/alecseiterr/Icon_Soft/blob/main/Tatarintsev_Dmitry/emergency_calls_net_sample.ipynb

```python
import app-code-cell-demo
```
```
Система распознания аварийных вызовов.

Нажмите Enter, чтобы начать запись...

Запись завершена! Файл сохранен как 'recording.wav'

Распознание речи...
Итог: Здравствуйте Меня зовут Дмитрий это система ГЛОНАСС пожалуйста расскажите что у вас произошло

Анализ вызова...
Ложный вызов.
```

Принцип работы. Запустить приложение (ячейку с кодом). Нажать Enter для записи как просят. Через 20 секунд запись прерывается. Начинается распознание сигнала: речь, вероятность аварии. Вывод всех результатов на экран.

Модуль захвата первых 20 секунд звонка. Модуль распознания речи от гугла. Анализ цветности частот как модель.

Дополнительных файлов не требуется, только  app.py .

<br>

**Полная функциональная версия.** app.py

<div>
    <img src="https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition/blob/main/img/full-app.png" alt="альтернативный текст" title="заголовок изображения" width="auto" style="float: left; margin-right: 10px">
</div>

Что бы установить полную функциональную версию вам потребуется докер образ или пакет-установщик anaconda.

Запуск через Anaconda CMD.exe Prompt
```CMD.exe
C:\Users\...>git clone https://github.com/DmitryTatarintsev/Automatic-Speech-Recognition.git
C:\Users\...>cd "C:\Users\...\Automatic-Speech-Recognition"
C:\Users\...\Automatic-Speech-Recognition>pip install -r requirements.txt
C:\Users\...\Automatic-Speech-Recognition>python -m streamlit run app.py
```

"C:\Users\..." - локальный каталог куда загружаются файлы с github

Ваш основной браузер автоматически запустит http://localhost с приложением.

Для отключения приложения достаточно выйти из браузера или в консоле нажав на кнопки клавиатуры "Cntrl" + "C"

</details>
</details>

<details>
<br> 
<summary><h4>🎲 Расписание задач проекта </h4></summary>

> Здесь указан примерный план и выполненные задачи. Задачи будут пополнятся по мере продвижения. Дедлайны плановых этапов и дедлайны их задач могут не совпадать, все сроки условные.
>
> Статусы задач: **Задача поставлена.** --> **В процессе.** --> **Завершено.**

<br>

Анализ стека **01.05 ~ 03.06.2024:**

- ~~02.05-09.05.2024: aнализ стека текста.~~ **Завершено.**
- ~~09.05-03.06.2024: aнализ стека обработки стримингового аудио в Python.~~ **Завершено.**

<br>

Парсинг **23.05 ~ 31.07.2024:**

- ~~23.05-30.05.2024: Aугментации датасета.~~ **Завершено.**
- ~~23.05-01.07.2024: Парсинг первых 70 файлов.~~ **Завершено.**
- ~~01.06.2024-01.07.2024: Объединить демо и vpn-сервера датасеты в один. Аугментировать.~~ **Завершено.**
- ~~03.06.2024-03.06.2024: Модуль предобработки данных.~~ **Завершено.**
<br>

Интерфейс **02.05 ~ 05.07.2024:**
- ~~02.05-02.05.2024: app.py - модуль взаимодействия с оператором. Заглушкa (без модели).~~ **Завершено.**
- ~~16.05-01.07.2024: train.ipynb - нейронная сеть для анализа аудиозаписей.~~ **Завершено.**
- ~~03.06.2024-03.06.2024: вписать модель вместо заглушки.~~ **Завершено.**

<br>

Тестирование **04.06 ~ 31.07.2024(Крайний срок):**
- ~~04.06-31.07.2024: провести тестирование модели на тестовом наборе данных, содержащем различные аудиозаписи звонков.~~ **Завершено.**
- ~~04.06-31.07.2024: оценить точность определения вызовов и скорость работы модели.~~ **Завершено.**
- ~~04.06-31.07.2024: провести интеграционное тестирование системы для проверки взаимодействия компонентов.~~ **Завершено.**

</details>

<details>
