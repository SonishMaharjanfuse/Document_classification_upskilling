FROM python:3.8-slim

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app

CMD ["streamlit", "run", "./Week_4/streamlit_demo.py"]
