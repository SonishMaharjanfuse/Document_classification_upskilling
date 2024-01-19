FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app

CMD ["streamlit", "run", "./Week_4/streamlit_demo.py"]
