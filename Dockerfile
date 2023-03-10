FROM python:3.8-slim-buster

RUN apt-get update \
    && apt-get install -y build-essential

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "mlflow_utils.py"]