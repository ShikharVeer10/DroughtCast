FROM python:3.14.7-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
CMD ["python", "-m", "data.build_dataset"]
