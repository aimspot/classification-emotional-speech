FROM python:3.6.13

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# CMD ["python3", "main.py"]
CMD ["python", "core/server.py"]