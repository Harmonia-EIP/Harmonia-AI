FROM python:3.11

WORKDIR /app

COPY requirement.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 && \
    pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--chdir", "scripts", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "server:app"]
