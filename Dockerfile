FROM python:3.11
WORKDIR /app
COPY . /app
USER root


RUN pip install --no-cache-dir -r requirements.txt

CMD ["pytest"]

CMD [ "python", "./main.py" ]