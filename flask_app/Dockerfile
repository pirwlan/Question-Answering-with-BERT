FROM python:3.7-slim
WORKDIR /app
ADD ./flask_app /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 8000
CMD ["python", "app.py"]