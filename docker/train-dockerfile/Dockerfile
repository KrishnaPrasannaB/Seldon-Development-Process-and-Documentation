FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN ls -la /app/*

RUN python3 --version
RUN pip3 --version

RUN pip install -r ./requirements.txt
#ADD ./TrainWineClassifier.py /
CMD ["python","TrainWineClassifier.py"]

RUN ls -la /app/*