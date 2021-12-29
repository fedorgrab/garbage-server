FROM python:3.7-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

RUN mkdir data repo
COPY garbage_server garbage_server
COPY manage.py manage.py
COPY requirements.txt requirements.txt
COPY run_server.sh run_server.sh
COPY model.pt model.pt

# Install required libraries
RUN pip3 install -r requirements.txt

# Run it once to trigger resnet download

EXPOSE 8080

# Start the server
CMD ["sh", "run_server.sh"]
