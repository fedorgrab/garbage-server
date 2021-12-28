FROM python:3.8

RUN apt update
RUN apt install -y python3-dev gcc

#ADD requirements.txt requirements.txt
#ADD export.pkl export.pkl
#ADD app.py app.py
#ADD run_server.sh run_server.sh
RUN mkdir data models repo
COPY garbage_server repo/garbage_server
COPY manage.py repo/manage.py
COPY requirements.txt repo/requirements.txt
COPY run_server.sh repo/run_server.sh
COPY model.pt models/model.pt
ENV NO_CUDA 1
# Install required libraries
RUN pip3 install -r repo/requirements.txt
RUN pip3 install torch

# Run it once to trigger resnet download
#RUN python app.py

EXPOSE 9999

# Start the server
CMD ["sh", "run_server.sh"]
