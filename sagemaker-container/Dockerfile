FROM ubuntu:24.04

RUN apt update -y
RUN apt upgrade -y

RUN apt install software-properties-common build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget curl unzip -y

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update -y
RUN apt install python3.11 -y

RUN cd /tmp && wget https://bootstrap.pypa.io/get-pip.py
RUN python3 /tmp/get-pip.py --break-system-packages

RUN cd /tmp && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install

COPY . .
RUN python3 -m pip install -r requirements.txt --break-system-packages

