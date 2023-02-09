FROM python:3.11.1-buster

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV MPLCONFIGDIR=/rlgameoflife/.cache/matplotlib

WORKDIR /rlgameoflife