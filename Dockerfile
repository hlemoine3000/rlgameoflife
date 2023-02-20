FROM python:3.11.1-buster

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV MPLCONFIGDIR=/rlgameoflife/.cache/matplotlib

# Create the user
ARG USERNAME=devcontainer
ARG USER_UID=1000
ARG USER_GID="$USER_UID"
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
# Set the default user.
USER $USERNAME

WORKDIR /rlgameoflife