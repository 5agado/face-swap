# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

RUN mkdir /app/models

# See https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
RUN apt-get update && apt-get install -y \
	build-essential cmake \
	wget git vim \
	#for ffmpeg
	yasm pkg-config \
	libgtk-3-dev \
	libboost-all-dev \
	tk-dev #see https://stackoverflow.com/questions/5459444/tkinter-python-may-not-be-configured-for-tk

# Compile and install ffmpeg from source
RUN git clone https://github.com/FFmpeg/FFmpeg /root/ffmpeg && \
    cd /root/ffmpeg && \
    ./configure --enable-nonfree --disable-shared && \
    make -j8 && make install -j8

# Install any needed packages specified in requirements.txt
RUN python setup.py develop

# Copy the models in the target models folders