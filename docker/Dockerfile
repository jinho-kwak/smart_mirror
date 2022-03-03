# https://pythonspeed.com/docker/ 사이트 참고하여 개발
# 이미지 생성시 base가 되는 이미지 지정.



ARG TENSOR_VERSION=2.4.1
FROM tensorflow/tensorflow:${TENSOR_VERSION}
# gpu 버전을 사용하고 싶으면 아래 사용
# FROM tensorflow/tensorflow:${TENSOR_VERSION}-gpu

# TENSOR_VERSION are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG TENSOR_VERSION


# apt update, upgrade
# DEBIAN_FRONTEND 사용 이유 https://github.com/phusion/baseimage-docker/issues/319
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt-get install -y libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
libharfbuzz-dev libfribidi-dev libxcb1-dev
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y cron
RUN apt-get install -y vim
RUN apt-get install -y net-tools
RUN apt-get install python3-opencv -y
RUN apt-get install -y libzbar0
#지역 설정/ 시간
ENV TZ=Asia/Seoul
RUN apt-get install -y tzdata

#locale 설정
ENV LC_ALL=C.UTF-8
ENV LANGUAGE=ko

# pip 필요 패키지 설치
RUN pip install --upgrade pip

# pip3 필요 패키지 설치
COPY requirements_pip.txt /tmp/
# COPY requirements_pip3.txt /tmp/
RUN pip install -r /tmp/requirements_pip.txt

# COPY smart_retail /home/

