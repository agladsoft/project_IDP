FROM ubuntu:20.04

RUN ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime

RUN apt-get update \
 && apt-get install -y git tzdata

ADD ./ /mnt
WORKDIR /mnt
RUN apt-get update && apt-get -y upgrade && \
    apt-get -y install python3.8 && \
    apt update && apt install python3-pip -y
RUN apt-get install -y pip
RUN pip install -r requirements.txt
RUN pip install pdf2image
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y libleptonica-dev
RUN apt-get install -y libtesseract-dev
RUN apt-get install -y tesseract-ocr-eng
RUN apt-get install -y tesseract-ocr-rus
RUN apt-get install apt-utils wget build-essential cmake libfreetype6-dev pkg-config libfontconfig-dev libjpeg-dev libopenjp2-7-dev libcairo2-dev libtiff5-dev -y
RUN tar -xf poppler-data-0.4.10.tar.gz \
    && cd poppler-data-0.4.10 \
    && make install \
    && cd .. \
    && tar -xf poppler-21.03.0.tar.xz \
    && cd poppler-21.03.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd ../.. \
    && ldconfig
CMD tail -f /dev/null
