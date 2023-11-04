FROM ubuntu:20.04

RUN ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    apt-get update && \
    apt-get install -y git tzdata python3.8 python3-pip ffmpeg libsm6 libxext6 tesseract-ocr libleptonica-dev libtesseract-dev tesseract-ocr-eng tesseract-ocr-rus apt-utils wget build-essential cmake libfreetype6-dev pkg-config libfontconfig-dev libjpeg-dev libopenjp2-7-dev libcairo2-dev libtiff5-dev -y && \
    tar -xf poppler-data-0.4.10.tar.gz && cd poppler-data-0.4.10 && make install && cd .. && \
    tar -xf poppler-21.03.0.tar.xz && cd poppler-21.03.0 && mkdir build && cd build && cmake .. && make && make install && cd ../.. && \
    ldconfig

ADD ./ /mnt
WORKDIR /mnt
RUN pip install -r requirements.txt

CMD tail -f /dev/null
