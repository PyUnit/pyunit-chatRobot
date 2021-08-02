FROM python:3.7

MAINTAINER Jytoui <jtyoui@qq.com>
EXPOSE 80

COPY ./main.py /app/main.py
COPY ./requirements.txt /requirements.txt

WORKDIR /app

# 安装Python3环境
RUN wget -O /mnt/model.tar.gz https://github.com/PyUnit/pyunit-chatRobot/releases/download/v1.0/model.tar.gz && \
    tar -zxvf /mnt/model.tar.gz  -C /app/ && \
    rm -rf /mnt/model.tar.gz && \
    pip3 install --no-cache-dir -r /requirements.txt

CMD ["python3","main.py"]