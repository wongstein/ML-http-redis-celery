FROM ubuntu:16.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3.5 \
         nginx \
         ca-certificates \
         libgcc-5-dev \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
RUN wget https://bootstrap.pypa.io/3.3/get-pip.py && python3.5 get-pip.py && \
    pip3 install numpy==1.14.3 scipy scikit-learn pandas flask gevent gunicorn && \
        (cd /usr/local/lib/python3.5/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
        rm -rf /root/.cache

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 5000
ENV DEBUG true

COPY ./api /api
COPY helpers.py /api/helpers.py
WORKDIR /api

# install requirements
RUN pip install -r requirements.txt

# expose the app port
EXPOSE 5000

RUN pip install gunicorn

# run the app server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "app:app"]