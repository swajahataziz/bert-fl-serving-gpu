# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM tensorflow/tensorflow:1.15.2-gpu-py3


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    pip install bert-for-tf2 sentencepiece tensorflow-hub pandas==0.24.2 Flask flask-cors pillow numpy ipywidgets waitress

COPY ./model /model

COPY ./app /app

WORKDIR /app

EXPOSE 8000
CMD ["python", "app.py"]