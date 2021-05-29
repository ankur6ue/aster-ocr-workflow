# FROM prefecthq/prefect:0.14.5-python3.7 as base_image
FROM myocr as base_image

# Install pip
RUN python -m pip install --upgrade pip

ENV PREFECT__USER_CONFIG_PATH='/opt/prefect/config.toml'
# RUN pip show prefect || pip install git+https://github.com/PrefectHQ/prefect.git@0.14.5#egg=prefect[all_orchestration_extras]
RUN pip install prefect==0.14.9
RUN pip install wheel
# needed to get rid of ImportError: libGL.so.1:
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# needed to get rid of ImportError: libtk8.6.so
RUN apt-get install tk -y
RUN apt-get install nano
COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip install -r requirements.txt

WORKDIR /opt/ocr
COPY /source/*.py /opt/ocr/
COPY rec_args.json /opt/ocr
COPY det_args.json /opt/ocr

ENTRYPOINT ["python", "run_flow.py"]

