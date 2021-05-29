#!/bin/sh

AWSENVPATH=/home/ankur/dev/apps/ML/OCR/credentials/awsenv.list
OCRENVPATH=/home/ankur/dev/apps/ML/OCR/aster-ocr-workflow/ocr_env_docker.list
IMAGE_NAME=ocr_prefect
CONTAINER_NAME=ocr_prefect

echo "reading AWS access credentials from "$AWSENVPATH
echo "reading OCR configuration parameters from "$OCRENVPATH
docker build -t $IMAGE_NAME .
# Takes the input image as parameter
if [ "$1" ]
  then
    docker run \
--env-file  $AWSENVPATH \
--env-file $OCRENVPATH \
--shm-size 8G \
--network="host" \
-i --cpus="5" ocr_prefect -i $1
fi
