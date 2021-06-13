#!/bin/bash
# This is to build and deploy the docker image implementing the OCR workflow. It starts from the myocr image
# that houses the aster OCR package

AWSENVPATH=/home/ankur/dev/apps/ML/OCR/credentials/awsenv.list
OCRENVPATH=/home/ankur/dev/apps/ML/OCR/aster-ocr-workflow/ocr_env_docker.list
IMAGE_NAME=ocr_prefect
CONTAINER_NAME=ocr_prefect

# build docker image on master
docker build -t $IMAGE_NAME .

# Now deploy to any worker nodes on the kubernetes cluster
REPONAME=aster-ocr-workflow
. ../deploy/deploy_workers.sh
# This connects to the worker node using ssh, pulls latest code and builds docker image on the worker
deploy_workers ${IMAGE_NAME} ${REPONAME}

# If input image provided as argument, run docker on that image. This can be used as a test
if [ "$1" ]; then
  echo "Running OCR on image:" $1
  echo "reading AWS access credentials from "$AWSENVPATH
  echo "reading OCR configuration parameters from "$OCRENVPATH
  docker run \
    --env-file $AWSENVPATH \
    --env-file $OCRENVPATH \
    --shm-size 8G \
    --network="host" \
    -i --cpus="5" ocr_prefect -i $1
fi
