#!/bin/bash

AWSENVPATH=/home/ankur/dev/apps/ML/OCR/credentials/awsenv.list
OCRENVPATH=/home/ankur/dev/apps/ML/OCR/aster-ocr-workflow/ocr_env_docker.list
IMAGE_NAME=ocr_prefect
CONTAINER_NAME=ocr_prefect

SSH_KEY_LOCATION=~/.ssh/home_ubuntu_laptop/id_rsa
WORKER_IPS="../k8s/workers.txt"
REMOTE_DIR=~/dev/apps/ML/OCR
REPONAME=aster-ocr-workflow
# read passwords from file
while read -r line; do declare "$line"; done <"../credentials/pwords.txt"
REPOSRC=https://ankur6ue:${GITHUB_PWORD}@github.com/ankur6ue/${REPONAME}.git

# build docker image on master
docker build -t $IMAGE_NAME .

while IFS= read -r line; do
  echo "$line"
  worker_ip=$line
  echo ${LOCAL_WORKER_PWORD} | ssh -tt ankur@$worker_ip -i $SSH_KEY_LOCATION 'mkdir -p' ${REMOTE_DIR} '&& cd' ${REMOTE_DIR} \
    '&& git clone' ${REPOSRC} ${REPONAME} '2> /dev/null || (cd' ${REPONAME} '; git clean -f; git pull)'

  # build docker image on the worker
  echo ${LOCAL_WORKER_PWORD} | ssh -tt ankur@$worker_ip -i $SSH_KEY_LOCATION 'cd' ${REMOTE_DIR}'/'${REPONAME} \
    '&& docker build -t' ${IMAGE_NAME} '.'

done <"$WORKER_IPS"

# Takes the input image as parameter
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
