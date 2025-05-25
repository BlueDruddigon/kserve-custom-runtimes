#!/bin/bash

# load secrets from `.env`
source .env

# define new variables
MODEL_NAME=qwen2.5
SERVICE_NAME=test-predictor.${SERVICE_NAME}.svc.cluster.local
INPUT_V2_PATH=@./input-v2.json
# shellcheck disable=SC2034
INPUT_V1_PATH=@./input.json

# # API v1
# curl -v -H "Host: ${SERVICE_NAME}" -H "Content-Type: application/json" "http://${SERVICE_NAME}/v1/models/${MODEL_NAME}:predict" -d ${INPUT_V1_PATH}
# API v2
curl -v -H "Host: ${SERVICE_NAME}" -H "Content-Type: application/json" "http://${SERVICE_NAME}/v2/models/${MODEL_NAME}/infer" -d ${INPUT_V2_PATH}
