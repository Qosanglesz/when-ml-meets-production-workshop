#!/bin/bash

curl -X POST \
  "http://127.0.0.1:8090/api/v1.0/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [40.0, 65988.0, 23811.0, 703.0]
  }'

curl -X POST \
  "http://127.0.0.1:8090/api/v1.1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [40.0, 65988.0, 23811.0, 703.0]
  }'
