# Machine Learning deployment template
This repo is for a machine learning model, downloaded from model registry and transformers available in hugging faces, and a template to serve this model with FastAPI and some input/output data and model validations

# Project Setup

This project requires Docker and Docker Compose to be installed on your machine. You can download Docker here and Docker Compose here. An internet connection is also required to download the Python image.

## Using Docker Compose
To run the project, use the following command in your terminal:

if you made changes in code run this first

```
docker-compose build
```

then to run the project
```
docker-compose up
```
if you need to change the port or reload values, you can do so in the docker-compose.yml file.

## Using Makefile

### RUN
If your machine is compatible with Make, you can use the Makefile to build and run the project. Use the following command in your terminal:

on command line:
```
    make run
```
### Google Cloud Artifact Registry Build/Push
To build and push the Docker image to Google Cloud Artifact Registry, you need to customize the PROJECT_ID, REGION, REPOSITORY, IMAGE, and TAG variables in the Makefile. Then, you can use the following commands:
```
make build
make push
```

### Kubernetes Deployment
With the image uri and tag you can deploy this project on a kubernetes cluster.
Check values on deployment.yaml.
Make sure to have kubectl configured on your machine.
```
kubectl -f deployment.yaml
```

# API Usage
The API has two main routes: /inference and /mock_inference.

## /inference
This route is used to make predictions with the model. It accepts a POST request with a csv_path parameter that specifies the path to the CSV file with the data to be predicted.

Example:
```
curl -X POST "http://localhost:8080/inference" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"csv_path\":\"path_to_your_file.csv\"}"
```

## /mock_inference
This route is used to test the model with mock data. It accepts a GET request and returns the predicted labels for the mock data.

Example:
```
curl -X GET "http://localhost:8080/mock_inference" -H  "accept: application/json"
```

For more information about the API and its routes, please visit the /docs route in your browser after running the project.

# TO-DO

- padronize model responses in model and api.py
- use panda methods to do the predictions/test_Y validation in model_validation
- use standard python project structure, and populate tests