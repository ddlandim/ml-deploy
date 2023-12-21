# Makefile

# Customize these as needed
PROJECT_ID=my-project-id
REGION=my-region
REPOSITORY=my-repository
IMAGE=my-image
TAG=latest

# Build the Docker image
build:
    docker build -t ${IMAGE}:${TAG} .

# Push the Docker image to Google Cloud Artifact Registry
push:
    docker tag ${IMAGE}:${TAG} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:${TAG}
    docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:${TAG}

# Run the project with Docker Compose
run:
    docker-compose build && docker-compose up

.PHONY: build push run