# Use an official Python runtime as a parent image
FROM python:3.11.3

# Set the working directory in the container to /app
WORKDIR /app

# Add current directory code to /app in container
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requiments.txt

# Make port 80 available to the world outside this container
EXPOSE 8080

# Run the command to start uvcorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]