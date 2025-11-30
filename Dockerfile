FROM python:3.9

WORKDIR /code

# Copy requirements first to cache dependencies
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for the embedding model cache and give permissions to user 1000
RUN mkdir -p /code/.cache && chmod -R 777 /code/.cache

# Switch to non-root user (required by Hugging Face)
USER 1000

# Start the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]