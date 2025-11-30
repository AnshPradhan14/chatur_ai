# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

# Create a non-root user (required by Hugging Face)
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
# This is where your ML dependencies will be installed
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your ENTIRE project (including api.py, index.html, db.py, etc.)
COPY --chown=user . /app

# The command to start your FastAPI app:
# Note the change from 'app:app' to 'api:app'
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]