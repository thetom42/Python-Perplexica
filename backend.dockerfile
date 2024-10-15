FROM python:3.12-slim

WORKDIR /home/perplexica

# Copy the backend directory contents
COPY backend /home/perplexica/backend

# Copy necessary files
COPY backend/requirements.txt /home/perplexica/
#COPY backend/setup.py /home/perplexica/
COPY backend/alembic.ini /home/perplexica/

# Create data directory
RUN mkdir /home/perplexica/data

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run database migrations
RUN alembic upgrade head

# Set the command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3001"]
