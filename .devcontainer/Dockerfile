FROM python:3.11

# Set working directory
WORKDIR /workspace

# Copy the entire project
COPY . .

# Set the default shell to bash
ENV SHELL /bin/bash

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Yarn
RUN npm install -g yarn

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
