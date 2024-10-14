# Perplexica Backend

This is the Python backend for the Perplexica project, implemented using FastAPI and LangChain.

## Prerequisites

- Python 3.12
- `toml` package (install via `pip install toml` or `conda install -c conda-forge toml`)

## Setup

1. Create a virtual environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate  # On Windows
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run database migrations:
   ```bash
   alembic upgrade head
   ```

4. Create a default `config.toml` file (if it doesn't exist):
   ```bash
   touch config.toml
   ```
   Update the `config.toml` file in the project root directory with your actual API keys and endpoints.

5. Ensure Redis is installed and running on your system. You can download it from the [official Redis website](https://redis.io/download).


## Running the Backend

To run the backend server, use the following command:

```bash
uvicorn main:app --reload
```

This will start the server on `http://localhost:8000`. The `--reload` flag enables auto-reloading when you make changes to the code.

## API Documentation

Once the server is running, you can access the auto-generated API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## WebSocket Connection

The WebSocket endpoint is available at `ws://localhost:8000/ws`. You can use this to establish real-time communication with the backend for features like web search and chat functionality.

## Project Structure

- `main.py`: The entry point of the application.
- `config.py`: Handles configuration management.
- `routes/`: Contains API route definitions.
- `agents/`: Implements AI agents for various functionalities.
- `lib/`: Contains utility functions and integrations.
- `utils/`: Provides utility functions for various tasks.
- `websocket/`: Implements WebSocket functionality.
- `db/`: Handles database operations and models.
- `alembic/`: Contains database migration scripts.
- `setup.py`: Script for setting up the development environment.
- `tests/`: Contains unit and integration tests.
- `locustfile.py`: Contains load testing scenarios.

## Database Migrations

To create a new migration after changing the database models:

```bash
alembic revision --autogenerate -m "Description of the changes"
```

To apply migrations:

```bash
alembic upgrade head
```

To revert the last migration:

```bash
alembic downgrade -1
```

## Caching

The application uses Redis for caching. Make sure Redis is running and the `REDIS_URL` in the config file is correct. The `cache_result` decorator in `utils/cache.py` can be used to cache function results.

## Testing

To run all tests (unit and integration), use the following command:

```bash
pytest
```

To run only unit tests:

```bash
pytest tests/test_*.py
```

To run only integration tests:

```bash
pytest tests/test_integration.py
```

## Load Testing

To run load tests using Locust:

1. Start your backend server:
   ```bash
   uvicorn main:app
   ```

2. In a new terminal, run the Locust command:
   ```bash
   locust -f locustfile.py
   ```

3. Open a web browser and go to `http://localhost:8089` to access the Locust web interface.

4. Enter the number of users to simulate, spawn rate, and the host (e.g., http://localhost:8000).

5. Start the test and monitor the results in real-time.

Remember to adjust the load testing scenarios in `locustfile.py` as needed for your specific use cases.

## Error Handling

The application implements comprehensive error handling and logging. All exceptions are caught and logged, with appropriate error responses sent back to the client.

## Contributing

Please refer to the CONTRIBUTING.md file in the project root for guidelines on how to contribute to this project.

## Next Steps

1. Conduct thorough integration testing with the frontend.
2. Analyze load testing results and optimize performance as needed.
3. Create user documentation for the entire Perplexica application.
