# Contributing to Perplexica

## Setting up the Development Environment

### Backend Setup

1. In the root directory, locate the `.env.example` file.
2. Copy it to `.env` and fill in the necessary configuration fields specific to the backend.
3. Navigate to the `backend` directory and run:
   ```bash
   python setup.py
   ```
   This will:
   - Install required dependencies
   - Create a default .env file (if it doesn't exist)
   - Run database migrations

4. Start the backend server:
   ```bash
   uvicorn main:app --reload --port 3001
   ```

### Frontend Setup

1. Navigate to the `ui` directory
2. Copy `.env.example` to `.env` and fill in the necessary fields
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```

### SearXNG Setup

1. Install SearXNG and allow `JSON` format in the SearXNG settings.
2. Configure the SearXNG endpoint in your `.env` file.

## Development Guidelines

1. Create a new branch for your feature/fix
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request

## Code Style

- Python: Follow PEP 8 guidelines
- TypeScript/JavaScript: Use Prettier for formatting
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and concise

## Testing

- Write unit tests for new functionality
- Ensure existing tests pass
- Use pytest for Python tests
- Use Jest for TypeScript/JavaScript tests

## Documentation

- Update README.md for significant changes
- Document new features and API changes
- Keep code comments up to date
- Use clear commit messages

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with details of changes if needed
3. The PR will be merged once you have the sign-off of other developers

## Questions?

Feel free to open an issue for any questions about contributing.
