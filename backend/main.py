"""
Main module for the Perplexica backend FastAPI application.

This module sets up the FastAPI application, including CORS configuration,
database initialization, WebSocket setup, and exception handling.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from config import get_port, get_cors_origins
from routes import router
from websocket.websocket_server import init_websocket
from db.database import engine
from db.models import Base
from utils.logger import logger

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Initialize WebSocket
init_websocket(app)

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler for the FastAPI application.

    This function is called when the application starts up. It initializes
    the database tables and logs the startup process.
    """
    logger.info("Starting up the application")
    try:
        # Create database tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Error creating database tables: %s", str(e))
        raise

@app.get("/api")
async def root():
    """
    Root endpoint for the API.

    Returns:
        dict: A dictionary with a status message.
    """
    return {"status": "ok"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Exception handler for HTTP exceptions.

    Args:
        request (Request): The incoming request.
        exc (HTTPException): The raised HTTP exception.

    Returns:
        JSONResponse: A JSON response containing the error message.
    """
    logger.error("HTTP error: %s - %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.

    Args:
        request (Request): The incoming request.
        exc (Exception): The raised exception.

    Returns:
        JSONResponse: A JSON response with a generic error message.
    """
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."},
    )

if __name__ == "__main__":
    port = get_port()
    logger.info("Server is running on port %s", port)
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error("Error starting the server: %s", str(e))
        raise
