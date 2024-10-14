"""
Locust load testing file for the Perplexica backend.

This module defines a Locust user class for load testing the Perplexica API endpoints.
It simulates user behavior by sending requests to various API endpoints.
"""

from locust import HttpUser, task, between

class PerplexicaUser(HttpUser):
    """
    Simulated user class for load testing Perplexica API endpoints.

    This class defines the behavior of a simulated user, including
    the tasks they perform and the time they wait between tasks.
    """

    wait_time = between(1, 5)  # Wait between 1 and 5 seconds between tasks

    @task
    def web_search(self):
        """
        Simulates a web search request to the Perplexica API.

        This task sends a POST request to the web search endpoint
        with a sample query.
        """
        self.client.post("/api/search/web", json={
            "query": "What is the capital of France?",
            "history": []
        })

    @task
    def health_check(self):
        """
        Simulates a health check request to the Perplexica API.

        This task sends a GET request to the root API endpoint
        to check the health of the service.
        """
        self.client.get("/api")

    # Add more tasks for other endpoints as needed
