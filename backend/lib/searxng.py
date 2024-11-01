"""
SearxNG Search Module

This module provides functionality for performing searches using the SearxNG API.
It allows customization of search parameters and returns structured search results.
"""

import httpx
from typing import Dict, Any, List
from config import get_searxng_api_endpoint

async def search_searxng(query: str, opts: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Perform a search using the SearxNG API.

    This function sends a GET request to the SearxNG API with the provided query
    and optional parameters. It then processes the response and returns the
    search results and suggestions.

    Args:
        query (str): The search query string.
        opts (Dict[str, Any], optional): Additional search options to be passed
            to the SearxNG API. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing two keys:
            - 'results': A list of dictionaries, each representing a search result.
            - 'suggestions': A list of strings containing search suggestions.

    Raises:
        httpx.HTTPStatusError: If the HTTP request to the SearxNG API fails.
    """
    searxng_url = get_searxng_api_endpoint()
    url = f"{searxng_url}/search"

    params = {
        "q": query,
        "format": "json",
    }

    if opts:
        params.update(opts)

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    results: List[Dict[str, Any]] = data.get("results", [])
    suggestions: List[str] = data.get("suggestions", [])

    return {"results": results, "suggestions": suggestions}
