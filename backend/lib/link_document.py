"""
Link Document Processing Module

This module provides functionality for retrieving and processing documents from web links.
It supports both HTML and PDF content, and splits the retrieved text into manageable chunks.
"""

import httpx
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pypdf
import io
import logging
from typing import List

async def get_documents_from_links(links: List[str]) -> List[Document]:
    """
    Retrieve and process documents from a list of web links.

    This function performs the following steps for each link:
    1. Fetches the content from the link.
    2. Determines if the content is HTML or PDF.
    3. Extracts and processes the text content.
    4. Splits the text into smaller chunks.
    5. Creates Document objects with the processed text and metadata.

    Args:
        links (List[str]): A list of URLs to retrieve and process.

    Returns:
        List[Document]: A list of Document objects, each containing a chunk of
        processed text and metadata (title and URL).

    Raises:
        Exception: If there's an error processing a link, it's logged and a
        Document object with an error message is created instead.
    """
    splitter = RecursiveCharacterTextSplitter()
    docs = []

    async with httpx.AsyncClient() as client:
        for link in links:
            try:
                response = await client.get(link)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")

                if "application/pdf" in content_type:
                    pdf_content = io.BytesIO(response.content)
                    pdf_reader = pypdf.PdfReader(pdf_content)
                    text = " ".join(page.extract_text() for page in pdf_reader.pages)
                else:
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)

                text = " ".join(text.split())  # Normalize whitespace
                split_texts = splitter.split_text(text)

                for split_text in split_texts:
                    docs.append(
                        Document(
                            page_content=split_text,
                            metadata={
                                "title": soup.title.string if soup.title else link,
                                "url": link,
                            }
                        )
                    )

            except Exception as e:
                logging.error("Error processing link %s: %s", link, str(e))
                docs.append(
                    Document(
                        page_content=f"Failed to retrieve content from the link: {str(e)}",
                        metadata={
                            "title": "Failed to retrieve content",
                            "url": link,
                        }
                    )
                )

    return docs
