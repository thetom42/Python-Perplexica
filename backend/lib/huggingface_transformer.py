"""
Hugging Face Transformers Embeddings Module

This module provides a class for generating embeddings using Hugging Face Transformers models.
It implements the Embeddings interface from LangChain.
"""

from typing import List, Dict, Any
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class HuggingFaceTransformersEmbeddings(Embeddings):
    """
    A class for generating embeddings using Hugging Face Transformers models.

    This class implements the Embeddings interface from LangChain and provides methods
    for embedding documents and queries using a pre-trained transformer model.

    Attributes:
        model_name (str): The name of the pre-trained model to use.
        tokenizer: The tokenizer associated with the model.
        model: The pre-trained transformer model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFaceTransformersEmbeddings instance.

        Args:
            model_name (str, optional): The name of the pre-trained model to use.
                Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts (List[str]): A list of documents to embed.

        Returns:
            List[List[float]]: A list of embeddings, one for each input document.
        """
        return self._run_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text (str): The query to embed.

        Returns:
            List[float]: The embedding of the input query.
        """
        return self._run_embedding([text])[0]

    def _run_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        This method tokenizes the input texts, runs them through the model,
        and applies mean pooling to generate the final embeddings.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            List[List[float]]: A list of embeddings, one for each input text.
        """
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean Pooling - Take attention mask into account for correct averaging
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).numpy()

        return embeddings.tolist()

    @property
    def _embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            int: The dimension of the embeddings produced by the model.
        """
        return self.model.config.hidden_size
