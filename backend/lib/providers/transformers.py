from backend.lib.huggingface_transformer import HuggingFaceTransformersEmbeddings
from backend.utils.logger import logger

async def load_transformers_embeddings_models():
    try:
        embedding_models = {
            'xenova-bge-small-en-v1.5': {
                'displayName': 'BGE Small',
                'model': HuggingFaceTransformersEmbeddings(
                    model_name='Xenova/bge-small-en-v1.5'
                ),
            },
            'xenova-gte-small': {
                'displayName': 'GTE Small',
                'model': HuggingFaceTransformersEmbeddings(
                    model_name='Xenova/gte-small'
                ),
            },
            'xenova-bert-base-multilingual-uncased': {
                'displayName': 'Bert Multilingual',
                'model': HuggingFaceTransformersEmbeddings(
                    model_name='Xenova/bert-base-multilingual-uncased'
                ),
            },
        }

        return embedding_models
    except Exception as err:
        logger.error(f"Error loading Transformers embeddings model: {err}")
        return {}
