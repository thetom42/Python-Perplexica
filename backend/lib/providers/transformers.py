from huggingface_transformer import HuggingFaceTransformersEmbeddings

async def load_transformers_embeddings_models():
    models = {
        "all-MiniLM-L6-v2": HuggingFaceTransformersEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        "all-mpnet-base-v2": HuggingFaceTransformersEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    }
    return models
