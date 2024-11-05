from fastapi import APIRouter

router = APIRouter()

# Import all route modules
from routes.images import router as images_router
from routes.videos import router as videos_router
from routes.config import router as config_router
from routes.suggestions import router as suggestions_router
from routes.chats import router as chats_router
from routes.search import router as search_router
from routes.models import router as models_router

# Include all routers with their prefixes
router.include_router(images_router, prefix="/images", tags=["images"])
router.include_router(videos_router, prefix="/videos", tags=["videos"])
router.include_router(config_router, prefix="/config", tags=["config"])
router.include_router(suggestions_router, prefix="/suggestions", tags=["suggestions"])
router.include_router(chats_router, prefix="/chats", tags=["chats"])
router.include_router(search_router, prefix="/search", tags=["search"])
router.include_router(models_router, prefix="/models", tags=["models"])
