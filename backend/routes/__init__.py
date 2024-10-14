from fastapi import APIRouter
from .images import router as images_router
from .videos import router as videos_router
from .config import router as config_router
from .models import router as models_router
from .suggestions import router as suggestions_router
from .chats import router as chats_router
from .search import router as search_router

router = APIRouter()

router.include_router(images_router, prefix="/images", tags=["images"])
router.include_router(videos_router, prefix="/videos", tags=["videos"])
router.include_router(config_router, prefix="/config", tags=["config"])
router.include_router(models_router, prefix="/models", tags=["models"])
router.include_router(suggestions_router, prefix="/suggestions", tags=["suggestions"])
router.include_router(chats_router, prefix="/chats", tags=["chats"])
router.include_router(search_router, prefix="/search", tags=["search"])