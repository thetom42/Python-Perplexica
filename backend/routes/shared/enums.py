from enum import Enum

class FocusMode(Enum):
    GENERAL = "general"
    ACADEMIC = "academic"
    IMAGES = "images"
    VIDEOS = "videos"
    REDDIT = "reddit"
    WOLFRAM = "wolfram"
    YOUTUBE = "youtube"
    WEB = "web"

class OptimizationMode(Enum):
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"
