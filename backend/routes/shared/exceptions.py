from fastapi import HTTPException
from typing import Optional, Any, Dict


class BaseAPIException(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class InvalidModelConfigurationError(BaseAPIException):
    def __init__(self, detail: str = "Invalid model configuration"):
        super().__init__(status_code=400, detail=detail)


class InvalidInputError(BaseAPIException):
    def __init__(self, detail: str = "Invalid input"):
        super().__init__(status_code=422, detail=detail)


class ServerError(BaseAPIException):
    def __init__(self, detail: str = "An internal server error occurred"):
        super().__init__(status_code=500, detail=detail)
