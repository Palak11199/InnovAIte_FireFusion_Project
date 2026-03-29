from fastapi import APIRouter, Depends
from ..internal.services.hello_service import HelloService

router = APIRouter(prefix="/hello", tags=["hello"])

@router.get("/")
async def hello(
    service: HelloService = Depends(HelloService)
):
    return await service.hello()
