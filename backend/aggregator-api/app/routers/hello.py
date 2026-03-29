from fastapi import APIRouter, Depends
from app.internal.services.hello_service import HelloService
from app.dependencies import verify_api_key

# Router for hello endpoints
router = APIRouter(prefix="/hello", tags=["hello"])

@router.get("/", dependencies=[Depends(verify_api_key)])
async def hello(
    service: HelloService = Depends(HelloService)
):
    # Delegate to HelloService
    return await service.hello();