from fastapi import FastAPI
from app.routers import hello

# Initialise the FastAPI application
app = FastAPI()

# Register routers
app.include_router(hello.router)