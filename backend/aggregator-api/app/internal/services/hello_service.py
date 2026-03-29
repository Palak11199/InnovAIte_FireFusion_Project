
# Service layer for hello logic
class HelloService:
    async def hello(self):
        return {"message": "Hello from the Aggregator API!"}
