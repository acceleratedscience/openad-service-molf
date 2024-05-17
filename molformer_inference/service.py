from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from call_property_services import service_requester


app = FastAPI()

requester = service_requester()


@app.get("/health", response_class=HTMLResponse)
async def health():
    return "UP"


@app.post("/service")
async def service(property_request: dict):
    result = requester.route_service(property_request)
    return result


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
