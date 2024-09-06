from fastapi import FastAPI

from graph_service.routers import ingest, retrieve

app = FastAPI()


app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get('/')
def read_root():
    return {'Hello': 'World'}
