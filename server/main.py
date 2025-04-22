import threading
from fastapi import FastAPI
from server.grpc_server import serve_grpc

app = FastAPI()

@app.on_event("startup")
def start_grpc_server():
    threading.Thread(target=serve_grpc, kwargs={"device": "cuda:0", "port": 50051}, daemon=True).start()

@app.get("/")
def read_root():
    return {"message": "FastAPI with gRPC backend is running"}
