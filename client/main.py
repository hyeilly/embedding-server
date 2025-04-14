from fastapi import FastAPI, Request
from pydantic import BaseModel
import grpc

from generated import embed_pb2, embed_pb2_grpc
import numpy as np

app = FastAPI()

# gRPC 서버 주소
GRPC_SERVER_ADDRESS = "192.168.170.217:50051"

# 요청 모델
class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: TextRequest):
    # gRPC 채널 열기
    channel = grpc.insecure_channel(GRPC_SERVER_ADDRESS)
    stub = embed_pb2_grpc.EmbedderStub(channel)
    response = stub.GetEmbedding(embed_pb2.EmbedRequest(texts=[request.text]))
    if not response.embeddings:
        return {"result": False}
    try:
        embedding_matrix = np.array(response.embeddings).reshape(-1, response.dimension)
        print(embedding_matrix)
    except ValueError as e:
        print(f"배열 reshape 오류: {e}")

    return {"embedding": embedding_matrix.tolist()}


