from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import grpc
from generated import embed_pb2, embed_pb2_grpc

app = FastAPI()

# 요청 바디 스키마
class EmbedRequest(BaseModel):
    text: str

# gRPC 채널 및 스텁 설정
channel = grpc.insecure_channel("192.168.170.217:50051")
stub = embed_pb2_grpc.EmbedderStub(channel)

@app.post("/embed")
def get_embedding(request: EmbedRequest):
    try:
        grpc_request = embed_pb2.EmbedRequest(texts=[request.text])
        grpc_response = stub.GetEmbedding(grpc_request)

        # 첫 번째 결과만 사용한다고 가정
        result = {
            "dimension": grpc_response.dimension,
            "embedding": list(grpc_response.vectors[0].values)
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
