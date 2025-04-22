import grpc
from concurrent import futures
from sentence_transformers import SentenceTransformer
import torch

from generated import embed_pb2, embed_pb2_grpc

# 모델 로딩
MODEL_PATH = "/home/dev/embedding-server/KcELECTRA/models"  # 너가 클론한 경로
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_PATH, device=device)

# gRPC 서비스 구현
class EmbedderService(embed_pb2_grpc.EmbedderServicer):
    def GetEmbedding(self, request, context):
        texts = request.texts
        print(f"💬 받은 문장 개수: {len(texts)}")

        # 임베딩 수행
        embeddings = model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]

        # 응답 준비
        vectors = []
        for emb in embeddings:
            vector = embed_pb2.Vector(values=emb.tolist())
            vectors.append(vector)

        return embed_pb2.EmbedResponse(
            vectors=vectors,
            dimension=dim,
            embeddings=embeddings.flatten().tolist()  # optional
        )

# 서버 실행
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embed_pb2_grpc.add_EmbedderServicer_to_server(EmbedderService(), server)
    server.add_insecure_port("[::]:50051")
    print("🚀 gRPC 서버 실행 중 (포트 50051)...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
