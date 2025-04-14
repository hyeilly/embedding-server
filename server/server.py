import grpc
from concurrent import futures
import time
from generated import embed_pb2, embed_pb2_grpc
import numpy as np

class Embedder(embed_pb2_grpc.EmbedderServicer):
    def GetEmbedding(self, request, context):
        # 요청된 텍스트
        texts = request.texts
        print(f"Received texts: {texts}")

        # 임베딩 벡터 생성 (여기서는 더미 벡터 사용)
        embeddings = np.random.rand(len(texts), 128)  # 예시로 128차원 임베딩 벡터 생성
        dimension = embeddings.shape[1]  # 임베딩 벡터의 차원

        # EmbedResponse 생성
        response = embed_pb2.EmbedResponse()

        for embedding in embeddings:
            response.embeddings.extend(embedding.tolist())  # 각 벡터를 실수 값 리스트로 추가

        response.dimension = dimension  # 차원 추가

        print(f"Generated embeddings: {embeddings.tolist()}")
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embed_pb2_grpc.add_EmbedderServicer_to_server(Embedder(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started, listening on port 50051...")
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)  # 1일 동안 서버 실행
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
