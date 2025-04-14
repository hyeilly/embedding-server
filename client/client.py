import grpc
import sys
import os

from generated import embed_pb2, embed_pb2_grpc

import numpy as np


def run():
    texts = input("문장을 입력하세요 (쉼표로 구분): ").split(",")
    texts = [text.strip() for text in texts]  # 각 텍스트의 양옆 공백 제거

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = embed_pb2_grpc.EmbedderStub(channel)
        response = stub.GetEmbedding(embed_pb2.EmbedRequest(texts=texts))

        if not response.embeddings:
            print(response)
            print("임베딩 벡터가 비어 있습니다.")
            return

        print("임베딩 벡터 차원:", response.dimension)
        try:
            embedding_matrix = np.array(response.embeddings).reshape(-1, response.dimension)
            print(embedding_matrix)
        except ValueError as e:
            print(f"배열 reshape 오류: {e}")

if __name__ == "__main__":
    run()
