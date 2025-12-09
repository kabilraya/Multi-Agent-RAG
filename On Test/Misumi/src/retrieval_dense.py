from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.abspath(os.path.join(current_dir,"..","test_logs","payload_score_logs"))
collection_name = "Dense Only Misumi"
client = QdrantClient(url="http://localhost:6333")
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
def retrieval(query):
    #embedding the query
    dense_embed = next(dense_encoder.query_embed(query))
    points = client.query_points(
        collection_name=collection_name,
        using="dense",
        query = dense_embed,
        limit = 10,
        with_payload=True,
        with_vectors=False
    ).points

    return points

def retrieve_part_numbers(query):
    points = retrieval(query=query)
    with open(file_name,"a",encoding="utf-8") as f:
        f.write(f"\n\n\nQuery: {query} \n\n")
        for point in points:
            payload = point.payload.get("part_info")
            score = point.score
            f.write(f"Scored Part Number: {payload}\n\nSimilarity Score: {score}\n\n\n")
            

def main():
    query = input("Enter the Part Number: ")
    retrieve_part_numbers(query)

if __name__ == "__main__":
    main()
