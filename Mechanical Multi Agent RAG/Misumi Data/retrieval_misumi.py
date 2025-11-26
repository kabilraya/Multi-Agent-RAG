from qdrant_client import QdrantClient,models
import re
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import time
#GLOBAL CONFIG
collection_name = "Misumi Bearing Nuts"
client = QdrantClient(url="http://localhost:6333")
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

def retrieval(query):
    #embedding the query
    retrieve_start = time.perf_counter()
    dense_query = next(dense_encoder.query_embed(query))
    sparse_query = next(sparse_encoder.query_embed(query=query))
    late_query = next(late_encoder.query_embed(query=query))

    prefetch = [
        models.Prefetch(
            query=dense_query,
            using="dense",
            limit=80,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query.as_object()),
            using = "sparse",
            limit = 80,
        )
    ]

    points = client.query_points(
        collection_name = collection_name,
        query=late_query,
        using= "lateinteract",
        prefetch=prefetch,
        limit = 30,
        with_payload = True,
        with_vectors=False,

    ).points
    retrieve_end = time.perf_counter()
    retrieval_time_total = retrieve_end - retrieve_start
    with open("retrieval_time.txt","a",encoding = "utf-8") as f:
        f.write(f"Query: {query}\n Time for retrieval: {retrieval_time_total}\n\n")
    return points

def retrieve_part_numbers(query):
    parts = retrieval(query)
    part_numbers = []
    for part in parts:
        part_number = part.payload.get("part_info")
        part_numbers.append(part_number)
    return part_numbers

def main():
    query = "C-RM25"
    parts = retrieve_part_numbers(query=query)

    for part in parts:
        print(f'{part} \n\n')

if __name__ == "__main__":
    main()