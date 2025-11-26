from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding


#GLOBAL CONFIG
collection_name = "NSK Bearing Nuts"
client = QdrantClient(url = "http://localhost:6333")
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

def retrieval(query):
    #embedding the query into three vectors

    dense_embed = next(dense_encoder.query_embed(query))
    sparse_embed = next(sparse_encoder.query_embed(query))
    late_embed = next(late_encoder.query_embed(query))

    prefetch = [
        models.Prefetch(
            query = dense_embed,
            using = "dense",
            limit = 80
        ),
        models.Prefetch(
            query = models.SparseVector(**sparse_embed.as_object()),
            using = "sparse",
            limit=80
        )
    ]
    points = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        using = "lateinteract",
        query = late_embed,
        limit = 40,
        with_payload=True,
        with_vectors=True
    ).points

    return points

def part_numbers_retrieval(query):
    points = retrieval(query)
    all_parts = []
    for point in points:
        part_info = point.payload.get("part_info")
        all_parts.append(part_info)

    return all_parts

def main():
    query = "ANO2"
    all_parts = part_numbers_retrieval(query=query)
    for part in all_parts:
        print(part + "\n\n")
    
if __name__ == "__main__":
    main()