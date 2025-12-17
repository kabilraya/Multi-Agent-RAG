from qdrant_client import QdrantClient, models
import os
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
client = QdrantClient(url="http://localhost:6333")
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
collection_name = "Test Misumi"

def specification_retrieval(user_query):
    
    res, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="part_name",match=models.MatchValue(value=user_query))
            ]
        ),
        limit = 1,
        with_payload=True,
        with_vectors=False
    
    )
    if res:
        specification = res[0].payload.get("specification")
        return specification
    else:
        print("No such part number found in the db")

def retrieve_similar_part_numbers(user_query):
    spec_part_retrieval = {}
    specification = specification_retrieval(user_query=user_query)
    parts = [p.strip() for p in specification.split(" | ")]
    cleaned_spec = []
    for p in parts:
        if ":" in p:
            cleaned_spec.append(p.strip())
    cleaned_spec.append(specification)
    for spec in cleaned_spec:
        dense_embed = next(dense_encoder.query_embed(spec))
        sparse_embed = next(sparse_encoder.query_embed(spec))
        late_embed = next(late_encoder.query_embed(spec))

        prefetch = [
            models.Prefetch(
                query = dense_embed,
                limit = 40,
                using = "dense_spec"
            ),
            models.Prefetch(
                query = models.SparseVector(**sparse_embed.as_object()),
                limit = 40,
                using = "sparse_spec"
            )
        ]
        points = client.query_points(
            collection_name=collection_name,
            using = "lateinteract",
            limit = 10,
            prefetch=prefetch,
            query=late_embed
        ).points
        part_names = []
        for point in points:
            part_name = point.payload.get("part_name")
            part_names.append(part_name)
        spec_part_retrieval[spec] = part_names
    return spec_part_retrieval





def main():
    user_query = "AWL24"
    part_number_dict = retrieve_similar_part_numbers(user_query=user_query)
    print(part_number_dict)

if __name__ == "__main__":
    main()

