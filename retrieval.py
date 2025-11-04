from qdrant_client import QdrantClient, models
import re
from fastembed import TextEmbedding,SparseTextEmbedding,LateInteractionTextEmbedding
#GLOBAL CONFIG
collection_name = "Smartphones Products"
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
client = QdrantClient(url = "http://localhost:6333")


def get_product_ids(query):
    
    # embedding of the given query

    dense_query = next(dense_encoder.query_embed(query))
    sparse_query = next(sparse_encoder.query_embed(query))
    late_query = next(late_encoder.query_embed(query))

    prefetch = [
    models.Prefetch(query = dense_query,
                    using = "dense",
                    limit = 10,),
    models.Prefetch(query = models.SparseVector(**sparse_query.as_object()),
                    using = "sparse",
                    limit = 10,
    )
    ]

    query_results = client.query_points(
    collection_name = collection_name,
    prefetch=prefetch,
    query = late_query,
    using = "lateInteraction",
    limit = 10,
    with_payload = True,
    with_vectors = False
    ).points


    product_ids = []
    for result in query_results:
        product_id = result.payload.get("product_id")
        product_ids.append(product_id)
    
    return product_ids

def retrieve_relevant_documents(query):
    retrieved_products = get_product_ids(query=query)
    print(retrieved_products)
    all_chunks = []
    for product in retrieved_products:
        info,_ = client.scroll(
        collection_name = collection_name,
        scroll_filter= models.Filter(
            must = [
                models.FieldCondition(key = "product_id",match = models.MatchValue(value = product))
            ]
        ),
        limit=30,
        with_payload = True,
        with_vectors = False
        )
        chunks_of_a_product = []
        for point in info:
            chunk = point.payload.get("chunk")
            cleaned_chunks = re.sub(r'[ \t]+', ' ', chunk.strip())
            chunks_of_a_product.append(cleaned_chunks)
    
        all_chunks.append(chunks_of_a_product)
    print(len(all_chunks))
    print(all_chunks)
    # return all_chunks

    
retrieve_relevant_documents("Amoled")