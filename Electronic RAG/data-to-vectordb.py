#Imports
from langchain_text_splitters import MarkdownHeaderTextSplitter,TokenTextSplitter
from fastembed import TextEmbedding, LateInteractionTextEmbedding,SparseTextEmbedding
from qdrant_client import QdrantClient,models
import re

# Global Configs
file_name = "info-hukut.md"
collection_name = "Smartphones Products"
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
client = QdrantClient(url = "http://localhost:6333")
splitter = TokenTextSplitter(chunk_size = 128, chunk_overlap = 0)

def chunking_of_products(all_docs):
    all_chunks = []
    for doc in all_docs:
        #Joining of the metadata and the page_content to make a whole product description
        description = doc.metadata["laptop-name"] + "\n" + doc.page_content
        chunk = splitter.split_text(description)
        all_chunks.append(chunk)
    # print(all_chunks)
    return all_chunks

def create_collection_payloads():
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense" : models.VectorParams(
                    size = dense_encoder.embedding_size,
                    distance = models.Distance.COSINE
                ),
                "lateInteraction" : models.VectorParams(
                    size = late_encoder.embedding_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)
                )
            },
            sparse_vectors_config= {"sparse":models.SparseVectorParams(modifier=models.Modifier.IDF)}
        )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "product_id",
        field_schema = "integer"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name = "chunk_id",
        field_schema = "integer"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "chunk",
        field_schema = "keyword"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "URL",
        field_schema = "keyword"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "product_name",
        field_schema = "keyword"
    )

def to_vectordb():
    with open(file_name,"r",encoding="utf-8") as f:
        markdown_text = f.read()
    headers_to_split_on = [("#","laptop-name")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    all_docs = markdown_splitter.split_text(markdown_text) 
    all_chunks = chunking_of_products(all_docs)
    
    url_pattern =  r"URL:\s*(https://[^\s]+)"
    create_collection_payloads()
    offset = 0
    product_offset = 0
    info = client.get_collection(collection_name=collection_name)
    count = info.points_count
    if(count != 0):
        res,_ = client.scroll(
            collection_name = collection_name,
            limit = 1,
            with_payload = True,
            with_vectors = False,
            order_by={
                "key" : "chunk_id",
                "direction" : "desc"
            }
        )
        if(res):
            product_number = res[0].payload.get("product_id")
            product_offset = product_number + 1
            last_id = res[0].id
            offset = last_id + 1
        else:
            offset = 0
            product_offset = 0
    else:
        offset = 0
        product_offset = 0

    for doc in range(len(all_chunks)):
        for idx in range(len(all_chunks[doc])):
            match = re.search(url_pattern,all_docs[doc].page_content)
            product_url = match.group(1)
            client.upsert(
                collection_name = collection_name,
                points = [
                    models.PointStruct(
                        id = offset,
                        payload = {
                            "product_id" : product_offset,
                            "chunk_id" : offset,
                            "chunk" : all_chunks[doc][idx],
                            "URL" : product_url,
                            "product_name" : all_docs[doc].metadata["laptop-name"]
                        },
                        vector = {
                            "dense" : list(dense_encoder.embed(all_chunks[doc][idx]))[0],
                            "sparse" : list(sparse_encoder.embed(all_chunks[doc][idx]))[0].as_object(),
                            "lateInteraction" : list(late_encoder.embed(all_chunks[doc][idx]))[0]
                        }
                    )
                ]
            )
            offset+=1
        product_offset+=1

def main():
    to_vectordb()

if __name__ == "__main__":
    main()