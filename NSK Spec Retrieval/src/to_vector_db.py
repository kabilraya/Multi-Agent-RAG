import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding,TextEmbedding,LateInteractionTextEmbedding
import os
import time
#GLOBAL CONFIG

collection_name = "NSK Test"
base_directory = os.path.abspath(os.path.dirname(__file__))
file_name = os.path.join(os.path.abspath(base_directory,"..","data","nsk.xlsx"))

dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
client = QdrantClient("http://localhost:6333")

website = "NSK"

def create_collection_with_payloads():
    if not client.collection_exists(collection_name = collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_encoder.embedding_size,
                    distance=models.Distance.COSINE
                ),
                "lateinteract" : models.VectorParams(
                    size=late_encoder.embedding_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)
                ),
                "dense_spec": models.VectorParams(
                    size = dense_encoder.embedding_size,
                    distance=models.Distance.COSINE
                ),
                "lateinteract_spec" : models.VectorParams(
                    size = late_encoder.embedding_size,
                    distance = models.Distance.COSINE,
                    multivector_config= models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)
                )
            },
            sparse_vectors_config={"sparse":models.SparseVectorParams(modifier = models.Modifier.IDF),
                                "sparse_spec": models.SparseVectorParams(modifier=models.Modifier.IDF)}
        )
    client.create_payload_index(
        collection_name = collection_name,
        field_name="filename",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "sub-category",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema = "keyword"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name = "URL",
        field_schema = "keyword"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "chunk_id",
        field_schema = "integer"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "subcategory_number",
        field_schema="integer"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name = "website",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "specification",
        field_schema = "keyword"
    )

def to_dataframes():
    # loading the excel table into pandas dataframes cleaning the NaN values
    part_numbers_tables = pd.read_excel(file_name)
    
    columns_to_join = part_numbers_tables.drop(columns=["URL"])
    part_numbers_tables["row-text"] = part_numbers_tables.apply(lambda r: " | ".join(f"{c}:{r[c]}" for c in columns_to_join.columns if pd.notna(r[c]) and r[c] != ""),axis = 1)
    exclude_cols = {
        "product_name","rows"
    }
    part_numbers_tables["specification"] = part_numbers_tables.apply(lambda r: " | ".join(f"{c}:{r[c]}" for c in columns_to_join.columns if c not in exclude_cols and (pd.notna(r[c] and r[c]!=""))),axis = 1)
    print(part_numbers_tables["row-text"][0])
    print(part_numbers_tables["specification"][0])
    return part_numbers_tables

def to_vectordb():
    
    
    category = "bearing lock nuts"
    
    

    create_collection_with_payloads()

    table = to_dataframes()

    part_number_offset = 0
    subcategory_offset = 0

    info = client.get_collection(collection_name = collection_name)
    count = info.points_count
    if(count!=0):
        res,_ = client.scroll(
            collection_name = collection_name,
            limit = 1,
            with_payload=True,
            with_vectors = False,
            order_by={
                "key" : "chunk_id",
                "direction" : "desc"
            }
        )
        if(res):
            part_number = res[0].payload.get("chunk_id")
            subcategory_number = res[0].payload.get("subcategory_number")
            part_number_offset = part_number + 1
            subcategory_offset = subcategory_number + 1
        else:
            part_number_offset = 0
            subcategory_offset = 0
    else:
        part_number_offset = 0
        subcategory_offset = 0
    
    for idx,row in table.iterrows():
        
        url= row['URL']
        
        

        part_info = row['row-text'] +f" | Category:{category} | Subcategory:{row['product_name']} | URL: {url} | Website: {website}"
        info_for_embedding = f"Category:{category} | Subcategory:{row['product_name']} | " + row["row-text"]
        client.upsert(
            collection_name = collection_name,
            points=[
                models.PointStruct(
                    id = part_number_offset,
                    payload = {
                        "chunk_id" : part_number_offset,
                        "subcategory_number": subcategory_offset,
                        "URL" : url,
                        "category" : category,
                        "sub-category":row["product_name"],
                        "part_info" : part_info,
                        "file_name" : file_name,
                        "part_name": row["product_name"],
                        "website" : website
                    },
                    vector={
                        "dense" : list(dense_encoder.embed(info_for_embedding))[0],
                        "sparse" : list(sparse_encoder.embed(info_for_embedding))[0].as_object(),
                        "lateinteract" : list(late_encoder.embed(info_for_embedding))[0],
                        "dense_spec" : list(dense_encoder.embed(row["specification"]))[0],
                        "sparse_spec" : list(sparse_encoder.embed(row["specification"]))[0].as_object(),
                        "lateinteract_spec" : list(late_encoder.embed(row["specification"]))[0]
                    }
                )
            ]
        )
        
        part_number_offset+=1
        
    

    
def main():
    to_vectordb()

if __name__ == "__main__":
    main()


