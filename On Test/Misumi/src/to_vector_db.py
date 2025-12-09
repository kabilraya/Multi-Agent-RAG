import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding,TextEmbedding,LateInteractionTextEmbedding
import os
import time
#GLOBAL CONFIG

collection_name = "Dense Only Misumi"
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.abspath(os.path.join(current_dir,"..","data","washer_for_rolling_bearings_retaining_and_tightening_washer_series_awl.xlsx"))
file_name = file_name
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
base = os.path.basename(file_name)

client = QdrantClient("http://localhost:6333")
main_domain= "https://vn.misumi-ec.com"
website = "Misumi"

def create_collection_with_payloads():
    if not client.collection_exists(collection_name = collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_encoder.embedding_size,
                    distance=models.Distance.COSINE
                ),
            
            },
        
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

def to_dataframes():
    # loading the excel table into pandas dataframes cleaning the NaN values
    part_numbers_tables = pd.read_excel(file_name)
    part_numbers_tables = part_numbers_tables.replace("-","none")
    columns_to_join = part_numbers_tables.drop(columns=["Part Number URL"])
    part_numbers_tables["row-text"] = part_numbers_tables.apply(lambda r: " | ".join(f"{c}:{r[c]}" for c in columns_to_join.columns),axis = 1)
    exclude_cols = {
        "Price", "Volumn Discount", "Minimum order Qty",
        "Days to Ship", "Seal", "Part Number URL","Volume Discount","Part Number URL","row-text"
    }
    part_numbers_tables["embedding-text"] = part_numbers_tables.apply(
    lambda r: " | ".join(
        f"{c}:{r[c]}"
        for c in part_numbers_tables.columns
        if c not in exclude_cols
    ),
    axis=1
    )
    print(part_numbers_tables["embedding-text"][0])
    print(part_numbers_tables["row-text"][0])
    return part_numbers_tables

def to_vectordb():

    name,ext = os.path.splitext(base)
    file_parts = name.split('_')
    category = "bearing lock nuts"
    subcategory = " ".join(file_parts[0:])
    

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
    row_count = len(table)
    row_time_total = 0
    for idx,row in table.iterrows():
        row_start = time.perf_counter()
        href = row['Part Number URL']
        
        url = main_domain + href

        part_info = row['row-text'] +f" | Category:{category} | Subcategory:{subcategory} | URL: {url} | Website: {website}"
        info_for_embedding = f"Category:{category} | Subcategory:{subcategory} | " + row["embedding-text"]
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
                        "sub-category":subcategory,
                        "part_info" : part_info,
                        "file_name" : base,
                        "part_name": row["Part Number Name"],
                        "website" : website
                    },
                    vector={
                        "dense" : list(dense_encoder.embed(info_for_embedding))[0],
                    
                    }
                )
            ]
        )
        
        part_number_offset+=1
        
    

    
def main():
    to_vectordb()

if __name__ == "__main__":
    main()


