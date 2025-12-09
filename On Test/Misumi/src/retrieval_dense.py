from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import os
import openpyxl

current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.abspath(os.path.join(current_dir,"..","test_logs","payload_score_logs.xlsx"))
collection_name = "Dense Only Misumi"
client = QdrantClient(url="http://localhost:6333")
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

def retrieval(query):
    """
    Retrieves data points from the Qdrant collection based on a query.

    Args:
        query (str): The search query.

    Returns:
        list: A list of retrieved points.
    """
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

def retrieve_part_numbers(query, sheet):
    """
    Retrieves part numbers and writes them to an Excel sheet.

    Args:
        query (str): The search query.
        sheet (openpyxl.worksheet.worksheet.Worksheet): The Excel sheet to write to.
    """
    points = retrieval(query=query)
    for i, point in enumerate(points):
        payload = point.payload.get("part_info")
        score = point.score
        rank = i + 1
        sheet.append([query, payload, score, rank])

def main():
    """
    Main function to execute the part number retrieval and writing process.
    """
    
    if os.path.exists(file_name):
        
        workbook = openpyxl.load_workbook(file_name)
        sheet = workbook.active
    else:
        
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Query", "Part Number", "Similarity Score", "Rank"])

    query = input("Enter the Part Number: ")
    retrieve_part_numbers(query, sheet)

    
    workbook.save(file_name)
    print(f"Data saved to {file_name}")

if __name__ == "__main__":
    main()