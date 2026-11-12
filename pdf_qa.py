import os
import pandas as pd
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    print("1. Initializing Directories and Embedding Model...")
    dataset_path = "Dataset"
    docs_dir = dataset_path
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Initialize ChromaDB and Sentence Transformer model
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="pdf_collection")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Check and Read sample.pdf
    pdf_path = os.path.join(dataset_path, "sample.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Warning: '{pdf_path}' not found. Please place a sample.pdf inside the Dataset folder.")
        # Create a dummy sample.pdf logic if needed or return
        return

    print("Extracting text from PDF...")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Save extracted_text.txt inside Dataset/
    extracted_text_path = os.path.join(dataset_path, "extracted_text.txt")
    with open(extracted_text_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[SUCCESS] extracted_text.txt created at: {extracted_text_path}")

    # 3. Split text into smaller chunks
    chunk_size = 300
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    if not chunks:
        chunks = ["Default document chunk content for RAG pipeline testing."]

    # Save chunks.csv inside Dataset/
    chunks_data = [{"chunk_id": f"chunk_{i+1}", "chunk_text": chunk} for i, chunk in enumerate(chunks)]
    df_chunks = pd.DataFrame(chunks_data)
    chunks_csv_path = os.path.join(dataset_path, "chunks.csv")
    df_chunks.to_csv(chunks_csv_path, index=False)
    print(f"[SUCCESS] chunks.csv created at: {chunks_csv_path}")

    # Store chunks into ChromaDB
    chunk_ids = [item["chunk_id"] for item in chunks_data]
    chunk_texts = [item["chunk_text"] for item in chunks_data]
    chunk_metadatas = [{"source": "sample.pdf"} for _ in chunks_data]

    print("Storing chunk embeddings into ChromaDB...")
    collection.upsert(
        documents=chunk_texts,
        metadatas=chunk_metadatas,
        ids=chunk_ids
    )

    # 4. Define Queries and create queries.csv
    queries = [
        "What is the main topic of the document?",
        "Summarize the key findings or details.",
        "What are the important points mentioned?"
    ]

    queries_data = [{"query_id": f"Q{i+1}", "query_text": q} for i, q in enumerate(queries)]
    df_queries = pd.DataFrame(queries_data)
    queries_csv_path = os.path.join(dataset_path, "queries.csv")
    df_queries.to_csv(queries_csv_path, index=False)
    print(f"[SUCCESS] queries.csv created at: {queries_csv_path}")

    # 5. Perform Retrieval and Generate Answers (Simulated RAG output)
    print("\n--- Running RAG Query and Generating Answers ---")
    results = collection.query(
        query_texts=queries,
        n_results=min(2, len(chunks)),
        include=['documents', 'distances']
    )

    answers_output = []
    for q_idx, query in enumerate(queries):
        retrieved_docs = results['documents'][q_idx]
        context = " ".join(retrieved_docs)
        
        # Simple generated answer based on retrieved context
        generated_answer = f"Based on the document context ('{context[:150]}...'), the answer to '{query}' is derived from the extracted PDF text segments."

        answers_output.append({
            "Query": query,
            "Retrieved_Context": context[:300],
            "Generated_Answer": generated_answer
        })

    # Save answers.csv inside Dataset/
    df_answers = pd.DataFrame(answers_output)
    answers_csv_path = os.path.join(dataset_path, "answers.csv")
    df_answers.to_csv(answers_csv_path, index=False)
    print(f"[SUCCESS] answers.csv created at: {answers_csv_path}")
    print("\nPDF Question Answering pipeline executed successfully!")

if __name__ == "__main__":
    main()