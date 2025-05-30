import faiss

# Load the FAISS index directly
index = faiss.read_index("rules_index/index.faiss")

# Check vector dimension
dimension = index.d
print(f"Vector dimension: {dimension}")

# Check number of stored vectors
print(f"Number of vectors: {index.ntotal}")

