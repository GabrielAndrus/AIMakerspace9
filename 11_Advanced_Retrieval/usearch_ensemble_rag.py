import numpy as np
from sentence_transformers import SentenceTransformer
from usearch.index import BatchMatches, Index, Matches, MetricKind, search

model = SentenceTransformer("all-MiniLM-l6-v2")
sentences = ["How can I do a DIY roof?", "DIY rooves are made with shingles."]
embeddings = model.encode(sentences)
print(embeddings.shape)
print(embeddings.dtype)

index = Index(
    ndim=384,
    metric="cos",
    dtype="float32",
    multi=True,
)
vector = np.array(embeddings[1])  # Can be a matrix for batch operations
key_vector = np.array(embeddings[0])
index.add(0, vector)  # Add one or many vectors in parallel
matches = index.search(vector, 1)  # Find 10 nearest neighbors

best_match = matches.keys[0]
print(best_match)
