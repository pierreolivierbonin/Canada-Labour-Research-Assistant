# Reference doc and inspiration: https://huggingface.co/blog/train-sparse-encoder#contrastive-sparse-representation-csr 

from sentence_transformers import SparseEncoder # requires Sentence-Transformers>=5.0.0

# Download from the 🤗 Hub
model = SparseEncoder("naver/splade-v3")

# Run inference
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# (3, 30522)

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[   32.4323,     5.8528,     0.0258],
#         [    5.8528,    26.6649,     0.0302],
#         [    0.0258,     0.0302,    24.0839]])

# Let's decode our embeddings to be able to interpret them
decoded = model.decode(embeddings, top_k=10)
for d, sentence in zip(decoded, sentences):
    print(f"Sentence: {sentence}")
    print(f"Decoded:")
    for i in d:
        print(i)
    print()

# Let's also compute the intersection/overlap of the first two embeddings
intersection_embedding = model.intersection(embeddings[0], embeddings[1])
decoded_intersection = model.decode(intersection_embedding)
print(decoded_intersection)

