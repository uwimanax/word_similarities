import re
from pathlib import Path
from gensim.models import Word2Vec

sentences = [
    "the king is a powerful man",
    "the queen is a powerful woman",
    "the king and queen live in a palace",
    "the man and woman are people",
    "the boy and girl are young people",
    "a car is a vehicle",
    "a truck is a vehicle",
    "a bus is a vehicle",
    "a car and truck are transportation",
    "a bus is used for transportation",
    "an apple is a fruit",
    "a banana is a fruit",
    "an orange is a fruit",
    "an apple and banana are healthy food",
    "fruit is healthy food",
    "a computer is a technology device",
    "a laptop is a computer",
    "a smartphone is a technology device",
    "a computer and laptop are electronic devices",
    "a smartphone is an electronic device",
    "happy people feel joyful",
    "happy people feel positive",
    "joyful people feel happy",
    "sad people feel unhappy",
    "happy and joyful are positive emotions"
]

tokenized_sentences = [
    re.sub(r"[^a-z\\s]", "", sentence.lower()).split()
    for sentence in sentences
]

output_path = Path("model/word2vec.model")
output_path.parent.mkdir(parents=True, exist_ok=True)

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100,
    seed=42
)

model.save(str(output_path))

print("Model trained and saved to:", output_path)
print("Vocabulary size:", len(model.wv))
