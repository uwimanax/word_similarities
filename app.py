import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Word Similarity System",
    page_icon="🔤",
    layout="wide"
)

st.title("🔤 Word Similarity System")
st.subheader("Explore Relationships Between Words Using Word2Vec")
st.write(
    "Enter a word to find words with similar vector representations."
)

MODEL_PATH = Path(__file__).parent / "model" / "word2vec.model"

@st.cache_resource
def load_word2vec_model():
    return Word2Vec.load(str(MODEL_PATH))

model = load_word2vec_model()

st.sidebar.header("Settings")
top_n = st.sidebar.slider(
    "Number of similar words",
    min_value=3,
    max_value=10,
    value=5
)

word = st.text_input(
    "Enter a word:",
    placeholder="Example: car"
).lower().strip()

if word:
    if word not in model.wv:
        st.error(f"'{word}' is not in the vocabulary.")
        st.info(
            "Try: car, vehicle, truck, king, queen, apple, "
            "banana, computer, laptop, happy, or joyful."
        )
    else:
        results = model.wv.most_similar(word, topn=top_n)
        results_df = pd.DataFrame(
            results,
            columns=["Word", "Similarity"]
        )

        st.success(f"Found similar words for '{word}'.")

        st.subheader(f"Similar Words to '{word}'")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Similarity Scores")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df["Word"], results_df["Similarity"])
        ax.set_xlabel("Words")
        ax.set_ylabel("Cosine similarity")
        ax.set_title(f"Words Similar to '{word}'")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Word Vector")
        vector = model.wv[word]
        st.write(f"'{word}' is represented by {len(vector)} numerical values.")
        st.write(vector)

st.subheader("2D Word Embedding Visualization")

visual_words = [
    "car", "vehicle", "truck", "bus", "transportation",
    "king", "queen", "man", "woman",
    "apple", "banana", "fruit",
    "computer", "laptop", "smartphone",
    "happy", "joyful"
]
visual_words = [w for w in visual_words if w in model.wv]

vectors = np.array([model.wv[w] for w in visual_words])
vectors_2d = PCA(n_components=2).fit_transform(vectors)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

for i, w in enumerate(visual_words):
    ax2.annotate(w, (vectors_2d[i, 0], vectors_2d[i, 1]))

ax2.set_xlabel("PCA Dimension 1")
ax2.set_ylabel("PCA Dimension 2")
ax2.set_title("Word2Vec Word Relationships")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")
st.write("Built with Python, Word2Vec, Gensim, Streamlit, PCA and Matplotlib.")
