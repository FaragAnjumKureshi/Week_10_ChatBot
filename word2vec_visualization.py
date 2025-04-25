import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt  # While imported, Plotly is used for display

def word2vec_visualization_page():
    st.title("Word2Vec Embedding Visualization")
    st.write("Here are the 2D and 3D visualizations of word embeddings from a Word2Vec model trained on the following sentences.")

    # Sample sentences
    sentences = [
        "The fluffy golden retriever barked happily at the mailman who walked down the sunny street.",
        "A sleek black cat silently stalked a small mouse hiding beneath the old wooden table.",
        "Eating a ripe red apple after a long run is incredibly refreshing and healthy.",
        "Sipping a cold glass of freshly squeezed orange juice on a hot day is quite invigorating.",
        "Bananas and grapes, along with other fruits, are essential for a balanced diet and provide vital nutrients.",
        "The diligent king carefully considered the complex laws governing his vast and prosperous kingdom.",
        "With wisdom and grace, the benevolent queen addressed the concerns of the people throughout her peaceful land.",
        "The programmer skillfully coded a new algorithm to efficiently process large datasets of information.",
        "Artificial intelligence is rapidly evolving, enabling machines to learn and perform tasks that once required human intellect.",
        "Natural language processing techniques allow computers to understand and interpret human language in various forms."
    ]

    # Display the 10 sentences with an expander
    with st.expander("Show/Hide Training Sentences"):
        for i, sentence in enumerate(sentences):
            st.write(f"Sentence {i+1}: {sentence}")

    # Preprocess the sentences
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

    # Train a Word2Vec model
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, seed=42)

    # Get the word vectors
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

    # Reduce the dimensions to 3D using PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(word_vectors)

    # Assign a color to each word based on its sentence
    color_map = {i: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i in range(len(sentences))}
    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_map[i])
                break

    # 2D Visualization
    st.subheader("2D Visualization of Word Embeddings")
    pca_2d = PCA(n_components=2)
    reduced_vectors_2d = pca_2d.fit_transform(word_vectors)

    fig_2d = go.Figure(data=[go.Scatter(
        x=reduced_vectors_2d[:, 0],
        y=reduced_vectors_2d[:, 1],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=8),
        hovertemplate="Word: %{text}<br>Sentence Color: %{marker.color}"
    )])
    fig_2d.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        width=800,
        height=800
    )
    st.plotly_chart(fig_2d)

    # 3D Visualization
    st.subheader("3D Visualization of Word Embeddings")
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=5),
        hovertemplate="Word: %{text}<br>Sentence Color: %{marker.color}"
    )])
    fig_3d.update_layout(
        scene=dict(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2", zaxis_title="Principal Component 3"),
        width=800,
        height=800
    )
    st.plotly_chart(fig_3d)

if __name__ == "__main__":
    word2vec_visualization_page()