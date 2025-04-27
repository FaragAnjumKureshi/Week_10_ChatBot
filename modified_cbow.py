import streamlit as st
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

def modified_cbow_page():
    st.title("Modified CBOW Model Results")
    st.write("Here are the results of training a CBOW Word2Vec model on the sample sentences (with stopword removal), along with a comparison to a Skip-gram model.")

    # Sample sentences
    sentences = [
        "The fluffy golden retriever dog barked happily at the mailman who walked down the sunny street.",
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

    # --- CBOW Model ---
    st.subheader("CBOW Model (with Stopword Removal)")
    vector_size = 23
    window_size = 5
    min_count = 1
    workers = 4
    sg_flag_cbow = 0

    tokenized_sentences_cbow = [[word for word in simple_preprocess(remove_stopwords(sentence))] for sentence in sentences]
    cbow_model = Word2Vec(tokenized_sentences_cbow, vector_size=vector_size, window=window_size, min_count=min_count, workers=workers, sg=sg_flag_cbow, seed=42) # Added seed

    st.write(f"Vector for 'cat' (CBOW):")
    st.code(cbow_model.wv['cat'].tolist())

    similar_words_cbow = cbow_model.wv.most_similar('cat')
    st.write(f"Most similar words to 'cat' (CBOW):")
    st.dataframe(similar_words_cbow)

    st.markdown("---")

    # --- Comparison with Skip-gram ---
    st.subheader("Comparison with Skip-gram Model (with Stopword Removal)")
    sg_flag_ngram = 1

    skip_gram_model = Word2Vec(tokenized_sentences_cbow, vector_size=vector_size, window=window_size, min_count=min_count, workers=workers, sg=sg_flag_ngram, seed=42) # Added seed

    st.write(f"SKIP-gram vector for 'cat':")
    st.code(skip_gram_model.wv['cat'].tolist())

    st.write(f"CBOW vector for 'cat':")
    st.code(cbow_model.wv['cat'].tolist())

    similarity_cat_dog_skipgram = skip_gram_model.wv.similarity('cat', 'dog')
    st.write(f"Similarity of SKIP-gram 'cat' to 'dog': {similarity_cat_dog_skipgram:.4f}")

    similarity_cat_dog_cbow = cbow_model.wv.similarity('cat', 'dog')
    st.write(f"Similarity of CBOW 'cat' to 'dog': {similarity_cat_dog_cbow:.4f}")

if __name__ == "__main__":
    modified_cbow_page()