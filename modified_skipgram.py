import streamlit as st
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

def modified_skipgram_page():
    st.title("Modified Skip-gram Model Results")
    st.write("Here are the results of training a Skip-gram Word2Vec model on the sample sentences, both with and without stopword removal.")

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
    

    # --- Skip-gram Model WITHOUT Stopword Removal ---
    st.subheader("Skip-gram Model (Without Stopword Removal)")
    vector_size_no_stopwords = 21
    window_size = 6
    min_count = 1
    workers = 4
    sg_flag = 1

    tokenized_sentences_no_stopwords = [simple_preprocess(sentence) for sentence in sentences]
    model_no_stopwords = Word2Vec(tokenized_sentences_no_stopwords, vector_size=vector_size_no_stopwords, window=window_size, min_count=min_count, workers=workers, sg=sg_flag, seed=42) # Added seed for reproducibility

    st.write(f"Vector for 'cat' (without stopwords):")
    st.code(model_no_stopwords.wv['cat'].tolist()) # Display as a list for better readability

    similar_words_no_stopwords = model_no_stopwords.wv.most_similar('cat')
    st.write(f"Most similar words to 'cat' (without stopwords):")
    st.dataframe(similar_words_no_stopwords) # Display as a table

    st.markdown("---") # Separator

    # --- Skip-gram Model WITH Stopword Removal ---
    st.subheader("Skip-gram Model (With Stopword Removal)")
    vector_size_with_stopwords = 12
    window_size = 6
    min_count = 1
    workers = 4
    sg_flag = 1

    tokenized_sentences_with_stopwords = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences]
    model_with_stopwords = Word2Vec(tokenized_sentences_with_stopwords, vector_size=vector_size_with_stopwords, window=window_size, min_count=min_count, workers=workers, sg=sg_flag, seed=42) # Added seed for reproducibility

    st.write(f"Vector for 'cat' (with stopwords removed):")
    st.code(model_with_stopwords.wv['cat'].tolist()) # Display as a list

    similar_words_with_stopwords = model_with_stopwords.wv.most_similar('cat')
    st.write(f"Most similar words to 'cat' (with stopwords removed):")
    st.dataframe(similar_words_with_stopwords) # Display as a table

    st.markdown("---") # Another separator

    st.subheader("Try with a New Sentence")
    new_sentence = st.text_input("Enter a new sentence to see tokenization (with stopword removal):")
    if new_sentence:
        processed_sentence = simple_preprocess(remove_stopwords(new_sentence))
        st.write("Processed sentence (with stopwords removed):")
        st.code(processed_sentence)

if __name__ == "__main__":
    modified_skipgram_page()