import streamlit as st
from openai import OpenAI
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

placeholderstr = "Type a sentence about which page you'd like to visit"
user_name = "Your Name"
user_image = "https://www.w3schools.com/howto/img_avatar.png"
page_names = ["Chatbot", "SVD vs. Word2Vec Analysis", "Modified Skip-gram Model", "Modified CBOW Model"]
navigation_prompts = [
    "Take me to the chatbot page.",
    "I want to see the SVD and Word2Vec analysis.",
    "Show me the modified Skip-gram model results.",
    "Navigate to the CBOW model section.",
    "Can you go to the chatbot?",
    "Let's look at the comparison of SVD and Word2Vec.",
    "I'm interested in the Skip-gram model with changes.",
    "Show me the results for the altered CBOW.",
    "Go to the page with the chat interface.",
    "I'd like to see the difference between SVD and Word2Vec embeddings.",
    "Present the findings for the adjusted Skip-gram.",
    "Take me to the modified CBOW results.",
    "Chatbot please.",
    "SVD Word2Vec comparison.",
    "Modified Skipgram.",
    "CBOW Model.",
]

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def generate_response(prompt):
    vectorizer = TfidfVectorizer()
    all_prompts = navigation_prompts + [prompt]
    tfidf_matrix = vectorizer.fit_transform(all_prompts)
    user_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix[:-1])[0]
    most_similar_index = np.argmax(similarity_scores)

    if similarity_scores[most_similar_index] > 0.5:  # You can adjust this threshold
        if most_similar_index < len(navigation_prompts) // len(page_names) * 1:
            return f"Okay, it sounds like you want to go to the Chatbot page."
        elif most_similar_index < len(navigation_prompts) // len(page_names) * 2:
            return f"Alright, navigating to the SVD vs. Word2Vec Analysis page."
        elif most_similar_index < len(navigation_prompts) // len(page_names) * 3:
            return f"Taking you to the Modified Skip-gram Model page."
        elif most_similar_index < len(navigation_prompts):
            return f"Heading over to the Modified CBOW Model page."
    else:
        return "Sorry, I'm not sure which page you're asking for. Please try rephrasing."

def chatbot_page():
    st.title(f"💬 {user_name}'s Navigation Bot")
    st.write("Tell me which page you'd like to visit.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! Which page are you interested in? Just type a sentence describing what you'd like to see."})
        st.session_state['current_page'] = "Chatbot" # Initial page

    st_c_chat = st.container(border=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st_c_chat.chat_message(msg["role"], avatar=user_image).markdown((msg["content"]))
        elif msg["role"] == "assistant":
            st_c_chat.chat_message(msg["role"]).write_stream(stream_data(msg["content"]))

    def navigate_to_page(page_name):
        st.session_state['current_page'] = page_name

    def chat(prompt: str):
        st_c_chat.chat_message("user", avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Delayed navigation based on the response
        if "chatbot" in response.lower():
            time.sleep(1)
            navigate_to_page("Chatbot")
        elif "svd vs word2vec analysis" in response.lower():
            time.sleep(1)
            navigate_to_page("SVD vs. Word2Vec Analysis")
        elif "skip-gram model" in response.lower():
            time.sleep(1)
            navigate_to_page("Modified Skip-gram Model")
        elif "cbow model" in response.lower():
            time.sleep(1)
            navigate_to_page("Modified CBOW Model")

    if prompt := st.chat_input(placeholder=placeholderstr, key="navigation_bot"):
        chat(prompt)

def svd_word2vec_page():
    st.title("SVD vs. Word2Vec Analysis")
    st.write("Here are the 2D and 3D visualizations of word embeddings...")
    # Add your plotting code here
    st.write("Findings on the differences between SVD and Word2Vec...")
    # Add your analysis here

def modified_skipgram_page():
    st.title("Modified Skip-gram Model")
    st.write("Results of Skip-gram model with modified parameters...")
    # Add your Skip-gram model training and result display here
    new_sentence = st.text_input("Enter a new sentence to try with the Skip-gram model:")
    if new_sentence:
        output = f"You entered: '{new_sentence}'. Here's some output from the Skip-gram model (to be implemented)."
        st.write(output)

def modified_cbow_page():
    st.title("Modified CBOW Model")
    st.write("Results of CBOW model with modified parameters...")
    # Add your CBOW model training and result display here
    new_sentence = st.text_input("Enter a new sentence to try with the CBOW model:")
    if new_sentence:
        output = f"You entered: '{new_sentence}'. Here's some output from the CBOW model (to be implemented)."
        st.write(output)

# No sidebar navigation anymore
if 'current_page' not in st.session_state or st.session_state['current_page'] == "Chatbot":
    chatbot_page()
elif st.session_state['current_page'] == "SVD vs. Word2Vec Analysis":
    svd_word2vec_page()
elif st.session_state['current_page'] == "Modified Skip-gram Model":
    modified_skipgram_page()
elif st.session_state['current_page'] == "Modified CBOW Model":
    modified_cbow_page()