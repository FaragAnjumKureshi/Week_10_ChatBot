import streamlit as st
import re
from openai import OpenAI
import time
import re
from word2vec_visualization import word2vec_visualization_page
from modified_skipgram import modified_skipgram_page
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

placeholderstr = "Type a sentence about which page you'd like to visit"
user_name = "Farag"
user_image = "https://www.w3schools.com/howto/img_avatar.png"
page_names = ["Chatbot", "Word2Vec Visualization", "Modified Skip-gram Model", "Modified CBOW Model"]


def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def generate_response(prompt):
    prompt_lower = prompt.lower()

    if re.search(r"(chatbot|chat\s*bot|chat\s+interface)", prompt_lower):
        return "Okay, navigating to the Chatbot page."
    elif re.search(r"(word2vec\s+visual(s)?|word\s+embeddings\s+visual(s)?|word2vec\s+plot(s)?|word\s+embedding\s+plot(s)?)", prompt_lower):
        return "Alright, let's go to the Word2Vec Visualization page."
    elif re.search(r"(modified\s+skip\s*-?gram|skip\s*-?gram\s+modified|altered\s+skip\s*-?gram)", prompt_lower):
        return "Taking you to the Modified Skip-gram Model page."
    elif re.search(r"(modified\s+cbow|cbow\s+modified|altered\s+cbow)", prompt_lower):
        return "Heading over to the Modified CBOW Model page."
    else:
        return "Sorry, I'm not sure which page you're asking for. Please try being more specific."

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
        print(f"Navigating to: {page_name}")

    def chat(prompt: str):
        st_c_chat.chat_message("user", avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

        prompt_lower = prompt.lower()
        if re.search(r"(chatbot|chat\s*bot|chat\s+interface)", prompt_lower):
            time.sleep(1)
            navigate_to_page("Chatbot")
        elif re.search(r"(word2vec\s+visual(s)?|word\s+embeddings\s+visual(s)?|word2vec\s+plot(s)?|word\s+embedding\s+plot(s)?)", prompt_lower):
            time.sleep(1)
            navigate_to_page("Word2Vec Visualization")
        elif re.search(r"(modified\s+skip\s*-?gram|skip\s*-?gram\s+modified|altered\s+skip\s*-?gram)", prompt_lower):
            time.sleep(1)
            navigate_to_page("Modified Skip-gram Model")
        elif re.search(r"(modified\s+cbow|cbow\s+modified|altered\s+cbow)", prompt_lower):
            time.sleep(1)
            navigate_to_page("Modified CBOW Model")

    if prompt := st.chat_input(placeholder=placeholderstr, key="navigation_bot"):
        chat(prompt)


def modified_cbow_page():
    st.title("Modified CBOW Model")
    st.write("Results of CBOW model with modified parameters...")
    new_sentence = st.text_input("Enter a new sentence to try with the CBOW model:")
    if new_sentence:
        output = f"You entered: '{new_sentence}'. Here's some output from the CBOW model (to be implemented)."
        st.write(output)

if 'current_page' not in st.session_state or st.session_state['current_page'] == "Chatbot":
    chatbot_page()
elif st.session_state['current_page'] == "Word2Vec Visualization":
    word2vec_visualization_page()
elif st.session_state['current_page'] == "Modified Skip-gram Model":
    modified_skipgram_page()
elif st.session_state['current_page'] == "Modified CBOW Model":
    modified_cbow_page()

#st.write(st.session_state.get('current_page'))