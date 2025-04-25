import streamlit as st
from openai import OpenAI
import time
import re

placeholderstr = "Type the page you want to visit (Chatbot, SVD Analysis, Skip-gram Model, CBOW Model)"
user_name = "Your Name"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def generate_response(prompt):
    prompt = prompt.lower()
    if "chatbot" in prompt:
        return "Okay, navigating to the Chatbot page."
    elif "svd analysis" in prompt or "svd vs word2vec" in prompt:
        return "Alright, let's go to the SVD vs. Word2Vec Analysis page."
    elif "skip-gram model" in prompt:
        return "Taking you to the Modified Skip-gram Model page."
    elif "cbow model" in prompt:
        return "Heading over to the Modified CBOW Model page."
    else:
        return "Sorry, I can't help you with that page request right now. Please type 'Chatbot', 'SVD Analysis', 'Skip-gram Model', or 'CBOW Model'."

def chatbot_page():
    st.title(f"💬 {user_name}'s Navigation Bot")
    st.write("Tell me which page you'd like to visit.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! Which page would you like to go to? You can say 'Chatbot', 'SVD Analysis', 'Skip-gram Model', or 'CBOW Model'."})
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
        if "chatbot" in prompt.lower():
            time.sleep(1)
            navigate_to_page("Chatbot") # Already on chatbot, but for consistency
        elif "svd analysis" in prompt.lower() or "svd vs word2vec" in prompt.lower():
            time.sleep(1)
            navigate_to_page("SVD vs. Word2Vec Analysis")
        elif "skip-gram model" in prompt.lower():
            time.sleep(1)
            navigate_to_page("Modified Skip-gram Model")
        elif "cbow model" in prompt.lower():
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