import streamlit as st
from openai import OpenAI
import time
import re
from word2vec_visualization import word2vec_visualization_page # Import the function

placeholderstr = "Type a sentence about which page you'd like to visit"
user_name = "Farag"
user_image = "https://www.w3schools.com/howto/img_avatar.png"
page_names = ["Chatbot", "Word2Vec Visualization", "Modified Skip-gram Model", "Modified CBOW Model"]
navigation_prompts = [
    "Take me to the chatbot page.",
    "I want to see the Word2Vec visualization.",
    "Show me the modified Skip-gram model results.",
    "Navigate to the CBOW model section.",
    "Can you go to the chatbot?",
    "Let's look at the Word2Vec embeddings.",
    "I'm interested in the Skip-gram model with changes.",
    "Show me the results for the altered CBOW.",
    "Go to the page with the chat interface.",
    "I'd like to see the Word2Vec visualization.",
    "Present the findings for the adjusted Skip-gram.",
    "Take me to the modified CBOW results.",
    "Chatbot please.",
    "Word2Vec visualization.",
    "Modified Skipgram.",
    "CBOW Model.",
]

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def generate_response(prompt):
    prompt = prompt.lower()
    if "chatbot" in prompt:
        return "Okay, navigating to the Chatbot page."
    elif "word2vec" in prompt and "visual" in prompt:
        return "Alright, let's go to the Word2Vec Visualization page."
    elif "skip-gram" in prompt:
        return "Taking you to the Modified Skip-gram Model page."
    elif "cbow" in prompt:
        return "Heading over to the Modified CBOW Model page."
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
        elif "word2vec" in response.lower() and "visual" in response.lower():
            time.sleep(1)
            navigate_to_page("Word2Vec Visualization")
        elif "skip-gram" in response.lower():
            time.sleep(1)
            navigate_to_page("Modified Skip-gram Model")
        elif "cbow" in response.lower():
            time.sleep(1)
            navigate_to_page("Modified CBOW Model")

    if prompt := st.chat_input(placeholder=placeholderstr, key="navigation_bot"):
        chat(prompt)

def modified_skipgram_page():
    st.title("Modified Skip-gram Model")
    st.write("Results of Skip-gram model with modified parameters...")
    new_sentence = st.text_input("Enter a new sentence to try with the Skip-gram model:")
    if new_sentence:
        output = f"You entered: '{new_sentence}'. Here's some output from the Skip-gram model (to be implemented)."
        st.write(output)

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