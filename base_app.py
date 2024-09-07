import streamlit as st
from transformers import pipeline

# Set up the Hugging Face pipeline
@st.cache_resource
def load_model():
    return pipeline(task="text-generation", model="viniljpk/distilgpt2_pmmed_finetuned_vinil", max_length=200)

pipe = load_model()

st.title("Chat with Custom Llama 2 Model")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = pipe(f"<s>[INST] {prompt} [/INST]")
            response_text = response[0]['generated_text'].split("[/INST]")[-1].strip()
            st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})