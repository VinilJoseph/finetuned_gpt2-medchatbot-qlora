import streamlit as st
import re
from transformers import pipeline

# Set a professional theme for the app
st.set_page_config(
    page_title="Medical Chat Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up the Hugging Face pipeline
@st.cache_resource
def load_model():
    return pipeline(task="text-generation", model="viniljpk/distilgpt2_pmmed_finetuned_vinil", max_length=100)

pipe = load_model()

def format_response(text):
    # Extract the actual response (after the last [/INST] tag)
    response_parts = text.split("[/INST]")
    if len(response_parts) > 1:
        text = response_parts[-1].strip()
    
    # Remove repeated "</s>" tags
    text = re.sub(r'</s>\s*</s>', '</s>', text)
    
    # Remove any text after the last "</s>" tag
    text = text.split("</s>")[0].strip()
    
    # Replace "<s>" and "</s>" tags with newlines
    text = text.replace("<s>", "\n").replace("</s>", "\n")
    
    # Remove extra newlines
    text = re.sub(r'\n+', '\n', text).strip()
    
    return text

# Sidebar with tips
st.sidebar.title("Medical Chat Assistant 🩺")
st.sidebar.markdown("""
## Tips for using the Medical Chat Assistant:

1. **Be specific**: Provide as much relevant information as possible about your medical query.
2. **Ask for clarification**: If you don't understand something, ask the AI to explain it in simpler terms.
3. **Request examples**: Ask for examples to better understand medical concepts or procedures.
4. **Follow-up questions**: Feel free to ask follow-up questions for more detailed information.
5. **Structured responses**: If you want a structured answer, ask for bullet points or numbered lists.
6. **Symptoms description**: When describing symptoms, be as detailed as possible about their nature, duration, and severity.
7. **Medical history**: If relevant, mention any pertinent medical history or current medications.


Use this link to see the dataset the model works well : https://huggingface.co/datasets/viniljpk/pubmedQA_v1
                    

**Disclaimer**: This AI assistant provides general medical information and should not replace professional medical advice. Always consult with a healthcare professional for personalized medical guidance.
""")

# Main title and introduction
st.title("Chat with the Medical Chat Assistant")
st.subheader("Your personalized AI assistant for medical queries")
st.write(
    "Welcome to the **Medical Chat Assistant**! This tool can help you with general medical information and provide assistance on a wide range of medical queries. Remember, this is for informational purposes only and does not replace professional medical advice."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask your medical question here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(f"**You:** {prompt}")
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("The assistant is thinking..."):
            response = pipe(f"<s>[INST] {prompt} [/INST]")
            response_text = format_response(response[0]['generated_text'])
            st.markdown(f"**Assistant:** {response_text}")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
