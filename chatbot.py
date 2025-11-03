import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="mabear - Python Coding Assistant", layout="wide")

@st.cache_resource
def load_txt_generation():
    text_generator = pipeline("text-generation", model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator

SYSTEM_INSTRUCTIONS = (
    "You are a coding assistant that helps with Python programming questions. "
    "Answer as concisely as possible. "
    "If you don't know the answer, say you don't know. "
    "Provide code examples where applicable."
)

def build_conversation_prompt(chat_history, user_question):
    formatted_conversation = []
    for q, a in chat_history:
        formatted_conversation.append(f"User: {q}\nAssistant: {a}\n")
    formatted_conversation.append(f"User: {user_question}\nAssistant:")
    return SYSTEM_INSTRUCTIONS + "\n" + "\n".join(formatted_conversation)

st.title("mabear - Python Coding Assistant")
st.caption("Ask me anything about Python programming!")

# Sidebar
with st.sidebar:
    st.header("Model Controls")
    max_new_tokens = st.slider("Max New Tokens", 10, 300, 150, 10)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
    st.markdown("---")
    st.markdown("Developed by Your Mabear")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for user_message, ai_reply in st.session_state.chat_history:
    st.chat_message("user").markdown(user_message)
    st.chat_message("assistant").markdown(ai_reply)

# Handle user input
user_input = st.chat_input("Ask me anything about Python programming!")
if user_input:
    st.chat_message("user").markdown(user_input)
    with st.spinner("Thinking..."):
        text_generator = load_txt_generation()
        prompt = build_conversation_prompt(st.session_state.chat_history, user_input)

        generation_output = text_generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=text_generator.tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )[0]["generated_text"]

        generated_answer = generation_output[len(prompt):].strip()

    st.chat_message("assistant").markdown(generated_answer)
    st.session_state.chat_history.append((user_input, generated_answer))
