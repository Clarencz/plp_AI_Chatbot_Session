import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="mabear - Python Coding Assistant", layout="wide")

def load_txt_generation():
    text_generator = pipeline("text generation",model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator

SYSTEM_INSTRUCTIONS = (
    "you are a coding assistant that helps with python programming questions."
    " Answer as concisely as possible."
    " If you don't know the answer, just say that you don't know."
    " Provide code examples where applicable."
)
def build_conversation_prompt(chat_history, user_question):
    formatted_conversation = []
    for previous_question, previous_answer in chat_history:
        formatted_conversation.append(f"User: {previous_question}\nAssistant: {previous_answer}\n")
    formatted_conversation.append(f"User: {user_question}\nAssistant:")
    return SYSTEM_INSTRUCTIONS + "\n" + "\n".join(formatted_conversation)

st.title("mabeya - Python Coding Assistant")
st.caption("Ask me anything about Python programming!")

#sidebar
with st.sidebar:
    st.header("Model Controls")
    max_new_tokens = st.slider("Max New Tokens", min_value=10, max_value=100, value=150, step=10)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    if st.button("clear chat history"):
        st.session_state.chat_history = ["start new chat"]
        st.success("Chat history cleared.")
    st.markdown(
        """
        mabear is a coding assistant powered by GPT-2, designed to help you with Python programming questions.
        Simply ask your question, and mabear will provide concise answers and code examples where applicable.
        """
    )
    st.markdown("---")
    st.markdown("Developed by Your Mabear")

#initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#display chat history
for user_message,ai_reply in st.session_state.chat_history:
    st.chat_message("user").markdown(user_message)
    st.chat_message("assistant").markdown(ai_reply)

#user input
user_input = st.chat_input("Ask me anything about Python programming!")
if user_input:
    st.chat_message("user").markdown(user_input)

    with st.spinner("thinking..."):
        text_generator = load_txt_generation()
        prompt = build_conversation_prompt(st.session_state.chat_history, user_input)

        generation_output = text_generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=text_generator.tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=text_generator.tokenizer.eos_token_id,
        )[0]["generated_text"]

        #xtract the assistant's reply
        generated_answer = generation_output.split("Answer:")[-1].strip()
        if "Question:" in generation_output:
            generated_answer = generation_output.split("Question:")[0].strip()

#displaying and storing chatbot response
st.chat_message("assistant").markdown(generated_answer)
st.session_state.chat_history.append((user_input, generated_answer))