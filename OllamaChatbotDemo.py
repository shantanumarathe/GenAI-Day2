# streamlit + langchain + ollama (LLM - gemma2:2b)

import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Step 1 - Create Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond clearly to the questions asked."),
        ("user", "Question: {question}")
    ]
)

# Step 2 - Streamlit App UI
st.title("LangChain Demo with Gemma Model (Ollama)")

input_txt = st.text_input("What question do you have in your mind?")

# Step 3 - Load Ollama Model
llm = Ollama(model="gemma2:2b")

# Convert model output to string
output_parser = StrOutputParser()

# Create pipeline
chain = prompt | llm | output_parser

# Step 4 - Run when user enters question
if input_txt:
    response = chain.invoke({"question": input_txt})
    st.write(response)
