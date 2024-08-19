from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

st.title("LangChain Demo With Groq X Llama 3.1")

# Get the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Initialize the LLaMA model from Groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# Initialize the output parser
output_parser = StrOutputParser()

# Combine the prompt, LLM, and output parser into a chain
chain = prompt | llm | output_parser

# Streamlit UI elements
input_text = st.text_input("Ask a question")

if input_text:
    # Use the chain to get a response
    response = chain.invoke({"question": input_text})
    st.write(response)
