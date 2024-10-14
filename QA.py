import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the QA model (you can replace it with your own model)
qa_pipeline = pipeline("question-answering")

# Set the title of the app
st.title("Question Answering System")

# Sidebar for additional settings or information
st.sidebar.header("Instructions")
st.sidebar.write("Enter a question related to the provided context below.")

# Input context (this could be dynamic or loaded from a file)
context = st.text_area("Context", height=200, 
                        help="Enter the context for the question answering system here.")

# Input for the user's question
question = st.text_input("Your Question", help="Ask a question about the context.")

# Button to get the answer
if st.button("Get Answer"):
    if context and question:
        # Run the QA pipeline
        answer = qa_pipeline(question=question, context=context)
        st.subheader("Answer:")
        st.write(answer['answer'])
    else:
        st.warning("Please provide both context and a question.")
