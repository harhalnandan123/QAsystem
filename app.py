# Step1 - Install necessary librareis

import fitz  # PyMuPDF
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import requests # for reading HTML
from bs4 import BeautifulSoup # for reading HTML

# Step2 - used pre-trained model for summarization and question=answering 

# Used pre-trained model from hugging face this will help to summarize the text
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Used pre-trained model from hugging face this will help to convert text into numbers (vectorization)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight model for embeddings

# Step 3 - created necessory user defined functions

# created a function to extract text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path) # used fitz library to open pdf file 
    text = "" # stored into 'text' variable
    for page_num in range(doc.page_count): 
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Extract text from URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = " ".join([para.get_text() for para in paragraphs])
    return article_text

# Split text into chunks of 100 words
def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size]) # store text from 0 to 100 in 'chunk' variable
        chunks.append(chunk)
    return chunks

# Find the most similar chunk for answering the question
def get_most_relevant_chunk(question, chunks):
    # Encode the question and the chunks
    question_embedding = embedding_model.encode(question, convert_to_tensor=True) #tensor help to find cosine similarity
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True) #tensor help to find cosine similarity
    
    # Compute cosine similarity between question and each chunk
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings).squeeze(0) # last parameter for reshaping
    
    # Find the the most similar chunk
    most_similar_index = similarities.argmax().item() #argmax is used to find most similar chunk
    
    # Return the most similar chunk
    return chunks[most_similar_index]

# Define summarization function
def summarize_text(text, max_chunk_length=256):
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)] #split text into small chunks
    summary = ""
    for chunk in chunks:
        chunk_summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False) # lst code ensure same output each time
        summary += chunk_summary[0]['summary_text'] + " "
        # Limit the summary to 200 words
        if len(summary.split()) > 200:
            summary = " ".join(summary.split()[:200]) + "..."
            break
    return summary if summary else "No summary generated."


# Step4 - Created User-Interface

# UI Setup
st.title("📄 AI-powered PDF Reader")

# Background and custom styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f2f2f2;
        background-image: linear-gradient(to right, #e0f7fa, #e1bee7);
        background-attachment: fixed;
    }
    .sidebar .sidebar-content {
        background-color: #1e88e5;
        color: white;
    }
    .css-1q8dd3e {
        font-size: 20px;
        color: #4CAF50;
    }
    .css-16huue1 {
        color: #1e88e5;
        font-size: 24px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #ff9800;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        margin-top: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #e65100;
    }
    div[role="textbox"] {
        background-color: #fafafa;
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 10px;
    }
    .stTextInput>div>div>input {
        border: 2px solid #1e88e5;
        border-radius: 12px;
    }
    .css-17eq0hr {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar instructions
st.sidebar.image(r"C:\Users\Harshal\OneDrive\Desktop\abhi_capstone\model_logo.png", width=250)
st.sidebar.markdown("### About This App")
st.sidebar.markdown("""
This app allows you to upload a PDF or any URL and provides:
- 📄 Summarization with Hugging Face Transformers (Limited to 200 words)
- ❓ Question Answering with Sentence Transformers
""")

st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
1. **Upload a PDF** or **Enter an Article URL**.
2. Click **Summarize** to get a brief overview of the content.
3. Optionally, enter a question related to the content.
4. Receive answers based on the content.
""")

# Step5 - use my functions in UI

uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")
if uploaded_file:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

# URL Input
url_input = st.text_input("🔗 Enter an article URL:")

# Input Question
question = st.text_input("💬 Ask a question about the PDF:")

# Summarize and Answer Section
summary =""
if url_input:
    with st.spinner("Fetching and summarizing article..."):
        article_text = extract_text_from_url(url_input)
        summary = summarize_text(article_text)
        st.markdown(f"<div style='background-color: #E0F7FA; padding: 15px; border-radius: 8px;'>"
                    f"<strong>Summary of Article:</strong> {summary}</div>", unsafe_allow_html=True)

if uploaded_file:
    # Extract and split text into chunks
    text = extract_text("uploaded_pdf.pdf")
    chunks = split_text_into_chunks(text, chunk_size=200)
    
    if st.button("✨ Summarize"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(text)
            st.markdown(f"<div style='background-color: #E0F7FA; padding: 15px; border-radius: 8px;'>"
                        f"<strong>Summary:</strong> {summary}</div>", unsafe_allow_html=True)

    if question:
        with st.spinner("Finding the most relevant section..."):
            relevant_chunk = get_most_relevant_chunk(question, chunks)
            st.markdown(f"<div style='background-color: #FFF3E0; padding: 15px; border-radius: 8px;'>"
                        f"<strong>Answer:</strong> {relevant_chunk}</div>", unsafe_allow_html=True)
    

# Expander for usage instructions with improved layout
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    - **Step 1**: Upload a PDF document.
    - **Step 2**: Click 'Summarize' to get a brief overview of the document.
    - **Step 3**: Ask a question to receive a specific answer based on the document.
    """)