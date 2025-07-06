import os
import pickle
import re
import torch
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
import time  # Import time module
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Function to read text from a PDF file object
def read_pdf(file) -> str:
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Function to split text into chunks
def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    end = 0
    while end < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

# Function to process multiple PDF files
def process_pdfs(uploaded_files) -> List[Tuple[str, str]]:
    all_chunks = []
    if uploaded_files is None:
        return all_chunks

    for uploaded_file in uploaded_files:
        try:
            pdf_text = read_pdf(uploaded_file)
            chunks = split_text(pdf_text, chunk_size=2000, overlap=100)
            all_chunks.extend([(chunk, uploaded_file.name) for chunk in chunks])
        except Exception as e:
            print(f"Error processing file {uploaded_file.name}: {e}")
    
    return all_chunks

# Load models
if "model" not in st.session_state:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.session_state.model_1 = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    model_id = "Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    st.session_state.model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id)

# Function to get most similar chunks
def get_most_similar_chunks(question, embeddings, chunks, sources, top_n=3):
    query_embeddings = st.session_state.model_1.encode([question], convert_to_tensor=False)
    similarities = cosine_similarity(query_embeddings, embeddings)
    Similarity_list = similarities[0].tolist()

    chunks_copy = chunks[:]
    sources_copy = sources[:]

    final_list = []
    for _ in range(min(top_n, len(Similarity_list))):
        most_similar_idx = np.argmax(Similarity_list)
        final_list.append((chunks_copy[most_similar_idx], sources_copy[most_similar_idx]))
        Similarity_list.pop(most_similar_idx)
        chunks_copy.pop(most_similar_idx)
        sources_copy.pop(most_similar_idx)

    return final_list

# Function to generate answer from LLM
def generate_answer_from_llm(question, retrieved_chunks_and_sources):
    context = "\n\n".join([f"Source: {source}\n{chunk}" for chunk, source in retrieved_chunks_and_sources])

    SYSTEM_PROMPT = "You are an AI assistant. You are able to find answers to questions from the contextual passage snippets provided."
    USER_PROMPT = f"""
    Here is some context:

    {context}

    Based on this context, please answer the following question:

    Question: {question}

    Please answer using the following format:

    Answer: <your answer here>

    Sources: <just the source filenames used>
    """


    inputs = st.session_state.tokenizer(SYSTEM_PROMPT + USER_PROMPT, return_tensors="pt", truncation=True, max_length=4096).to(st.session_state.model.device)
    outputs = st.session_state.model.generate(**inputs, max_length=2048, num_return_sequences=1)
    response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    # Extract the answer part from the response
    answer_match = re.search(r'(.*?)(Sources:|$)', response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else "Response does not contain the expected answer."

    # Append the sources to the answer
    unique_sources = set([source for _, source in retrieved_chunks_and_sources])
    sources_text = "\n".join(unique_sources)
    return f"Answer: {answer}\n\nSources:\n{sources_text}"

# Streamlit UI
st.markdown("""
<style>
.title {
    text-align: center;
    position: relative;
}
.return {
    position: absolute;
    right: 0;
    bottom: -20px;
    font-size: 12px;
    color: gray;
}
.return strong {
    color: red;
}
.timer-box {
    display: inline-block;
    background-color: #f1f1f1;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 5px 10px;
    margin-left: 10px;
    font-size: 12px;
    color: #333;
}
</style>
<div class="title">
    <h1>AI-Powered Information Retrieval System</h1>
    <span class="return"> <strong>- By Puru Koli</strong></span>
</div>
""", unsafe_allow_html=True)

# Create tabs for functionality
tab1, tab2 = st.tabs(["Create Embeddings", "Upload Embeddings"])

# Tab 1: Create embeddings
with tab1:
    # Initialize session state for uploaded files and embeddings
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'embeddings_created' not in st.session_state:
        st.session_state.embeddings_created = False

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    # Update session state with uploaded files
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.write("Files uploaded. Click the button below to create embeddings.")

    # Button to create embeddings
    if st.button("Create Embeddings"):
        if st.session_state.uploaded_files:
            st.write("Processing the uploaded PDF files...")
            
            # Create columns for progress bar and timer box
            col1, col2 = st.columns([3, 1])  # Adjust the column widths as needed

            with col1:
                progress_bar = st.progress(0)  # Initialize progress bar

            with col2:
                timer_box = st.empty()  # Placeholder for the timer box

            timer_start = time.time()  # Start the timer

            # Process PDFs and create embeddings
            chunks_with_sources = process_pdfs(st.session_state.uploaded_files)
            
            if chunks_with_sources:
                chunks = [chunk for chunk, _ in chunks_with_sources]
                sources = [source for _, source in chunks_with_sources]

                # Convert text chunks to embeddings
                chunk_embeddings = st.session_state.model_1.encode(chunks, convert_to_tensor=False)

                # Save embeddings
                with open('embeddings.pkl', 'wb') as f:
                    pickle.dump((chunk_embeddings, chunks, sources), f)

                st.session_state.embeddings_created = True
                st.write("Embeddings have been created and saved.")

                # Provide download link for the embeddings file
                with open('embeddings.pkl', 'rb') as f:
                    st.download_button('Download Embeddings', f, file_name='embeddings.pkl')

            else:
                st.write("No valid chunks were processed from the uploaded files.")

            # Update progress bar to 100%
            progress_bar.progress(100)

            timer_end = time.time()  # End the timer
            elapsed_time = timer_end - timer_start

            # Display the elapsed time in the timer box
            with col2:
                timer_box.markdown(f"<div class='timer-box'>Time taken: {elapsed_time:.2f} seconds</div>", unsafe_allow_html=True)

        else:
            st.write("Please upload PDF files before creating embeddings.")

    # Form for query input and response generation
    with st.form("query_form"):
        query = st.text_input("Enter your query:")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if query:
            if st.session_state.embeddings_created and os.path.exists('embeddings.pkl'):
                with open('embeddings.pkl', 'rb') as f:
                    embeddings, chunks, sources = pickle.load(f)
                
                retrieved_chunks_and_sources = get_most_similar_chunks(query, embeddings, chunks, sources, top_n=3)
                if retrieved_chunks_and_sources:
                    answer = generate_answer_from_llm(query, retrieved_chunks_and_sources)
                    st.expander("Similar Chunks", expanded=True).write("\n\n".join([f"Source: {source}\n{chunk}" for chunk, source in retrieved_chunks_and_sources]))
                    st.write(answer)
                else:
                    st.write("No relevant chunks found.")
            else:
                st.write("Embeddings file not found. Please create embeddings first.")
        else:
            st.write("Please enter a query.")

# Function to update and save embeddings
def update_embeddings(existing_file: str, new_chunks: List[str], new_sources: List[str]) -> None:
    try:
        # Load existing embeddings
        with open(existing_file, 'rb') as f:
            existing_embeddings, existing_chunks, existing_sources = pickle.load(f)
        
        # Convert new chunks to embeddings
        new_chunk_embeddings = st.session_state.model_1.encode(new_chunks, convert_to_tensor=False)
        
        # Merge existing and new embeddings
        updated_embeddings = np.vstack([existing_embeddings, new_chunk_embeddings])
        updated_chunks = existing_chunks + new_chunks
        updated_sources = existing_sources + new_sources
        
        # Save updated embeddings
        with open('updated_embeddings.pkl', 'wb') as f:
            pickle.dump((updated_embeddings, updated_chunks, updated_sources), f)
        
        st.write("Embeddings updated successfully.")
        
        # Provide download link for the updated embeddings
        with open('updated_embeddings.pkl', 'rb') as f:
            st.download_button('Download Updated Embeddings', f, file_name='updated_embeddings.pkl')

        # Set the updated embeddings file for querying
        st.session_state.embeddings_file = 'updated_embeddings.pkl'
    except Exception as e:
        st.error(f"Error loading or updating embeddings: {e}")

# Tab 2: Upload embeddings
with tab2:
    st.header("Manage Existing Embeddings")

    uploaded_embeddings = st.file_uploader("Upload Existing Embeddings File", type="pkl", key="existing_embeddings")
    
    if uploaded_embeddings:
        st.session_state.embeddings_created = True  # Set to true to enable querying

        # Save the uploaded embeddings file temporarily
        with open('existing_embeddings.pkl', 'wb') as f:
            f.write(uploaded_embeddings.read())
        
        st.write("Existing embeddings file uploaded successfully.")

        st.subheader("Optional: Add More Data")
        new_pdfs = st.file_uploader("Upload Additional PDF Files to Update Embeddings", type="pdf", accept_multiple_files=True)

        if new_pdfs:
            st.write("Processing additional PDF files...")

            # Process the new PDF files
            new_chunks_with_sources = process_pdfs(new_pdfs)
            
            if new_chunks_with_sources:
                new_chunks = [chunk for chunk, _ in new_chunks_with_sources]
                new_sources = [source for _, source in new_chunks_with_sources]

                update_embeddings('existing_embeddings.pkl', new_chunks, new_sources)
        
        query = st.text_input("Enter your query:")
        submit_button = st.button("Submit")

        if submit_button:
            if query:
                # Load the updated embeddings file for querying
                embeddings_file = st.session_state.get('embeddings_file', 'existing_embeddings.pkl')
                if os.path.exists(embeddings_file):
                    with open(embeddings_file, 'rb') as f:
                        embeddings, chunks, sources = pickle.load(f)

                    #st.write(f"Loaded embeddings count: {len(embeddings)}")
                    
                    retrieved_chunks_and_sources = get_most_similar_chunks(query, embeddings, chunks, sources)

                    st.subheader("Similarities")
                    with st.expander("View Similar Chunks"):
                        for chunk, source in retrieved_chunks_and_sources:
                            st.write(f"Source: {source}")
                            st.write(chunk)
                            st.write("---")
                    
                    answer = generate_answer_from_llm(query, retrieved_chunks_and_sources)
                    st.subheader("Refined Answer")
                    st.write(answer)
                else:
                    st.write("Please upload a valid embeddings file first.")
    else:
        st.write("Upload your existing embeddings file to get started.")
