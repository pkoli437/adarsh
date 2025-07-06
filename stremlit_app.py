import os
import fitz  # PyMuPDF
import streamlit as st
from db import process_pdf_and_store, perform_search
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# --------------------
# Initialize the model
# --------------------
@st.cache_resource
def load_llm():
    model_id = "Meta-Llama-3-8B-Instruct"  # Or your preferred local LLM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_llm()

# --------------------
# Load vector DB
# --------------------
PERSIST_DIR = "RSP"
COLLECTION_NAME = "test-1"

@st.cache_resource
def load_chroma_db():
    # Use existing ChromaDB collection from `db.py`
    return process_pdf_and_store(None, persist_directory=PERSIST_DIR, collection_name=COLLECTION_NAME)

db = load_chroma_db()

# --------------------
# Generate answer from LLM using context
# --------------------
def generate_answer_llm(context, question):
    SYSTEM_PROMPT = "You are an AI assistant. Use the context below to answer the user's question clearly and accurately."
    USER_PROMPT = f"""
Context:
{context}

Question:
{question}

Answer:"""

    prompt = SYSTEM_PROMPT + "\n" + USER_PROMPT
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.replace(prompt, "").strip()

# --------------------
# Streamlit UI
# --------------------
st.markdown("""
<h1 style='text-align:center;'>📄 AI-Powered PDF QA</h1>
<h4 style='text-align:center;color:gray;'>Built with RAG (Retrieval-Augmented Generation)</h4>
""", unsafe_allow_html=True)

st.divider()
st.subheader("Ask a question from your document")

question = st.text_input("🔍 Enter your question:")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Retrieving and generating..."):
            context = perform_search(db, question, similarity_threshold=0.05)
            if not context.strip():
                st.error("No relevant context found.")
            else:
                answer = generate_answer_llm(context, question)
                st.subheader("🧠 Answer")
                st.write(answer)
    else:
        st.warning("Please enter a question.")

