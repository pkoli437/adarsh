import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from db import db  # ✅ Pre-loaded ChromaDB

import time

# ==== MODEL CONFIGURATION ====
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# ==== UI ====

st.set_page_config(page_title="🤖 UIDAI Q&A Bot", layout="wide")

st.title("🤖 UIDAI Question Answering Assistant (RAG)")
st.write("Ask any Aadhaar-related question and get an answer grounded in official documents.")

# Question input
question = st.text_input("Ask your UIDAI-related question:", placeholder="e.g., How to update my address in Aadhaar?")
submit = st.button("Get Answer")

if submit and question.strip():
    start = time.time()

    # 🔍 Retrieve top relevant chunks
    results = db.similarity_search_with_relevance_scores(question, k=3)
    filtered = [(doc, score) for doc, score in results if score > 0.5]
    context = "\n\n".join(doc.page_content for doc, _ in filtered)

    if not context:
        answer = """
        I'm sorry, I couldn't find relevant information to answer your question.  
        Please contact UIDAI at 1947 or visit https://uidai.gov.in/ for help.
        """
    else:
        prompt = f"""
        You are a helpful assistant for UIDAI-related queries.

        Only use the information provided in the "Context" section below. 
        If the answer cannot be found in the context, say you don't know.

        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:
        """

        output = generator(prompt, max_new_tokens=512, do_sample=True, temperature=0.3)[0]["generated_text"]
        answer = output.split("### Answer:")[-1].strip()

    st.success("✅ Answer Generated")
    st.text_area("Answer", value=answer.strip(), height=250)
    st.caption(f"🕐 Took {round(time.time() - start, 2)} seconds")
