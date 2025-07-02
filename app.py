import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datetime, re, threading
from duckduckgo_search import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          TextIteratorStreamer)
import torch, gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = dict(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    **bnb_config
).eval()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(collection_name="biz_ideas", embedding_function=embeddings)

def web_search_snippets(query, k=5):
    """Yield (title, snippet, url) tuples from DuckDuckGo."""
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            yield r["title"], r["body"], r["href"]

def retrieve_context(question, idea, num_docs=6):
    """Search + store + retrieve semantically similar passages."""
    search_q = f"{idea} market size trend {datetime.datetime.now().year}"
    for title, body, url in web_search_snippets(search_q, k=10):
        content = f"{title}\n{body}\nSource: {url}"
        vectordb.add_texts([content])
    similar = vectordb.similarity_search(question, k=num_docs)
    context = "\n\n".join(doc.page_content for doc in similar)
    return context[:4000]   # stay within prompt budget



template = """
You are an experienced venture‑capital analyst.

Context:
{context}

User question:
{question}

Respond with **exactly** these sections:

1. **Market size & CAGR** – in USD and local currency if available  
2. **Current trend line** – growth, decline, plateau? Cite fresh data.  
3. **Five‑point SWOT** – bullets (3 pros, 2 cons is fine)  
4. **Traffic‑light verdict** – "Green", "Yellow", or "Red" with one‑sentence reason

Be concise but data‑driven. Use markdown where helpful.
"""
PROMPT = PromptTemplate.from_template(template)


def llm_chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_params = dict(**inputs,
                      streamer=streamer,
                      max_new_tokens=512,
                      temperature=0.7,
                      top_p=0.9)
    thread = threading.Thread(target=model.generate, kwargs=gen_params)
    thread.start()
    for chunk in streamer:
        yield chunk

history = ChatMessageHistory()

def chat(user_msg, idea):
    history.add_user_message(user_msg)
    ctx = retrieve_context(user_msg, idea)
    full_prompt = PROMPT.format(context=ctx, question=user_msg)
    response_stream = llm_chat(full_prompt)

    collected = ""
    for piece in response_stream:
        collected += piece
        yield collected
    history.add_ai_message(collected)

with gr.Blocks(title="Business‑Idea Analyst") as demo:
    gr.Markdown("# 💼📊 AI Business‑Idea Analyst\nAsk anything about your startup idea.")
    idea_box = gr.Textbox(label="Business idea", placeholder="e.g., Vertical farming in Lagos")
    q_box    = gr.Textbox(label="Question", placeholder="Is this idea worth pursuing?")
    answer   = gr.Markdown()

    send_btn = gr.Button("Send")
    send_btn.click(chat, inputs=[q_box, idea_box], outputs=answer)
    gr.Markdown("*(Uses Phi‑3‑Mini 4‑bit + live DuckDuckGo snippets)*")

demo.launch(share=True)

