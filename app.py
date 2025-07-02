import os
import datetime
import threading
from duckduckgo_search import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
import gradio as gr
from groq import Groq  # You need to install groq: pip install groq

DEVICE = "cpu"  # no GPU needed

# Setup Groq client with your API key from environment variable
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    return context[:4000]  # stay within prompt budget


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
    # Use Groq Cloud API streaming completions
    stream = GROQ.chat.completions.create(
        model="llama3-70b-8192",  # example, change to your chosen model on Groq
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    for delta in stream:
        yield delta.choices[0].delta.get("content", "")


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
    q_box = gr.Textbox(label="Question", placeholder="Is this idea worth pursuing?")
    answer = gr.Markdown()

    send_btn = gr.Button("Send")
    send_btn.click(chat, inputs=[q_box, idea_box], outputs=answer)
    gr.Markdown("*(Uses Groq Cloud LLM + live DuckDuckGo snippets)*")

demo.launch(share=True)
