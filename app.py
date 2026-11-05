"""UK Food Safety Q&A — a retrieval-augmented generation system.

Retrieves the most relevant guidance documents for a question and answers
only from those, refusing when nothing relevant is found.
"""

import os
import numpy as np
import gradio as gr
import anthropic
from sentence_transformers import SentenceTransformer

from documents import DOCUMENTS

TOP_K = 3
THRESHOLD = 0.25
MODEL_NAME = "claude-haiku-5-5"

# ---------- documents ----------
documents = [{"source": name, "text": text} for name, text in DOCUMENTS.items()]

# ---------- embeddings (computed once at startup) ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = embedder.encode([d["text"] for d in documents])

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def similarity(question):
    """Cosine similarity between the question and every document."""
    q_vec = embedder.encode(question)
    return np.dot(doc_vecs, q_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(q_vec))


def retrieve(question):
    """Return the top documents, or nothing when none clear the threshold."""
    scores = similarity(question)
    if scores.max() < THRESHOLD:
        return []
    top = np.argsort(scores)[::-1][:TOP_K]
    return [documents[i] for i in top]


def ask_model(prompt):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return "No answer was produced."


def answer(question):
    if not question or not question.strip():
        return "Please enter a question."

    hits = retrieve(question)
    if not hits:
        return "I could not find anything relevant in the documents."

    context = "\n\n---\n\n".join(h["text"] for h in hits)

    prompt = f"""<context>
{context}
</context>

Question: {question}

Answer using only the context above. If the answer is not there, say so.
Answer directly. Do not refer to "the context" or "the guidance"."""

    reply = ask_model(prompt)
    sources = sorted({h["source"] for h in hits})
    return reply + "\n\n---\n\n**Sources:** " + ", ".join(sources)


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(
        label="Question",
        placeholder="What temperature should a fridge be?",
        lines=2,
    ),
    outputs=gr.Markdown(label="Answer"),
    title="UK Food Safety Q&A",
    description=(
        "Answers questions from 22 UK food safety guidance documents. "
        "Retrieves the relevant passages and answers only from those. "
        "Says so when it does not know."
    ),
    article=(
        "Summarised from Food Standards Agency guidance. "
        "Not a substitute for the current guidance at food.gov.uk."
    ),
    examples=[
        "What temperature should a fridge be?",
        "do I need to label a sandwich I made this morning?",
        "someone's allergic to nuts, what do I do",
        "can I refreeze defrosted mince",
        "staff member cut their hand, what now",
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
