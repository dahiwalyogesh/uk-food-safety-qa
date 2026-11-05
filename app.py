"""UK Food Safety Q&A — a retrieval-augmented generation system.

Retrieves the most relevant guidance documents for a question and answers
only from those, refusing when nothing relevant is found.

Two hosting constraints shape this file:

1. A hosted web service is given a short window to open its port, and
   loading the embedding model takes longer than that window. So the port
   is opened first and the model loads on a background thread afterwards.

2. The instance has 512 MB of memory. The same model is run through ONNX
   Runtime rather than PyTorch, whose CPU build plus the framework around
   it does not fit. The weights are identical; only the runtime differs.
"""

import os

# Must be set before ONNX Runtime is imported. Without this, the runtime
# sizes its thread pools from the host's core count, not this container's
# share of a core, and allocates a stack per thread. On a 512 MB instance
# that alone is enough to get the process killed.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import threading

import numpy as np
import gradio as gr
import anthropic

from documents import DOCUMENTS

TOP_K = 3
THRESHOLD = 0.25
MODEL_NAME = "claude-haiku-5-5"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- documents ----------
documents = [{"source": name, "text": text} for name, text in DOCUMENTS.items()]

# ---------- lazily loaded state ----------
_state = {"embedder": None, "doc_vecs": None, "client": None, "error": None}
_lock = threading.Lock()


def load():
    """Load the embedding model and API client once, on first use.

    Guarded by a lock so two simultaneous requests cannot load it twice.
    """
    if _state["embedder"] is not None or _state["error"] is not None:
        return _state

    with _lock:
        if _state["embedder"] is not None or _state["error"] is not None:
            return _state
        try:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set in the environment.")

            from fastembed import TextEmbedding

            embedder = TextEmbedding(EMBED_MODEL, threads=1)
            _state["doc_vecs"] = np.array(list(embedder.embed(
                [d["text"] for d in documents], batch_size=4)))
            _state["client"] = anthropic.Anthropic(api_key=key)
            _state["embedder"] = embedder
        except Exception as exc:  # surfaced in the UI rather than crashing
            _state["error"] = str(exc)

    return _state


def similarity(question):
    """Cosine similarity between the question and every document."""
    s = load()
    q_vec = np.array(list(s["embedder"].embed([question]))[0])
    doc_vecs = s["doc_vecs"]
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
    response = load()["client"].messages.create(
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

    s = load()
    if s["error"]:
        return f"The service could not start: {s['error']}"

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


def scores(question):
    """Show the raw similarity scores behind a question.

    Retrieval is the half of a RAG system that usually fails, and it fails
    quietly. This makes it visible: which documents were closest, how close,
    and whether anything cleared the threshold.
    """
    if not question or not question.strip():
        return "Enter a question."

    s = load()
    if s["error"]:
        return f"The service could not start: {s['error']}"

    sc = similarity(question)
    top = np.argsort(sc)[::-1][:5]

    lines = [f"Highest score: **{sc.max():.3f}** (threshold {THRESHOLD})", ""]
    lines += [f"- `{sc[i]:.3f}` &nbsp; {documents[i]['source']}" for i in top]
    if sc.max() < THRESHOLD:
        lines.append("\nNothing clears the threshold, so the system refuses.")
    return "\n".join(lines)


ask = gr.Interface(
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
        "Says so when it does not know. "
        "The first question after the service wakes can take a minute "
        "while the embedding model loads."
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

diagnostics = gr.Interface(
    fn=scores,
    inputs=gr.Textbox(label="Question", lines=2),
    outputs=gr.Markdown(label="Similarity scores"),
    title="Retrieval scores",
    description=(
        "The cosine similarity between your question and each document. "
        f"Anything below {THRESHOLD} is treated as 'not in the documents'. "
        "The threshold was set by measuring relevant and off-topic questions "
        "and picking a value in the gap between them."
    ),
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [ask, diagnostics],
    ["Ask", "Retrieval scores"],
    title="UK Food Safety Q&A",
)

if __name__ == "__main__":
    # Warm the model in the background so the first visitor does not wait,
    # while the port below opens straight away.
    threading.Thread(target=load, daemon=True).start()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
