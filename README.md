# UK Food Safety Q&A

[![Live demo](https://img.shields.io/badge/Live_demo-Open_the_app-2ea44f?style=for-the-badge)](https://uk-food-safety-qa.onrender.com)

**Live app: https://uk-food-safety-qa.onrender.com**

> Hosted on a free instance, which sleeps after 15 minutes of no traffic.
> The first request after it wakes takes about a minute while the embedding
> model loads. Later requests are fast.

A retrieval-augmented generation (RAG) system that answers questions about UK
food safety guidance from 22 source documents, and says so when it does not
know.

## Why I built it

I worked in professional kitchens for several years. Finding the right line in
FSA guidance is slow, the answer is usually one paragraph, and nobody has time
to read a PDF mid-service. The useful version of that guidance is one you can
ask a question.

## How it works

| Stage | Implementation |
|---|---|
| Chunking | Each document is ~200 words covering one topic, so a document is already a chunk. Splitting further would break coherent topics. |
| Embedding | `all-MiniLM-L6-v2`, 384 dimensions, run through ONNX Runtime, computed once on first use |
| Retrieval | Cosine similarity, top 3 documents above a 0.25 threshold |
| Generation | Claude Haiku, answering only from the retrieved text |
| Refusal | Below-threshold questions return nothing; the prompt also requires the model to say when an answer is absent |

The app has two tabs. **Ask** answers questions. **Retrieval scores** shows the
cosine similarity for every document and whether anything cleared the
threshold. Retrieval is the half of a RAG system that fails quietly, so it is
worth making visible rather than leaving it as a black box.

## Setting the threshold

Retrieval always returns something, even when nothing is relevant. The
threshold is what lets the system refuse, and it was set by measurement rather
than guesswork:

| Question type | Score range |
|---|---|
| Relevant (7 questions) | 0.338 – 0.730 |
| Off topic (4 questions) | 0.018 – 0.224 |

Nothing lands between 0.224 and 0.338, so a threshold of **0.25** sits in the
gap with margin on both sides. An earlier value of 0.35 sat above the gap and
rejected three valid questions.

## Results

Tested on two sets. The second is how a chef would actually type mid-service,
and is the harder test.

| Test set | Result |
|---|---|
| Written questions (12) | 12/12 |
| Colloquial questions (9) | 8/9 |
| Out-of-scope correctly refused | 5/5 |

## Getting it running in production

The system worked in a notebook long before it worked when deployed. Three
things broke, none of them to do with the retrieval logic.

**1. The app was killed before it could start.** A hosted web service is given
roughly 40 seconds to open its port. The original code loaded the model and
embedded every document *before* calling `launch()`, which took minutes, so
the platform killed it every time. Fix: open the port first, load the model on
a background thread afterwards.

**2. A failed deploy blocked the fix.** The broken deploy sat retrying for 19
minutes, and because only one deploy runs at a time, the corrected version
never started building. Worth knowing that a stuck deploy has to be cancelled,
not waited out.

**3. PyTorch did not fit in 512 MB.** Installing `sentence-transformers` pulls
in PyTorch and the entire CUDA stack — on a server with no GPU. The process
was killed by the OS with no traceback, which is the signature of running out
of memory rather than a crash. Fix: run the same model weights through ONNX
Runtime instead, and pin the thread count, since the runtime otherwise sizes
its thread pools from the host's core count rather than the container's share.

The embedding model is unchanged throughout — same weights, same 384
dimensions. Only the runtime executing it differs.

## What doesn't work

- **Questions about elapsed time retrieve poorly.** "Sauce been out since 11,
  bin it?" scores 0.218 against `date_labels.txt`; the correct document
  (`hot_holding.txt`) is not in the top 3. Semantically the question is about
  time at room temperature while the document is about keeping food hot.
  Lowering the threshold would not fix this — it would return the wrong
  document with more confidence.
- **Corpus gaps are invisible to the user.** Nothing covers shelf life of
  specific foods or fryer oil changes. The system correctly declines, but
  cannot say whether the gap is in the corpus or in the law.
- **The threshold is a single global number.** A per-question confidence
  estimate would be better than one cutoff for all query types.
- **No re-ranking.** The top 3 are taken on raw similarity alone.
- **22 documents.** Real FSA guidance is far larger, and chunking would become
  necessary at that scale.

## What I would do differently

Use a proper vector database rather than NumPy; chunk on section boundaries
once the corpus grows; add a re-ranking step after retrieval; precompute the
document embeddings at build time rather than on first request; and build the
test set before writing any retrieval code rather than after.

## Evaluation notes

Two test expectations were changed after seeing the output. Both were cases
where I had assumed a vague or unanswerable question should be refused at the
retrieval stage; measurement showed that retrieving related documents and
letting the model state the gap produced a more useful answer. The expectations
were wrong, not the system — but the distinction matters, and changing a test
to raise a score would not have been legitimate.

The threshold was first measured with the PyTorch build of the model. After the
deployment switched to ONNX Runtime, the same eleven probe questions were
re-run against the live app rather than assumed to be unchanged. The ranges
came back identical — relevant 0.338 – 0.730, off topic 0.018 – 0.224 — which
confirms the runtime change did not move the vectors. The **Retrieval scores**
tab shows the live number for any question.

## Running it yourself

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # never commit this
python app.py
```

## Source material

Summarised from Food Standards Agency guidance at food.gov.uk. Not a
substitute for the current official guidance.
