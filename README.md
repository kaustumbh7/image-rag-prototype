# image-rag-prototype

Text-to-image retrieval over COCO 2017 using open_clip + FAISS, with an optional Claude vision RAG layer that answers questions grounded in the retrieved images.

## Demo

📹 **[Watch the walkthrough video](https://drive.google.com/file/d/1exIDGgknseLXdU_r8IwSvV7JI15WBDlB/view?usp=drive_link)** — short recorded demo of the prototype in action: the headline query, examples of queries that work well, known failure modes, and a brief tour of the tech stack.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # then fill ANTHROPIC_API_KEY
```

## Quick start (small set, ~5k images)

```bash
python scripts/download_coco.py --split val2017
python scripts/build_index.py
streamlit run src/app.py
```

## Full set (~118k images)

```bash
python scripts/download_coco.py --split train2017
python scripts/build_index.py --rebuild
```

Index build on Apple Silicon (MPS) takes roughly 30–60 min for train2017 at ViT-B/32, batch 64.

## Layout

- `src/embed.py` — open_clip wrapper (image + text encoders, L2-normalized).
- `src/index.py` — FAISS `IndexFlatIP` build/load + `paths.txt` sidecar.
- `src/retrieve.py` — `search(query, k)` → list of (path, score).
- `src/rag.py` — Claude vision call (`claude-sonnet-4-6`) over retrieved images.
- `src/app.py` — Streamlit UI.
- `scripts/download_coco.py`, `scripts/build_index.py` — one-time setup.

## Documentation

- **`WRITEUP.md`** — design notes covering the four big questions about the prototype: which query shapes it handles well, which it doesn't (and why), the tech-stack choices with their trade-offs, and how I'd actually evaluate the system's quality.

- **`presentation.html`** — self-contained 10-slide deck used as the visual backdrop when recording a walkthrough demo. Open it directly in a browser (`open presentation.html`) — no build step or server needed. Navigate with **← / →** arrow keys, **Space**, **PageUp/PageDown**, **Home/End**, or click the left/right half of the screen. The current slide is reflected in the URL hash (`#s=4`), so refreshing keeps your place. Covers: title, problem statement, headline query, example queries that work / fail, pipeline overview, end-to-end flow, components, evaluation approach, and a closing slide.

## Example query

> Give me 5 images with men wearing a hat 

In the Streamlit app, type the query, set k=5, optionally toggle the Claude RAG to get a grounded answer that references each retrieved image by index.
