# Image RAG prototype — design notes

Companion writeup to the prototype. Covers the queries the system handles well, the queries it doesn't, the choices behind the tech stack with their trade-offs, and how I'd actually measure whether the system is any good.

---

## 1. Queries it handles well

CLIP was trained on hundreds of millions of (image, caption) pairs scraped from the open web, so its sweet spot is the kind of language those captions use: concrete nouns, simple verbs, common adjectives, ordinary scenes. The prototype inherits that sweet spot directly. In practice these query shapes work reliably:

**Object + scene.** A single subject placed in a recognizable setting.
- `a red double-decker bus on a city street`
- `a golden retriever on a couch`
- `a slice of pizza on a wooden table`

**Single-subject + simple attribute.** One concrete attribute attached to one subject.
- `a woman in a red dress`
- `a man wearing sunglasses`
- `a child holding a balloon`

**Activity.** A subject paired with a common verb. Captions are full of these.
- `a woman walking`
- `two people playing tennis`
- `a person riding a horse`

**Style or mood.** Stylistic descriptors live everywhere in alt-text.
- `a black and white photo of a person riding a bicycle at night`
- `a vintage portrait`
- `a foggy mountain landscape`

**Common categorical adjectives.** Coarse demographics, sizes, colours when paired with one subject.
- `an elderly man at a table`
- `a small dog on grass`

The pattern: one subject, one or two attributes, one setting. Past three attributes, retrieval quality degrades noticeably even on queries that *seem* simple.

---

## 2. Queries it doesn't handle well

These failure modes aren't bugs in the prototype — they're inherent to how CLIP-family embeddings represent meaning. The contrastive training objective rewards a kind of "bag of concepts" matching: the vector encodes *what is present* in the image, much less so *how the concepts relate*.

**Counting.** CLIP has no reliable notion of cardinality.
- `exactly three apples on a wooden table` returns apple-and-table-shaped images regardless of count. One apple, a dozen apples, all score similarly.

**Spatial relations.** Left/right/above/below are mostly ignored.
- `a cat sitting to the left of a dog` returns images that contain a cat and a dog. The spatial relation between them is essentially noise to the model.

**Attribute binding across multiple objects.** Colours can swap between subjects.
- `a woman in a red shirt and blue pants` will often return women in *blue* shirts and *red* pants. CLIP doesn't strictly tie attributes to the right objects when more than one subject and one attribute are in play.

**Negation.** "Without" embeds almost identically to "with".
- `a kitchen without a microwave` is roughly equivalent in CLIP-space to `a kitchen with a microwave` — both push toward kitchen-and-microwave images. This is true of essentially every CLIP-family model today.

**Text inside images (OCR-style queries).** CLIP only weakly reads pixel-text.
- `a sign that says STOP`, `a t-shirt with the word HELLO` are unreliable. Newer models (SigLIP 2, OCR-aware variants) handle this much better.

**Long multi-clause queries.** CLIP's text encoder has a 77-token limit and short captions dominated its training data, so verbose queries get truncated or averaged into mush.
- A 50-word description with five conditions usually retrieves worse than a 5-word distillation of the most important one or two.

Useful framing for a demo: showing the failure modes alongside the wins makes the system more trustworthy, not less. Knowing the boundary lets you steer queries away from it — and the next-step work (re-rankers, stronger embedders) is exactly aimed at this boundary.

---

## 3. Tech stack and trade-offs

The prototype is small on purpose. Every choice is the simplest thing that demonstrates the idea end-to-end, with deliberate notes on what each one trades away.

**Dataset — COCO 2017 (~120k images).**
Diverse, well-known, free, with caption annotations available for the validation split (useful for quantitative evaluation). The trade-off is generality: for a fashion-specific or product-search demo, a domain dataset like DeepFashion (~200k clothing images) would retrieve dramatically better on niche queries but lose the general-purpose feel. COCO is the right neutral choice for a general prototype.

**Embedding model — OpenCLIP ViT-B/32, LAION-2B pretrained.**
A 151M-parameter Vision Transformer with a small text transformer, producing 512-d embeddings. Small enough to embed 120k images on an Apple-Silicon laptop in roughly 45 minutes on MPS; queries are essentially instantaneous. The trade-off is quality on the harder cases — a larger model like ViT-L/14 or SigLIP would noticeably improve compositional and fine-grained retrieval, at the cost of needing a GPU for the index build to finish in a reasonable time. ViT-B/32 is the right "demo on a laptop" point on the curve.

**Vector index — FAISS `IndexFlatIP`.**
A flat brute-force inner-product index. At 120k vectors of 512 dimensions, each query is effectively one matrix multiply followed by a top-k argmax — well under 10 milliseconds, and the recall is exact. I deliberately avoided IVF or HNSW: approximate indices only pay off above a few million vectors and they trade exact recall for speed we don't need at this scale. The same code generalizes to millions of vectors by swapping `IndexFlatIP` for `IndexIVFFlat` later.

**Normalization — L2 at write time.**
Image and text vectors are L2-normalized before they enter the index. Once everything is unit-norm, inner product equals cosine similarity, so we can use the simpler `IndexFlatIP` instead of an explicit cosine-distance index. Small detail, but it's the reason scores are interpretable: similarities consistently fall in roughly `0.20–0.40` for good matches, which makes thresholding and debugging easy.

**RAG layer — Claude Sonnet 4.6 (vision).**
For grounded question-answering over the retrieved images. Retrieved JPEGs are downsized to a max of 1024 px, base64-encoded, and packed into a single multimodal message together with the user's question. A system prompt forces Claude to answer *only* from the provided images and to refuse if the images don't support an answer — this is what makes it "grounded" rather than free-form hallucination. The trade-off is cost and latency: each RAG call sends ~5 images through, so it's gated behind a UI toggle to avoid burning tokens on every search.

**UI — Streamlit.**
A single Python file gets you a web UI with sliders, toggles, and image grids. Hot-reload is built in. The model and FAISS index are loaded once via `@st.cache_resource` so only the first query pays the load cost. Not a production choice — for that I'd reach for FastAPI + a proper frontend — but ideal for the "single command, demo on a laptop" shape of this prototype.

---

## 4. Evaluating quality

A retrieval-plus-RAG system has two distinct quality questions — *did we retrieve the right images?* and *given those images, is the answer correct and grounded?* — and they need separate evaluation. The approach I'd take, in increasing rigor:

**Spot-checks.** A handful of representative queries across categories (objects, activities, style, plus the known hard cases — counting, spatial, negation) and a manual look at the top 5. Fast, qualitative, and the right first signal whenever something in the pipeline changes. Not enough on its own but cheap to repeat.

**Recall@k against COCO captions.** The cleanest quantitative number for retrieval quality. COCO ships with 5 captions per image in `val2017`. For each ground-truth caption, embed it and check whether the originally-captioned image lands in the top-k. Recall@1, Recall@5, and Recall@10 are the standard reporting metrics. ViT-B/32 typically lands around 30–40% R@1 and 60–70% R@5 on COCO; that's the baseline to beat when swapping in a different embedder.

**Per-category drill-down.** A single Recall@k number is too coarse. I'd partition a labelled query set by type — counting queries, spatial queries, compositional queries with multiple attribute–object bindings, colour queries, negation queries — and report recall per bucket. That turns "the model is 35% R@1" into a *map of where it fails*, which directly guides what to fix next.

**RAG quality.** Once retrieval is reasonable, the question shifts to grounding. The cheapest version is rubric-based human review on ~50 (query, retrieved-images, answer) tuples scored on three dimensions: *grounded* (does the answer reference what's actually in the images?), *refusal hygiene* (does it correctly say "the images don't support this" when the retrieval missed?), *hallucination* (does it invent details that aren't there?). The next step up is LLM-as-judge: a held-out stronger model scoring the same rubric, validated against the human scores on a small calibration set.

**Latency and cost.** For anything that goes past prototype, I'd track end-to-end query time (text encoding + FAISS lookup + thumbnail render), embed-throughput on the index build, and dollars per RAG call. None of these are bottlenecks at 120k images, but they're the constraints that bite first when you push toward millions of images or many concurrent users.

**A/B comparison harness for swaps.** The whole point of the metrics above is to give you a single number per change. Swapping ViT-B/32 → ViT-L/14, or adding a cross-encoder re-ranker, or filtering by a metadata index — each change should produce a R@k delta on the same held-out query set, a per-category delta, and a latency/cost delta. Without that harness, "this looks better in spot-checks" is the only signal, and that's not enough to make a real upgrade decision.
