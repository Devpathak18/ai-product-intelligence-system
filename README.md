# AI Product Intelligence System

> 3 AI features for e-commerce product intelligence,
> powered by one shared CLIP embedding backbone.
> Built on 6,000 real fashion product images.

**Dataset:** Fashion Product Images (Small) — Kaggle
**Model:** CLIP ViT-B-32 (OpenAI pretrained weights)
**Built by:** Dev Pathak | github.com/Devpathak18

---

## What It Does

| Feature | Input | Output |
|---|---|---|
| Reverse Search | Text query | Ranked product photos |
| Duplicate Detection | Full 6,000-item catalog | 5,649 unique products |
| Complementary Recs | Product ID | Items bought together |

One CLIP model. Embeddings computed once. 
Reused across all three features.

---

## Why This Architecture

CLIP maps both images and text into the same 
vector space. This single property enables:

- Image vs Image — duplicate detection
- Text vs Image — reverse search  
- Image embedding as style signal — recommendations

Most systems build three separate models for these.
This system builds one and uses it three ways.

---

## Feature 1 — Reverse Product Search

**Problem:**
Users want to search by describing what they want,
not by uploading a reference photo.

**How it works:**
```
User types: "blue casual shirt"
        |
CLIP text encoder → text embedding vector
        |
Cosine similarity against all 6,000 
product image embeddings
(dot product on L2-normalized vectors)
        |
Top-K products returned ranked by similarity
```

**Queries tested:**

Query: "blue casual shirt"
![Reverse search result](screenshots.md/reverse_search_blue_casual_shirt.png)

Query: "red party dress"
![Reverse search result](screenshots.md/reverse_search_red_party_dress.png)

---

## Feature 2 — Duplicate Catalog Detection

**Problem:**
Different sellers upload near-identical products,
polluting the catalog with redundant listings.

**Result:**
6,000 products → 5,649 unique products
277 duplicate groups identified and removed

**Two bugs found and fixed:**

Bug 1 — Image similarity alone is not enough.
Small studio-shot product photos on plain 
backgrounds cause CLIP to embed genuinely different 
products as similar. Image similarity alone 
falsely flagged them as duplicates.

Fix: require image similarity AND matching 
baseColour AND matching productDisplayName 
(85%+ string similarity ratio via difflib).

Bug 2 — Transitive chaining creates false groups.
Plain union-find merges A, B, C if A~B and B~C,
even when A and C are not actually duplicates.

Fix: a new item only joins a group if it is a 
duplicate of EVERY existing member — mutual clique,
not connected chain.

**Similarity distribution analysis:**
![Similarity distribution](screenshots.md/similarity_distribution_tshirts.png)

**Duplicate group detected and removed:**
![Duplicate group](screenshots.md/duplicate_group_belts_1.png)

**Thresholds:**
- Image similarity: 0.95 cosine
- Name similarity: 0.85 sequence match
- Scope: within same articleType only

---

## Feature 3 — Complementary Recommendations

**Problem:**
Visual similarity recommends more of the same.
A running shoe system should recommend socks and
a fitness watch, not more running shoes.

**How it works:**
```
Input: Running shoe product ID
        |
Look up articleType in category-affinity table
Sports Shoes → [Socks, Track Pants, 
               Sports Watches, Water Bottle]
        |
For each complementary category,
rank candidates by CLIP style similarity
to the input product embedding
        |
Return top-K per complementary category
```

**Example — Puma sneaker input:**
![Complementary recommendations](screenshots.md/recommendation_puma_sneaker.png)

**Honest limitation:**
Affinity table is hand-authored here because 
this dataset has no purchase history data.
In production, this would be learned from 
transaction logs via market-basket analysis
or association rule mining. CLIP similarity
is used only for the final style-ranking step,
which is the correct and honest use of the model.

---

## Day 1 — System Architecture Design

Before building, designed a production-grade 
multimodal AI pipeline for automatic product 
categorisation with Human-in-the-Loop review layer.

```
Product photo uploaded by seller
        |
Quality check
(blur / lighting / visibility)
        |
    Pass              Fail
      |                 |
Vision-Language     Human Review
Model (CLIP/BLIP)       Queue
      |                 |
Category + Description generated
        |
Published as product listing
```

Five architectural layers:
1. Seller application layer
2. API and data ingestion layer
3. AI and model inference layer
4. Human review and quality control layer
5. Data storage and catalog layer

![System architecture](screenshots.md/day1_pipeline.png)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| PyTorch | Deep learning framework |
| open_clip | CLIP model loading and inference |
| NumPy | Embedding computation and similarity |
| Pandas | Dataset loading and management |
| Matplotlib | Result visualisation |
| difflib | String similarity for name matching |
| Kaggle | Dataset source and GPU compute |
| Jupyter Notebook | Development environment |

---

## Results

| Metric | Value |
|---|---|
| Total products processed | 6,000 |
| Duplicate groups found | 277 |
| Unique products after dedup | 5,649 |
| Reverse search queries tested | 4 |
| Complementary categories mapped | 11 |
| Embedding dimensions | 512 (ViT-B-32) |

---

## Repository Structure

```
ai-product-intelligence-clip/
├── README.md
├── notebook.ipynb
├── day1_architecture.pdf
└── screenshots/
    ├── reverse_search_blue_casual_shirt.png
    ├── reverse_search_red_party_dress.png
    ├── similarity_distribution_tshirts.png
    ├── duplicate_group_belts_1.png
    ├── recommendation_puma_sneaker.png
    └── day1_pipeline.png
```

---

## Setup

**Requirements:**
```
pip install open_clip_torch torch numpy pandas 
matplotlib kagglehub pillow
```

**Run:**
1. Open notebook.ipynb in Jupyter or Kaggle
2. Dataset downloads automatically via kagglehub
3. CLIP embeddings computed and cached on first run
4. All three features run sequentially

---

## Related Projects

| Project | Description |
|---|---|
| [AI Email Automation](https://github.com/Devpathak18/email-ai-automation) | Gmail → Gemini AI → Google Sheets → Telegram |
| [AI Research Automation](https://github.com/Devpathak18/ai-research-automation) | Any topic → Local LLM → 7-step guide → Google Docs |

---

## Author

Dev Pathak
B.Tech Computer Science and Business Systems (CSBS)
Jain Deemed-to-be University, Bangalore
Program co-designed with Tata Consultancy Services

GitHub: github.com/Devpathak18
LinkedIn: linkedin.com/in/devpathak18
Email: devpathakpersonal@gmail.com

Open to remote internships in AI, Machine Learning,
Computer Vision, and Data Analytics.
