# TREES-LLM — Claude Code Project Configuration

> Project context loaded on every session start. Update this file when research direction or folder structure changes.
> **Last updated:** 2026-05-16

---

## Project Overview

**Name:** TREES-LLM (Tree Understanding and Reasoning Enhanced System)
**Domain:** Surveying & Remote Sensing / Point Cloud Processing / Forestry / Multimodal LLM
**Core:** PointLLM (ECCV 2024) route — Web platform where users upload MLS point clouds and a LLM directly understands 3D scenes to segment trees, extract parameters (DBH, height, crown, volume), answer questions, and generate reports.
**Stage:** Development (PointLLM skeleton complete, integration testing)

**Tech route:** PointNet++ Encoder → VQ-VAE Tokenizer → LLM Projector → GLM-4-Flash (cloud API, zero training)

---

## Behavioral Guidelines (karpathy-skills v1 + v2)

Follow these rules in every task, every turn.

### Rule 1 — Think Before Coding
For every non-trivial task: understand the problem, identify affected files, plan the minimal change. Do not start typing code without a clear picture of the start and end state.

### Rule 2 — Simplicity First
Choose the simplest solution that solves the actual problem. Avoid over-engineering, unnecessary abstractions, and premature generalization. If a task is a one-liner, write a one-liner.

### Rule 3 — Surgical Changes
Make targeted, minimal edits. Edit only what's necessary. Do not rewrite entire files unless the task explicitly requires it. If a function is broken, fix the function — not the module.

### Rule 4 — Goal-Driven Execution
Every action must serve a clear goal. When the goal is reached, stop. Do not refactor adjacent code "while you're here," do not add logging "for debugging," do not leave half-finished work.

### Rule 5 — Deterministic First (v2)
Prefer code with predictable output over clever shortcuts. Plain logic over cleverness, explicit over implicit. If the same input always produces the same output, the code is correct by construction.

### Rule 6 — Declare Budgets (v2)
When the task has constraints (time, scope, complexity), state them upfront before diving in. E.g., "this fix should take under 15 minutes and touch ≤2 files." Stay within budget.

### Rule 7 — Human-in-the-Loop (v2)
For destructive or hard-to-reverse actions (git reset --hard, dropping tables, force-push), always confirm with the user before proceeding. Irreversible actions require explicit authorization — one approval does not imply a blank check.

### Rule 8 — Schema Validation (v2)
When processing structured input (JSON, API responses, parsed data), validate at system boundaries. Trust internal code. Never silently swallow missing fields — surface them as clear errors.

### Rule 9 — Sanitize Input (v2)
Treat all external input as untrusted. This project handles user-uploaded PLY files: validate format, check bounds, never assume correct structure. Bad input should fail loudly, not produce garbage silently.

### Rule 10 — Log Rejections Silently (v2)
When input is rejected (bad file type, missing field, invalid parameter), log it for debugging but surface a friendly user-facing message. Do not dump raw exceptions to the user.

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Point cloud encoder | PointNet++ (Set Abstraction + Feature Propagation), `src/models/point_encoder.py` |
| Tokenizer | VQ-VAE, 2048-codebook, 128-dim embeddings, `src/models/tokenizer.py` |
| LLM projector | Linear + LayerNorm + GELU, 128→4096 dim, `src/models/llm_projector.py` |
| Language model | GLM-4-Flash via `zhipuai` SDK (智谱AI cloud API) |
| Backend | FastAPI + uvicorn |
| Frontend | Three.js (ES modules via importmap), vanilla HTML/JS |
| Point cloud lib | Open3D |
| Environment | Python 3.12, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 4060 Laptop GPU (8GB) |

---

## API Endpoints

```
python src/api/main.py   # → http://localhost:8000

GET  /                  health check, version info
GET  /web               serve web frontend
POST /extract          upload point cloud → rule-based parameter extraction
POST /pointllm          upload point cloud → PointLLM encode → GLM analyze
POST /chat              text-only question → GLM
POST /report            multi-tree → survey report via GLM
POST /encode-chat-upload  upload → PointLLM encode → GLM Q&A (main pipeline)
```

---

## Known Quirks

### Torch import chain
`src/models/point_llm.py` imports `point_encoder.py` → `tokenizer.py` → `llm_projector.py`. All import torch. **Never** put these imports at the top of `src/api/*.py` — use lazy import inside endpoint functions to avoid blocking the server when torch is unavailable. Pattern:
```python
def endpoint():
    # lazy — don't put at module top
    from src.models.point_llm import PointLLMForTrees
    ...
```

### PLY binary parsing (web frontend)
The browser parses binary PLY files. `web/index.html` has a custom `parsePlyBuffer()` that reads from DataView. **Stride must be computed from header property types** — `double`=8 bytes, `float`=4, `uchar`=1. The test file (`points3D.ply`) has 40-byte stride (8+8+8+1+1+1+1+4+4+4) for double xyz + uchar rgb + uchar pad + float normals. Always parse the property list from the header, don't assume fixed layout.

### GLM API key
Set as environment variable before server start:
```bash
set ZHIPUAI_API_KEY=32871b74afe147af83edfe74281edaaf.EyDpmMOAPjS85vJI
python src/api/main.py
```
Fallback env var: `GLM_API_KEY`.

### PLY point cloud test file
`H:\1reserch\02lidarsplatting\data\tree2\colmap_runbasicsfm\sparse\0\points3D.ply` — 19,497 points, binary_little_endian 1.0, double xyz, uchar rgb/pad, float normals. Good for end-to-end pipeline testing.

---

## Folder Structure

```
H:\1reserch\03TREES-LLM\
├── web/                      Web frontend (Three.js + chat)
│   └── index.html
├── src/
│   ├── models/               PointLLM core (PointNet++ encoder, VQ-VAE, projector)
│   │   ├── point_encoder.py
│   │   ├── tokenizer.py
│   │   ├── llm_projector.py
│   │   └── point_llm.py
│   ├── api/                  FastAPI endpoints
│   │   ├── main.py
│   │   ├── services.py
│   │   └── tokenizer_service.py
│   ├── data/
│   │   └── preprocessing.py  Ground filter, clustering, param extraction
│   └── scripts/
│       ├── inference.py
│       └── eval_metrics.py
├── literature/               Paper PDFs, literature_database.xlsx
├── planning/                 Research plan, weekly progress
├── competition/              Contest materials
├── resources/prompts/        Prompt templates
├── CLAUDE.md                 This file
└── .gitignore
```

---

## Development Commands

```bash
# Start backend
set ZHIPUAI_API_KEY=<key>
python src/api/main.py

# Test data
python -c "from src.data.preprocessing import compute_tree_params; import numpy as np; pts = np.load('path/to.npy'); print(compute_tree_params(pts))"

# Web frontend test
curl http://localhost:8000/web
curl http://localhost:8000/health

# Kill stale server
taskkill /F /FI "WINDOWTITLE eq *python*8000*" 2>nul
```

---

## Change Log

| Date | Change |
|------|--------|
| 2026-04-27 | Project initialized |
| 2026-05-12 | Integrated scheme A/B into scheme C, built full code scaffold |
| 2026-05-16 | **Major refactor**: switched from 3DCity-LLM to PointLLM route, deleted old models, rebuilt core modules, fixed PLY binary parser, integrated karpathy-skills |