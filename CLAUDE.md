# TREES-LLM — Claude Code Project Configuration

> Project context loaded on every session start. Update this file when research direction or folder structure changes.
> **Last updated:** 2026-05-16 (架构重构后)

---

## Project Overview

**Name:** TREES-LLM (Tree Understanding and Reasoning Enhanced System)
**Domain:** Surveying & Remote Sensing / Point Cloud Processing / Forestry / Multimodal LLM
**Core:** PointLLM (ECCV 2024) route — Web platform where users upload MLS point clouds and a LLM directly understands 3D scenes to segment trees, extract parameters (DBH, height, crown, volume), answer questions, and generate reports.
**Stage:** Development (PointLLM skeleton complete, integration testing)

**Tech route:** ULIP-2 PointBERT Encoder → VQ-VAE Tokenizer → token文本化 → GLM-4-Flash (cloud API, zero training)

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
| Point cloud encoder | ULIP-2 PointBERT (Objaverse pretrained, 768-dim), fallback: TreePointEncoder |
| Tokenizer | VQ-VAE, 2048-codebook, 128-dim embeddings, `src/models/tokenizer.py` |
| Language model | GLM-4-Flash via `zhipuai` SDK (智谱AI cloud API) |
| Backend | FastAPI + uvicorn |
| Frontend | Three.js (ES modules via importmap), vanilla HTML/JS |
| Point cloud lib | Open3D |
| Environment | Python 3.12, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 4060 Laptop GPU (8GB) |

---

## API Endpoints

```
python src/api/main.py   # → http://localhost:8000

GET  /                health check, version info
GET  /web             serve web frontend
GET  /health          health check
POST /analyze         upload point cloud + question → 精确参数 + LLM语义回答 (核心端点)
POST /multi-analyze   多树分割分析 (DBSCAN + 逐树参数)
POST /ask             自然语言问答 (基于缓存, 非流式)
POST /ask/stream      自然语言问答 (SSE流式输出)
POST /report          完整调查报告 (standard/detailed/carbon)
POST /recommend-params 自动参数推荐
GET  /cache-status    缓存状态
DELETE /cache         清空缓存
```

---

## Known Quirks

### Torch import chain
`src/models/point_llm.py` imports `point_encoder.py` → `tokenizer.py`. Both import torch. **Never** put these imports at the top of `src/api/*.py` — use lazy import inside endpoint functions to avoid blocking the server when torch is unavailable. Pattern:
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
│   ├── models/               PointLLM core
│   │   ├── pretrained_encoder.py  ULIP-2 PointBERT (768-dim)
│   │   ├── tokenizer.py           VQ-VAE (2048 codebook)
│   │   ├── point_encoder.py      fallback encoder
│   │   └── point_llm.py          统一封装 (encoder + tokenizer)
│   ├── api/                  FastAPI endpoints
│   │   ├── main.py            路由（含流式SSE端点）
│   │   ├── service.py         唯一业务服务（统一入口）
│   │   ├── schemas.py         Pydantic模型（边界校验）
│   │   ├── cache.py           分析结果缓存（LRU, 线程安全）
│   │   ├── intent.py          意图分类器（10种意图）
│   │   ├── decision_engine.py 决策推理引擎（报告+问答）
│   │   ├── interpreters.py    树木参数语义解读
│   │   └── parameter_advisor.py 参数自动推荐
│   ├── data/
│   │   ├── preprocessing.py  ⚠️ DEPRECATED
│   │   └── forest_knowledge.py 林业知识库（林分类型+管理规则+生物量公式）
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
set ZHIPUAI_API_KEY=32871b74afe147af83edfe74281edaaf.EyDpmMOAPjS85vJI
python src/api/main.py

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
| 2026-05-16 | **架构重构v2**: 合并双Service为TreeAnalysisService，删除7个冗余端点，删除llm_projector.py，删除tokenizer_service.py/services.py，标记preprocessing.py为DEPRECATED，更新CLAUDE.md |
| 2026-05-17 | **LLM语义层+流式输出**: 新增intent/cache/decision_engine/interpreters/parameter_advisor/forest_knowledge模块，修复height_range序列化bug，添加SSE流式问答(/ask/stream)，前端流式渲染 |