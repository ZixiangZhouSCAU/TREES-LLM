# TREES-LLM — Claude Code Project Configuration

> Project context loaded on every session start. Update this file when research direction or folder structure changes.
> **Last updated:** 2026-05-20 (路线A架构对齐 + RAG林业专家)

---

## Project Overview

**Name:** TREES-LLM (Tree Understanding and Reasoning Enhanced System)
**Domain:** Surveying & Remote Sensing / Point Cloud Processing / Forestry / Multimodal LLM
**Core:** PointLLM (ECCV 2024) + 路线 A 架构对齐（参考 3DCITY-LLM 粗到细编码）— Web platform where users upload MLS point clouds and a LLM directly understands 3D scenes to segment trees, extract parameters (DBH, height, crown, volume), answer questions, and generate reports.

**Stage:** Development (PointLLM skeleton complete, Route A architecture aligned, RAG knowledge base initialized)

**Tech route:** ULIP-2 PointBERT Encoder → VQ-VAE Tokenizer → token文本化 → GLM-4-Flash (cloud API, zero training)

**路线 A（架构对齐 3DCITY-LLM）：**
- 三层编码器（Object / Relationship / Scene）+ 指令驱动路由
- FeatureProjector 训练（~6M 参数，RTX 4060 可训）
- RAG 林业专家知识库（FAISS + sentence-transformers）

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
| Three-layer encoder | Object/Relationship/Scene encoding (`src/models/tree_encoder.py`) |
| Feature projector | 3× Projector (~6M trainable params, `src/models/tree_encoder.py`) |
| Tokenizer | VQ-VAE, 2048-codebook, 128-dim embeddings, `src/models/tokenizer.py` |
| Language model | GLM-4-Flash via `zhipuai` SDK (智谱AI cloud API) |
| RAG | FAISS + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Knowledge base | 6× Markdown docs (`resources/knowledge/`) |
| Backend | FastAPI + uvicorn |
| Frontend | Three.js (ES modules via importmap), vanilla HTML/JS |
| Point cloud lib | Open3D + sklearn (DBSCAN) |
| Environment | Python 3.12, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 4060 Laptop GPU (8GB) |

---

## API Endpoints

```
python src/api/main.py   # → http://localhost:8000

GET  /                health check, version info
GET  /web             serve web frontend
GET  /health          health check
GET  /rag-status      RAG 知识库状态
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
│   ├── models/               PointLLM core + 三层编码器
│   │   ├── pretrained_encoder.py  ULIP-2 PointBERT (768-dim)
│   │   ├── tree_encoder.py        三层编码器 (Object/Relationship/Scene)
│   │   ├── tokenizer.py           VQ-VAE (2048 codebook)
│   │   ├── point_encoder.py      fallback encoder
│   │   └── point_llm.py          统一封装 (encoder + tokenizer)
│   ├── api/                  FastAPI endpoints
│   │   ├── main.py            路由（含流式SSE端点 + /rag-status）
│   │   ├── service.py         唯一业务服务（统一入口）
│   │   ├── schemas.py         Pydantic模型（边界校验）
│   │   ├── cache.py           分析结果缓存（LRU, 线程安全）
│   │   ├── intent.py          意图分类器（11种意图 + 三层路由）
│   │   ├── decision_engine.py 决策推理引擎（RAG + 三层编码）
│   │   ├── rag.py             RAG 检索系统（FAISS + embedding）
│   │   ├── knowledge_base.py  知识库加载（Markdown → chunks → FAISS）
│   │   ├── interpreters.py    树木参数语义解读
│   │   └── parameter_advisor.py 参数自动推荐
│   ├── training/             训练管道
│   │   ├── dataset.py        训练数据集加载器（JSONL）
│   │   └── trainer.py        两阶段训练器（Stage 1/2）
│   ├── data/
│   │   └── forest_knowledge.py 林业知识库（林分类型+管理规则+生物量公式）
│   ├── scripts/
│   │   ├── inference.py      推理脚本（已修复）
│   │   └── eval_metrics.py   评估指标
│   └── configs/
│       └── inference.yaml    推理配置（GLM-4-Flash）
├── scripts/                  独立脚本
│   ├── generate_training_data.py  训练数据生成
│   └── build_knowledge_index.py   RAG 索引构建
├── data/
│   ├── training/             训练数据 (JSONL)
│   └── knowledge_index/      FAISS 索引
├── resources/
│   ├── prompts/              Prompt templates
│   └── knowledge/            RAG 林业专家知识库
│       ├── 01_树种数据库.md
│       ├── 02_碳储量计算规范.md
│       ├── 03_林分管理指南.md
│       ├── 04_MLS参数提取规范.md
│       ├── 05_林业法规标准.md
│       └── 06_Q&A专家问答集.md
├── literature/               Paper PDFs, literature_database.xlsx
├── planning/                 Research plan, training plan, progress
├── competition/              Contest materials
├── CLAUDE.md                 This file
└── .gitignore
```

---

## Development Commands

```bash
# 启动后端
set ZHIPUAI_API_KEY=32871b74afe147af83edfe74281edaaf.EyDpmMOAPjS85vJI
python src/api/main.py

# 构建 RAG 知识库索引（首次使用或更新知识后）
python scripts/build_knowledge_index.py

# 生成训练数据（数据采集完成后运行）
python scripts/generate_training_data.py \
    --data-dir data/collected \
    --metadata data/tree_metadata.json \
    --output data/training/tree_training_data.jsonl

# 训练 projector（Stage 1 + Stage 2）
python src/training/trainer.py --stage 3 --epochs 10 --lr 1e-3

# 快速推理（命令行）
python src/scripts/inference.py --input test.ply --question "这棵树的碳储量有多少？"

# Web 前端测试
curl http://localhost:8000/web
curl http://localhost:8000/health
curl http://localhost:8000/rag-status

# 杀掉卡住的服务器
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
| 2026-05-20 | **路线A架构对齐 + RAG林业专家**: 新增三层编码器(tree_encoder.py)，新增RAG系统(rag.py + knowledge_base.py)，新增训练管道(dataset.py + trainer.py)，新增6个知识库文档框架，修复inference.py，更新intent.py三层路由，新增/rag-status端点 |