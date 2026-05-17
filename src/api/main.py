"""
FastAPI 后端入口 (v0.3)
端点：
- GET  /health          健康检查
- GET  /web             Web前端
- POST /analyze         单树分析 + 缓存（核心端点）
- POST /multi-analyze   多树分割分析
- POST /ask             自然语言问答（基于缓存）
- POST /report          生成完整调查报告
- POST /recommend-params 参数推荐
- GET  /cache-status    缓存状态
- DELETE /cache         清空缓存
"""

import os
import sys
from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api.schemas import (
    AnalyzeResponse,
    AskResponse,
    MultiAnalyzeResponse,
    ReportResponse,
    RecommendParamsResponse,
    CacheStatusResponse,
)
from src.api.service import TreeAnalysisService

app = FastAPI(
    title="TREES-LLM API",
    description="林业点云智能分析服务 — 传统算法做精确计算 + LLM做语义理解和自然语言交互",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
tree_service: TreeAnalysisService | None = None


@app.on_event("startup")
async def startup():
    global tree_service
    if not os.environ.get("ZHIPUAI_API_KEY") and not os.environ.get("GLM_API_KEY"):
        os.environ.setdefault("ZHIPUAI_API_KEY", "32871b74afe147af83edfe74281edaaf.EyDpmMOAPjS85vJI")
    tree_service = TreeAnalysisService()


# ============ 基础端点 ============

@app.get("/")
async def root():
    return {
        "message": "TREES-LLM API v0.3",
        "version": "0.3.0",
        "description": "林业点云智能分析 — 传统算法 + LLM语义理解",
        "docs": "/docs",
        "endpoints": {
            "/health": "健康检查",
            "/web": "Web前端",
            "/analyze": "单树分析（核心）",
            "/multi-analyze": "多树分割分析",
            "/ask": "自然语言问答",
            "/report": "生成完整报告",
            "/recommend-params": "参数推荐",
            "/cache-status": "缓存状态",
            "/cache": "清空缓存",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.3.0",
        "service": "TreeAnalysisService",
    }


@app.get("/web")
async def serve_web():
    web_path = os.path.join(project_root, "web", "index.html")
    if os.path.exists(web_path):
        return FileResponse(web_path)
    return JSONResponse({"error": "Web frontend not found"}, status_code=404)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# ============ 核心业务端点 ============

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_point_cloud(
    file: UploadFile = File(...),
    question: str = Form(default=""),
):
    """
    单树分析端点（核心）
    上传点云 + 可选问题 → 传统算法提取精确参数 + LLM语义回答

    自动缓存分析结果，后续可通过 /ask 进行自然语言问答
    """
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in (".ply", ".pcd", ".las", ".laz", ".npy"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp.write(content)
            tmp_path = tmp.name

        result = tree_service.analyze(tmp_path, question)
        os.unlink(tmp_path)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi-analyze", response_model=MultiAnalyzeResponse)
async def multi_analyze_point_cloud(
    file: UploadFile = File(...),
    eps: float = Form(default=0.5, description="DBSCAN聚类半径（米）"),
    min_samples: int = Form(default=50, description="DBSCAN最小点数"),
):
    """
    多树分割分析端点
    上传点云 → DBSCAN空间聚类分割 → 逐树提取参数 → 林分整体分析

    适用于MLS场景（包含多棵树木的点云）
    """
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in (".ply", ".pcd", ".las", ".laz", ".npy"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp.write(content)
            tmp_path = tmp.name

        result = tree_service.multi_analyze(tmp_path, eps=eps, min_samples=min_samples)
        os.unlink(tmp_path)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Multi-analyze failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    question: str = Form(...),
    file_key: str = Form(default=None),
):
    """
    自然语言问答端点
    基于已缓存的分析结果，回答用户关于林分的问题

    意图识别：分析/健康评估/风险识别/碳汇评估/管理建议/通用问答
    """
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not question or len(question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        result = tree_service.ask(question=question, file_key=file_key)

        if not result.get("success") and result.get("error") == "no_cached_data":
            raise HTTPException(status_code=400, detail="请先上传点云进行分析")

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_question_stream(
    question: str = Form(...),
    file_key: str = Form(default=None),
):
    """
    流式问答端点（SSE）
    返回Server-Sent Events格式的流式LLM输出
    """
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not question or len(question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question is required")

    def event_stream():
        try:
            for token in tree_service.stream_answer(question=question, file_key=file_key):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/report", response_model=ReportResponse)
async def generate_report(
    trees_data_str: str = Form(default=None, description="JSON序列化的树木参数列表"),
    report_type: str = Form(default="standard"),
    file_key: str = Form(default=None),
):
    """
    报告生成端点
    基于树木参数 → LLM生成完整的专业调查报告

    支持3种报告类型：
    - standard: 标准报告（概览+统计+管理建议）
    - detailed: 详细报告（含每棵树详情）
    - carbon: 碳汇专项报告（含碳储量排名+碳汇价值）
    """
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    import json

    trees_data = None
    if trees_data_str:
        try:
            trees_data = json.loads(trees_data_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid trees_data JSON")

    try:
        result = tree_service.generate_report(
            trees_data=trees_data,
            report_type=report_type,
            file_key=file_key,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend-params", response_model=RecommendParamsResponse)
async def recommend_params(file: UploadFile = File(...)):
    """
    参数推荐端点
    根据点云场景特征，自动推荐最佳处理参数（聚类参数、生物量公式等）
    """
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in (".ply", ".pcd", ".las", ".laz", ".npy"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp.write(content)
            tmp_path = tmp.name

        result = tree_service.recommend_params(tmp_path)
        os.unlink(tmp_path)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 缓存管理端点 ============

@app.get("/cache-status", response_model=CacheStatusResponse)
async def cache_status():
    """获取缓存状态"""
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return tree_service.get_cache_status()


@app.delete("/cache")
async def clear_cache():
    """清空缓存"""
    if not tree_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return tree_service.clear_cache()


# ============ 启动 ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )