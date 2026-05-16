"""
FastAPI 后端入口
提供树木参数提取API、智能问答API、报告生成API
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict


class ChatRequest(BaseModel):
    question: str
    params: Optional[Dict] = None
import numpy as np
import tempfile

from src.api.schemas import (
    TreeParamsResponse,
    QuestionRequest,
    ReportRequest,
    SceneGraphResponse,
)
from src.api.services import TreesService
from src.api.tokenizer_service import TokenizerService


app = FastAPI(
    title="TREES-LLM API",
    description="树木领域大语言模型推理服务",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve web frontend
web_dir = os.path.join(project_root, "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/web")
async def serve_web():
    web_path = os.path.join(project_root, "web", "index.html")
    if os.path.exists(web_path):
        return FileResponse(web_path)
    return {"error": "Web frontend not found"}

# 全局服务实例
trees_service: Optional[TreesService] = None
tokenizer_service: Optional[TokenizerService] = None


@app.on_event("startup")
async def startup():
    global trees_service, tokenizer_service
    trees_service = TreesService()
    tokenizer_service = TokenizerService()


# ============ API端点 ============

@app.get("/")
async def root():
    return {"message": "TREES-LLM API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/extract", response_model=TreeParamsResponse)
async def extract_tree_params(file: UploadFile = File(...)):
    """
    上传点云文件，提取单棵树参数
    支持 .las, .laz, .npy 格式
    """
    if not trees_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # 保存上传文件
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 处理
        result = await trees_service.extract_params(tmp_path)

        # 清理
        os.unlink(tmp_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    智能问答：输入点云文件和问题，返回回答
    """
    if not trees_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await trees_service.answer_question(
            point_file=request.point_cloud_path,
            question=request.question,
        )
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    直接问答：传入参数+问题，返回GLM回答
    """
    if not trees_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        import importlib.util, sys
        def import_from_path(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        svc_mod = import_from_path("svc", str(Path(__file__).parent / "services.py"))
        TreesService2 = svc_mod.TreesService
        svc = TreesService2()

        params = request.params or {}
        prompt = f"""你是一位资深林业专家。根据以下树木参数，回答用户的问题。如果不确定，请明确说明。

树木参数：
- 树高: {params.get('height', 0)} 米
- 胸径(DBH): {params.get('dbh', 0)} 厘米
- 冠幅: {params.get('crown_width', 0)} 米
- 碳储量: {params.get('carbon_stock', 0)} 千克
- 树冠体积: {params.get('crown_volume', 0)} 立方米
- 树干体积: {params.get('stem_volume', 0)} 立方米

用户问题: {request.question}

请用专业但易懂的语言回答，可以适当给出管理建议。"""

        if svc.llm_client:
            resp = svc.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return {"answer": resp.choices[0].message.content}
        else:
            return {"answer": "GLM API 未配置，请在环境变量中设置 ZHIPUAI_API_KEY"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
async def ask_with_upload(file: UploadFile = File(...), question: str = Form(...)):
    """
    上传点云文件 + 问题 → GLM回答
    """
    if not trees_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        answer = await trees_service.answer_question(tmp_path, question)
        os.unlink(tmp_path)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/report", response_model=Dict)
async def generate_report(request: ReportRequest):
    """
    生成树木调查报告
    """
    if not trees_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await trees_service.generate_report(
            trees_data=request.trees_data,
            report_type=request.report_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scene-graph", response_model=SceneGraphResponse)
async def build_scene_graph(trees_file: UploadFile = File(...)):
    """
    构建树木空间关系图
    """
    if not trees_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        suffix = os.path.splitext(trees_file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await trees_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = await trees_service.build_scene_graph(tmp_path)
        os.unlink(tmp_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode")
async def encode_point_cloud(file: UploadFile = File(...)):
    """
    上传点云文件，运行3DCity-LLM风格三分支编码器
    返回体素特征 + 多层次文本描述
    """
    if not tokenizer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = tokenizer_service.tokenize_file(tmp_path)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode-chat")
async def encode_and_chat(request: QuestionRequest):
    """
    3DCity-LLM风格分析：Token化点云 + GLM问答
    让LLM"看见"3D几何结构
    """
    if not tokenizer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = tokenizer_service.tokenize_and_chat(
            request.point_cloud_path,
            request.question,
        )
        return {"answer": result["answer"], "n_voxels": result["n_voxels"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pointllm")
async def pointllm_analyze(file: UploadFile = File(...), question: str = Form("")):
    """
    PointLLM风格分析：上传点云文件 → PointLLM编码 → Token化 → GLM回答
    端到端Pipeline，让LLM"看见"3D结构
    """
    if not trees_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = await trees_service.pointllm_analyze(tmp_path, question)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode-chat-upload")
async def encode_chat_upload(file: UploadFile = File(...), question: str = Form(...)):
    """
    上传文件 + 问题 → Token化 → GLM回答
    """
    if not tokenizer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = tokenizer_service.tokenize_and_chat(tmp_path, question)
        os.unlink(tmp_path)
        return {"answer": result["answer"], "n_voxels": result["n_voxels"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
