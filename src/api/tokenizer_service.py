"""
Tokenizer Service - PointLLM风格点云理解服务
整合 PointLLMEncoder + VQ-Tokenizer + GLM-4-Flash
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Optional

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from zhipuai import ZhipuAI
    HAS_ZHIPUAI = True
except ImportError:
    HAS_ZHIPUAI = False


def load_point_cloud(file_path: str) -> np.ndarray:
    """加载点云文件"""
    if file_path.endswith((".ply", ".pcd")):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points).astype(np.float32)
    elif file_path.endswith((".npy",)):
        return np.load(file_path)
    elif file_path.endswith((".las", ".laz")):
        import laspy
        las = laspy.read(file_path)
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {file_path}")


class TokenizerService:
    """
    PointLLM Token化 + GLM推理服务
    Pipeline: 点云 → PointEncoder → VQ-Tokenizer → GLM
    """

    def __init__(self):
        self.llm_client = None
        if HAS_ZHIPUAI:
            api_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
            if api_key:
                self.llm_client = ZhipuAI(api_key=api_key)

        # PointLLM模型
        self.pointllm = None
        try:
            from src.models.point_llm import PointLLMForTrees
            self.pointllm = PointLLMForTrees(device="cpu")
        except Exception as e:
            print(f"[TokenizerService] PointLLM加载失败: {e}")

    def tokenize_file(self, file_path: str) -> Dict:
        """对点云文件做完整Token化"""
        points = load_point_cloud(file_path)

        if self.pointllm:
            analysis = self.pointllm.analyze(points)
            description = self.pointllm.to_description(analysis)
        else:
            # Fallback: 简单几何分析
            z = points[:, 2]
            description = (
                f"点数: {len(points)}, "
                f"高度范围: {z.min():.2f}~{z.max():.2f}m, "
                f"高度: {z.max()-z.min():.2f}m"
            )
            analysis = None

        result = {
            "success": True,
            "n_input_points": len(points),
            "description": description,
        }
        if analysis:
            result["geometry"] = analysis.get("geometry", {})
            result["layers"] = analysis.get("layers", {})

        return result

    def tokenize_and_chat(self, file_path: str, question: str) -> Dict:
        """Token化 + GLM问答"""
        points = load_point_cloud(file_path)

        # PointLLM分析
        if self.pointllm:
            analysis = self.pointllm.analyze(points)
            description = self.pointllm.to_description(analysis)

            # 补充规则参数
            import importlib.util, sys as _sys
            def import_from_path(name, path):
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                _sys.modules[name] = mod
                spec.loader.exec_module(mod)
                return mod
            preproc = import_from_path("preproc", "src/data/preprocessing.py")
            params = preproc.compute_tree_params(points.astype(np.float32))
            analysis["params"] = params
        else:
            description = f"点数: {len(points)}"
            analysis = None
            params = {}

        # 构建prompt
        prompt = f"""你是一位资深林业专家，擅长分析LiDAR点云数据。
你收到了PointLLM树木点云分析结果，包含3D几何分析。

{description}

用户问题: {question}

请用专业但易懂的语言回答，适当引用3D几何数据。"""

        # 调用GLM
        if self.llm_client and analysis:
            resp = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            answer = resp.choices[0].message.content
        else:
            answer = f"基于点云分析：共{len(points)}点。树高约{params.get('height', 0):.2f}m。"

        return {
            "success": True,
            "answer": answer,
            "description": description,
            "n_voxels": (analysis.get("geometry", {}).get("n_points", len(points)) if analysis else len(points)),
        }

    def _fallback_chat(self, description: str, params: dict, question: str) -> str:
        return (
            f"基于PointLLM分析：树高{params.get('height', 0):.2f}m，"
            f"胸径{params.get('dbh_estimate', 0)*100:.1f}cm，冠幅{params.get('crown_width', 0):.2f}m。"
        )
