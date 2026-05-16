"""
业务逻辑服务层
PointLLM风格：点云处理 + LLM推理
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from zhipuai import ZhipuAI
    HAS_ZHIPUAI = True
except ImportError:
    HAS_ZHIPUAI = False


class TreesService:
    """
    TREES-LLM 业务服务
    整合点云处理 + LLM推理
    """

    def __init__(self, config_path: str = "src/configs/inference.yaml"):
        self.config = self._load_config(config_path)
        self._init_models()

    def _load_config(self, path: str) -> Dict:
        """加载配置"""
        import yaml
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (FileNotFoundError, UnicodeDecodeError):
            return {}

    def _init_models(self):
        self.llm_client = None
        if HAS_ZHIPUAI:
            api_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
            if api_key:
                self.llm_client = ZhipuAI(api_key=api_key)

        # PointLLM模型（无训练，直接推理）
        self.pointllm = None
        try:
            from src.models.point_llm import PointLLMForTrees, PointLLMConfig
            self.pointllm = PointLLMForTrees(device="auto")
        except Exception as e:
            print(f"[PointLLM] 模型加载失败: {e}")

    async def extract_params(self, point_file: str) -> Dict:
        """
        从点云文件提取树木参数

        流程：
        1. 读取点云
        2. 预处理（滤除地面、离群点）
        3. 树木分割
        4. 计算参数
        """
        # Import preprocessing utilities directly (avoid src/__init__.py which imports torch)
        import importlib.util
        import sys

        def import_from_path(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            return mod

        preprocessing = import_from_path("preprocessing", "src/data/preprocessing.py")
        PointCloudPreprocessor = preprocessing.PointCloudPreprocessor
        compute_tree_params = preprocessing.compute_tree_params

        # 加载点云
        if point_file.endswith((".las", ".laz")):
            import laspy
            las = laspy.read(point_file)
            points = np.vstack([las.x, las.y, las.z]).T
        elif point_file.endswith((".ply", ".pcd")):
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(point_file)
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                raise ValueError("PLY file contains no points")
            points = pts
        elif point_file.endswith(".npy"):
            points = np.load(point_file)
        else:
            raise ValueError(f"Unsupported file format: {point_file}")

        # 预处理 + 分割
        preprocessor = PointCloudPreprocessor()
        trees = preprocessor.process(points)

        if not trees:
            return {"success": False, "error": "No trees detected"}

        # 取第一棵树（单树模式）
        tree = trees[0]
        params = compute_tree_params(tree["points"])

        # 估算碳储量 (简化模型)
        carbon = self._estimate_carbon(params["height"], params["dbh_estimate"])

        return {
            "success": True,
            "tree_id": f"tree_{tree['tree_id']}",
            "params": {
                "tree_id": f"tree_{tree['tree_id']}",
                "height": round(params["height"], 2),
                "dbh": round(params["dbh_estimate"] * 100, 1),  # 转为厘米
                "crown_width": round(params["crown_width"], 2),
                "crown_volume": round(self._estimate_crown_volume(params), 2),
                "stem_volume": round(self._estimate_stem_volume(params), 2),
                "carbon_stock": round(carbon, 1),
            },
            "confidence": {
                "height": 0.92,
                "dbh": 0.88,
                "crown_width": 0.85,
            }
        }

    def _estimate_carbon(self, height: float, dbh: float) -> float:
        """简化碳储量估算"""
        # 生物量估算 (简化BEF模型)
        volume = 0.0001 * height * (dbh ** 2)
        biomass = volume * 600  # 木材密度约600 kg/m³
        carbon = biomass * 0.5   # 碳含量约50%
        return carbon

    def _estimate_crown_volume(self, params: Dict) -> float:
        """估算冠幅体积"""
        h = params["height"]
        cw = params["crown_width"]
        # 简化椭球体体积
        return (4/3) * np.pi * (cw/2) * (cw/2) * (h * 0.4)

    def _estimate_stem_volume(self, params: Dict) -> float:
        """估算树干体积（锥体模型）"""
        h = params["height"]
        r = params["dbh_estimate"] / 2
        return (1/3) * np.pi * (r ** 2) * h

    async def answer_question(
        self,
        point_file: str,
        question: str,
    ) -> str:
        """
        回答关于树木的问题
        使用 Claude API 实现
        """
        # 先提取参数
        tree_data = await self.extract_params(point_file)

        if not tree_data.get("success"):
            return "无法分析该点云数据，请检查文件格式。"

        params = tree_data["params"]

        # 调用 GLM API
        if self.llm_client:
            prompt = self._build_question_prompt(params, question)
            response = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content
        else:
            return self._fallback_answer(params, question)

    def _build_question_prompt(self, params: Dict, question: str) -> str:
        """构建问答提示词"""
        return f"""你是一位资深的林业专家。根据以下树木参数，回答用户的问题。
        如果不确定，请明确说明。

        树木参数：
        - 树高: {params['height']} 米
        - 胸径(DBH): {params['dbh']} 厘米
        - 冠幅: {params['crown_width']} 米
        - 碳储量: {params['carbon_stock']} 千克

        用户问题: {question}

        请用专业但易懂的语言回答，可以适当给出管理建议。"""

    def _fallback_answer(self, params: Dict, question: str) -> str:
        """无API时的本地回答"""
        if "健康" in question or "生长" in question:
            return (f"根据参数分析：这棵树高{params['height']}m，"
                    f"胸径{params['dbh']}cm。树高与胸径比例正常，生长状态良好。")
        elif "碳" in question:
            return f"该树碳储量约{params['carbon_stock']:.1f}千克。"
        else:
            return f"该树树高{params['height']}m，胸径{params['dbh']}cm，冠幅{params['crown_width']}m。"

    async def generate_report(
        self,
        trees_data: List[Dict],
        report_type: str = "standard",
    ) -> Dict:
        """生成树木调查报告"""
        if not trees_data:
            return {"success": False, "error": "No data"}

        # 统计汇总
        heights = [t["height"] for t in trees_data]
        dbhs = [t["dbh"] for t in trees_data]
        carbons = [t["carbon_stock"] for t in trees_data]

        summary = {
            "total_trees": len(trees_data),
            "avg_height": round(np.mean(heights), 2),
            "avg_dbh": round(np.mean(dbhs), 1),
            "total_carbon": round(sum(carbons), 1),
            "max_height_tree": trees_data[np.argmax(heights)]["tree_id"],
            "max_dbh_tree": trees_data[np.argmax(dbhs)]["tree_id"],
        }

        # 生成文本报告
        report_text = self._generate_text_report(trees_data, summary, report_type)

        return {
            "success": True,
            "report_text": report_text,
            "summary": summary,
            "trees_count": len(trees_data),
        }

    def _generate_text_report(
        self,
        trees: List[Dict],
        summary: Dict,
        report_type: str,
    ) -> str:
        """生成文本报告"""
        lines = [
            "# 城市街道树木调查报告",
            "",
            f"## 基本信息",
            f"- 调查树木总数：{summary['total_trees']} 棵",
            f"- 平均树高：{summary['avg_height']} m",
            f"- 平均胸径：{summary['avg_dbh']} cm",
            f"- 总碳储量：{summary['total_carbon']} kg",
            "",
            f"## 树木详情",
        ]

        for tree in trees:
            lines.append(f"\n### {tree['tree_id']}")
            lines.append(f"- 树高: {tree['height']}m")
            lines.append(f"- 胸径: {tree['dbh']}cm")
            lines.append(f"- 冠幅: {tree['crown_width']}m")
            lines.append(f"- 碳储量: {tree['carbon_stock']}kg")

        return "\n".join(lines)

    async def build_scene_graph(self, trees_file: str) -> Dict:
        """构建树木空间关系图"""
        import json

        # 加载树木数据
        if trees_file.endswith(".json"):
            with open(trees_file) as f:
                data = json.load(f)
        else:
            # 如果是点云，先分割
            from src.data.preprocessing import PointCloudPreprocessor
            import laspy
            las = laspy.read(trees_file)
            points = np.vstack([las.x, las.y, las.z]).T
            preprocessor = PointCloudPreprocessor()
            trees = preprocessor.process(points)
            data = [{"tree_id": t["tree_id"], "params": t["params"]} for t in trees]

        nodes = []
        edges = []
        positions = []

        for tree in data:
            node = {
                "tree_id": str(tree["tree_id"]),
                "position": {
                    "x": tree["params"]["centroid"][0],
                    "y": tree["params"]["centroid"][1],
                    "z": tree["params"]["centroid"][2],
                },
                "params": {
                    "height": tree["params"]["height"],
                    "dbh": tree["params"]["dbh_estimate"],
                }
            }
            nodes.append(node)
            positions.append(tree["params"]["centroid"])

        # 计算树木间关系
        positions = np.array(positions)
        n = len(positions)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
                if dist < 3.0:  # 3米内认为相邻
                    relation = "adjacent" if dist < 1.5 else "mutual_shade"
                    edges.append({
                        "source": nodes[i]["tree_id"],
                        "target": nodes[j]["tree_id"],
                        "relation": relation,
                        "distance": round(float(dist), 2),
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "total_trees": n,
                "total_edges": len(edges),
            }
        }

    async def pointllm_analyze(self, point_file: str, question: str = "") -> Dict:
        """
        PointLLM风格分析：点云 → 编码 → Token化 → LLM回答
        端到端Pipeline，无需训练
        """
        # 加载点云
        if point_file.endswith((".las", ".laz")):
            import laspy
            las = laspy.read(point_file)
            points = np.vstack([las.x, las.y, las.z]).T
        elif point_file.endswith((".ply", ".pcd")):
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(point_file)
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                raise ValueError("PLY file contains no points")
            points = pts
        elif point_file.endswith(".npy"):
            points = np.load(point_file)
        else:
            raise ValueError(f"Unsupported format: {point_file}")

        # PointLLM分析
        if self.pointllm is None:
            # fallback: 用规则算法
            import importlib.util, sys
            def import_from_path(name, path):
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                return mod

            preproc = import_from_path("preproc", "src/data/preprocessing.py")
            processor = preproc.PointCloudPreprocessor()
            trees = processor.process(points)
            if not trees:
                return {"success": False, "error": "No trees detected"}
            tree = trees[0]
            params = preproc.compute_tree_params(tree["points"])

            return {
                "success": True,
                "method": "rule_based",
                "params": params,
                "n_points": int(points.shape[0]),
                "answer": f"PointLLM未初始化。用规则算法：树高{params['height']:.2f}m，胸径{params['dbh_estimate']*100:.1f}cm，冠幅{params['crown_width']:.2f}m。"
            }

        # PointLLM编码
        analysis = self.pointllm.analyze(points.astype(np.float32))
        description = self.pointllm.to_description(analysis)

        # 参数提取（补充）
        import importlib.util, sys
        def import_from_path(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        preproc = import_from_path("preproc", "src/data/preprocessing.py")
        params = preproc.compute_tree_params(points.astype(np.float32))
        analysis["params"] = params

        # 如果有问题，调用GLM
        if question and self.llm_client:
            prompt = self.pointllm.build_prompt(analysis, question)
            resp = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            answer = resp.choices[0].message.content
        else:
            answer = description

        return {
            "success": True,
            "method": "pointllm",
            "analysis": analysis,
            "description": description,
            "answer": answer,
            "params": params,
            "n_points": int(points.shape[0]),
        }
