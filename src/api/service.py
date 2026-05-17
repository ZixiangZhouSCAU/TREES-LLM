"""
统一服务层 - TreeAnalysisService (v0.3)
解决痛点1-6: 传统算法做计算 + LLM做语义理解和交互

Pipeline: 点云 → 传统算法（精确参数）→ 缓存
                          ↓
                  LLM解读（语义结论）→ 自然语言问答
                          ↓
                  LLM生成报告（一键完整文档）
                          ↓
                  决策引擎（管理建议）
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import re

try:
    from zhipuai import ZhipuAI
    HAS_ZHIPUAI = True
except ImportError:
    HAS_ZHIPUAI = False

# 新增组件
from src.api.intent import classify_intent, Intent, get_intent_description, QUICK_QUESTIONS
from src.api.cache import get_cache, make_file_key, build_cached_analysis, AnalysisCache
from src.api.interpreters import TreeParameterInterpreter
from src.api.parameter_advisor import ParameterAdvisor, quick_recommend
from src.api.decision_engine import DecisionEngine


# ============ 点云加载（统一入口） ============

def load_point_cloud(file_path: str) -> np.ndarray:
    """加载点云文件，统一格式处理"""
    if file_path.endswith((".ply", ".pcd")):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            raise ValueError("PLY file contains no points")
        return pts.astype(np.float32)
    elif file_path.endswith(".npy"):
        data = np.load(file_path)
        return data[:, :3].astype(np.float32) if data.shape[1] >= 3 else data.astype(np.float32)
    elif file_path.endswith((".las", ".laz")):
        import laspy
        las = laspy.read(file_path)
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {file_path}. Supported: .ply .pcd .las .laz .npy")


# ============ 传统算法：精确参数提取 ============

def _to_py(val):
    """将numpy类型或Python类型转换为可JSON序列化的Python原生类型"""
    if hasattr(val, 'item'):  # numpy types
        return val.item()
    return val


def _sanitize_dict(d: Dict) -> Dict:
    """递归清理dict/list中的numpy类型，确保可JSON序列化"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = [
                _sanitize_dict(item) if isinstance(item, dict) else _to_py(item)
                for item in v
            ]
        else:
            result[k] = _to_py(v)
    return result


def extract_precise_params(points: np.ndarray) -> Dict[str, Any]:
    """
    使用传统几何算法提取精确参数
    这是PointLLM真正有价值的地方：精确的测量
    """
    z = points[:, 2]
    x = points[:, 0]
    y = points[:, 1]

    z_min, z_max = float(z.min()), float(z.max())
    height = z_max - z_min

    # 地面/树木分类
    tree_base_z = z_min + height * 0.15
    ground_mask = z < tree_base_z
    tree_mask = ~ground_mask
    ground_count = int(ground_mask.sum())
    tree_count = int(tree_mask.sum())

    # 水平范围
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())

    # 胸径估算（基于树干区域XY扩散）
    tree_z = z[tree_mask]
    trunk_base = z_min + height * 0.1
    trunk_top = z_min + height * 0.5
    trunk_mask = (z >= trunk_base) & (z < trunk_top)
    if trunk_mask.sum() > 10:
        trunk_x = x[trunk_mask]
        trunk_y = y[trunk_mask]
        trunk_std = float(max(np.std(trunk_x), np.std(trunk_y)))
        dbh = float(min(20 + trunk_std * 100, 100))
    else:
        dbh = 0.0

    # 冠幅估算
    crown_width = (x_range + y_range) / 4

    # 体积估算
    stem_volume = height * 3.14159 * (dbh / 200) ** 2
    crown_volume = x_range * y_range * height * 0.25

    # 碳储量估算
    carbon = height * (dbh / 100) ** 2 * 0.5 * 700

    return {
        "height": round(float(height), 2),
        "dbh": round(float(dbh), 1),
        "crown_width": round(float(crown_width), 2),
        "crown_volume": round(float(crown_volume), 2),
        "stem_volume": round(float(stem_volume), 3),
        "carbon_stock": round(float(carbon), 1),
    }


# ============ 传统算法：单木分割 ============

def segment_trees(points: np.ndarray, eps: float = 0.5, min_samples: int = 50) -> List[np.ndarray]:
    """
    基于DBSCAN的空间聚类分割单木
    这是传统算法的核心优势，LLM无法替代
    """
    try:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
        labels = clustering.labels_
        tree_list = []
        for label in set(labels):
            if label >= 0:
                tree_points = points[labels == label]
                if len(tree_points) >= min_samples:
                    tree_list.append(tree_points)
        return tree_list
    except ImportError:
        # 如果没有sklearn，返回整个点云作为单棵树
        return [points] if len(points) > 100 else []


# ============ 唯一业务服务 ============

class TreeAnalysisService:
    """
    TREES-LLM 统一业务服务 (v0.3)

    核心理念：
    - 传统算法做精确计算（参数提取、单木分割）
    - LLM做语义理解和自然语言交互（问答、报告生成）
    - PointLLM做场景识别和语义辅助

    数据流：
    点云上传 → 传统算法提取参数 → 缓存
                                    ↓
                        LLM解读 → 自然语言问答
                                    ↓
                        LLM生成完整报告
                                    ↓
                        决策引擎 → 管理建议
    """

    def __init__(self):
        self._init_llm()
        self._init_pointllm()
        self._init_components()

    # ---- 初始化 ----

    def _init_llm(self):
        self.llm_client = None
        if not HAS_ZHIPUAI:
            print("[TreeAnalysisService] WARNING: zhipuai package not installed")
            return
        api_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
        if not api_key:
            print("[TreeAnalysisService] WARNING: No GLM API key found")
            return
        self.llm_client = ZhipuAI(api_key=api_key)
        print(f"[TreeAnalysisService] GLM client initialized")

    def _init_pointllm(self):
        self.pointllm = None
        try:
            from src.models.point_llm import PointLLMForTrees
            self.pointllm = PointLLMForTrees(device="cpu", use_pretrained=True)
            enc_type = getattr(self.pointllm, '_encoder_type', 'unknown')
            print(f"[TreeAnalysisService] PointLLM loaded, encoder: {enc_type}")
        except Exception as e:
            print(f"[TreeAnalysisService] PointLLM加载失败: {e}")

    def _init_components(self):
        """初始化所有组件"""
        self.interpreter = TreeParameterInterpreter(self.llm_client)
        self.advisor = ParameterAdvisor(self.llm_client)
        self.decision_engine = DecisionEngine(self.llm_client)
        self.cache = get_cache()
        print(f"[TreeAnalysisService] All components initialized")

    # ---- 端点：/analyze → 单树分析 + 缓存 ----

    def analyze(self, file_path: str, question: str = "") -> Dict:
        """
        核心端点：上传点云 + 问题 → 精确参数 + LLM语义回答
        自动缓存分析结果，支持后续自然语言问答

        Args:
            file_path: 点云文件路径
            question: 用户问题（可选，无问题则返回默认分析）

        Returns:
            Dict: 包含精确参数、LLM回答、语义解读
        """
        points = load_point_cloud(file_path)
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()

        # Step 1: 传统算法提取精确参数（最可靠的部分）
        params = extract_precise_params(points)

        # Step 2: PointLLM场景识别（辅助角色）
        pointllm_analysis = {}
        if self.pointllm:
            try:
                pointllm_analysis = self.pointllm.analyze(points)
            except Exception as e:
                print(f"[TreeAnalysisService] PointLLM analysis failed: {e}")

        # Step 3: Ground/Tree分类统计
        height = z_max - z_min
        tree_base_z = z_min + height * 0.15
        ground_count = int((z < tree_base_z).sum())
        tree_count = int((z >= tree_base_z).sum())

        # Step 4: 语义解读（基于精确参数）
        semantic_interp = self.interpreter.interpret_single_tree(
            tree_id="tree_0",
            params=params,
            pointllm_analysis=pointllm_analysis,
        )

        # Step 5: LLM回答（基于语义解读 + 用户问题）
        if not question:
            # 无问题：生成默认分析
            answer = self._llm_describe(params, semantic_interp, pointllm_analysis, ground_count, tree_count)
        else:
            # 有问题：意图分类 + 定向回答
            intent = classify_intent(question)
            answer = self._llm_answer(question, intent, [params], {
                "n_points": len(points),
                "ground_count": ground_count,
                "tree_count": tree_count,
                "avg_height": params["height"],
                "avg_dbh": params["dbh"],
                "total_carbon": params["carbon_stock"],
                "height_range": [z_min, z_max],
            })

        # Step 6: 缓存结果（支持后续问答）
        file_key = make_file_key(file_path.split("/")[-1], len(points))
        cached = build_cached_analysis(
            file_name=file_path.split("/")[-1],
            n_points=len(points),
            ground_count=ground_count,
            tree_count=tree_count,
            params=params,
            trees_params=[params],
            pointllm_analysis=pointllm_analysis,
            llm_answer=answer,
            scene_stats={
                "n_points": len(points),
                "ground_count": ground_count,
                "tree_count": tree_count,
                "total_trees": 1,
                "avg_height": params["height"],
                "avg_dbh": params["dbh"],
                "total_carbon": params["carbon_stock"],
                "height_range": [float(z_min), float(z_max)],
            },
        )
        self.cache.set(file_key, cached)

        return {
            "success": True,
            "answer": answer,
            "params": params,
            "n_points": len(points),
            "ground_count": ground_count,
            "tree_count": tree_count,
            "semantic_interpretation": _sanitize_dict(semantic_interp.get("semantic_interpretation", {})),
            "method": "traditional_geometry + llm_semantic",
        }

    # ---- 端点：/multi-analyze → 多树分割 + 分析 ----

    def multi_analyze(self, file_path: str, eps: float = 0.5, min_samples: int = 50) -> Dict:
        """
        多树分割分析：上传点云 → DBSCAN分割 → 逐树提取参数 → 缓存

        Args:
            file_path: 点云文件路径
            eps: DBSCAN聚类半径（米）
            min_samples: 最小点数阈值

        Returns:
            Dict: 多棵树的分析结果
        """
        points = load_point_cloud(file_path)
        z = points[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        height = z_max - z_min

        # Step 1: DBSCAN单木分割
        trees = segment_trees(points, eps=float(eps), min_samples=int(min_samples))

        if not trees:
            return {"success": False, "error": "未检测到树木，请调整聚类参数", "n_trees": 0}

        # Step 2: 逐树提取参数
        trees_params = []
        for i, tree_pts in enumerate(trees):
            params = extract_precise_params(tree_pts)
            params["tree_id"] = f"tree_{i}"
            trees_params.append(params)

        # Step 3: PointLLM场景编码（批量）
        pointllm_analysis = {}
        if self.pointllm:
            try:
                pointllm_analysis = self.pointllm.analyze(points)
            except Exception as e:
                print(f"[TreeAnalysisService] PointLLM batch analysis failed: {e}")

        # Step 4: 场景统计
        tree_base_z = z_min + height * 0.15
        ground_count = int((z < tree_base_z).sum())
        tree_count = int((z >= tree_base_z).sum())

        scene_stats = {
            "n_points": int(len(points)),
            "ground_count": ground_count,
            "tree_count": tree_count,
            "n_trees": len(trees_params),
            "avg_height": float(round(np.mean([p["height"] for p in trees_params]), 2)) if trees_params else 0.0,
            "avg_dbh": float(round(np.mean([p["dbh"] for p in trees_params]), 1)) if trees_params else 0.0,
            "total_carbon": float(round(sum(p["carbon_stock"] for p in trees_params), 1)),
            "height_range": [float(z_min), float(z_max)],
        }

        # Step 5: 缓存结果
        file_key = make_file_key(file_path.split("/")[-1], len(points))
        cached = build_cached_analysis(
            file_name=file_path.split("/")[-1],
            n_points=len(points),
            ground_count=scene_stats["ground_count"],
            tree_count=scene_stats["tree_count"],
            params=trees_params[0] if trees_params else {},
            trees_params=trees_params,
            pointllm_analysis=pointllm_analysis,
            llm_answer="",
            scene_stats=scene_stats,
        )
        self.cache.set(file_key, cached)

        # Step 6: LLM生成整体分析
        if self.llm_client:
            stand_summary = self.decision_engine.answer_question(
                "这片林分整体情况如何？有哪些需要注意的问题？",
                trees_params,
                scene_stats,
                "analyze",
            )
        else:
            stand_summary = f"林分包含{len(trees_params)}棵树木，平均树高{scene_stats['avg_height']:.2f}m，平均胸径{scene_stats['avg_dbh']:.1f}cm。"

        return {
            "success": True,
            "n_trees": len(trees_params),
            "trees_params": [_sanitize_dict(p) for p in trees_params],
            "scene_stats": _sanitize_dict(scene_stats),
            "stand_summary": stand_summary,
            "params_recommendation": _sanitize_dict(quick_recommend(len(points), tree_count, [float(z_min), float(z_max)])),
            "method": "dbscan_segmentation + traditional_geometry",
        }

    # ---- 端点：/ask → 自然语言问答（基于缓存） ----

    def ask(self, question: str, file_key: Optional[str] = None) -> Dict:
        """
        自然语言问答：基于已缓存的分析结果回答问题

        Args:
            question: 用户问题
            file_key: 缓存key（可选，默认用最新的）

        Returns:
            Dict: LLM回答 + 语义解读
        """
        # 获取缓存
        cached = None
        if file_key:
            cached = self.cache.get(file_key)
        else:
            keys = self.cache.keys()
            if keys:
                cached = self.cache.get(keys[-1])

        if not cached:
            return {
                "success": False,
                "answer": "请先上传点云进行分析。",
                "error": "no_cached_data",
            }

        # 意图分类
        intent = classify_intent(question)
        intent_desc = get_intent_description(intent)

        # LLM回答
        try:
            answer = self.decision_engine.answer_question(
                question=question,
                trees_params=cached.trees_params if cached.trees_params else [cached.params],
                scene_stats=cached.scene_stats if cached.scene_stats else {},
                intent=intent.value,
            )
        except Exception as e:
            print(f"[TreeAnalysisService] answer_question failed: {e}")
            answer = f"分析完成，但无法生成详细回答。错误：{e}"

        return {
            "success": True,
            "answer": str(answer),
            "intent": str(intent.value),
            "intent_description": str(intent_desc),
            "scene_stats": _sanitize_dict(cached.scene_stats) if cached.scene_stats else {},
            "trees_count": int(len(cached.trees_params) if cached.trees_params else 1),
        }

    # ---- 流式问答端点 ----

    def stream_answer(self, question: str, file_key: Optional[str] = None):
        """
        流式回答：yield LLM token流
        用于 SSE /ask 端点
        """
        cached = None
        if file_key:
            cached = self.cache.get(file_key)
        else:
            keys = self.cache.keys()
            if keys:
                cached = self.cache.get(keys[-1])

        if not cached:
            yield "请先上传点云进行分析。"
            return

        intent = classify_intent(question)
        trees_params = cached.trees_params if cached.trees_params else [cached.params]
        scene_stats = cached.scene_stats if cached.scene_stats else {}

        for token in self.decision_engine.stream_answer(
            question=question,
            trees_params=trees_params,
            scene_stats=scene_stats,
            intent=intent.value,
        ):
            yield token

    # ---- 端点：/report → 生成完整报告 ----

    def generate_report(
        self,
        trees_data: Optional[List[Dict]] = None,
        report_type: str = "standard",
        file_key: Optional[str] = None,
    ) -> Dict:
        """
        生成完整调查报告

        Args:
            trees_data: 树木参数列表（可选，直接从缓存获取）
            report_type: standard | detailed | carbon
            file_key: 缓存key（可选）

        Returns:
            Dict: 报告文本 + 摘要 + 建议
        """
        # 尝试从缓存获取数据
        if not trees_data:
            cached = None
            if file_key:
                cached = self.cache.get(file_key)
            else:
                keys = self.cache.keys()
                if keys:
                    cached = self.cache.get(keys[-1])

            if cached:
                trees_data = cached.trees_params if cached.trees_params else [cached.params]
                scene_stats = cached.scene_stats
            else:
                return {"success": False, "error": "没有树木数据，请先上传点云进行分析"}

        if not trees_data:
            return {"success": False, "error": "没有树木数据"}

        # 计算场景统计
        heights = [t.get("height", 0) for t in trees_data]
        dbhs = [t.get("dbh", 0) for t in trees_data]

        scene_stats = {
            "n_trees": len(trees_data),
            "avg_height": round(np.mean(heights), 2) if heights else 0,
            "avg_dbh": round(np.mean(dbhs), 1) if dbhs else 0,
            "total_carbon": round(sum(t.get("carbon_stock", 0) for t in trees_data), 1),
            "height_range": [min(heights) if heights else 0, max(heights) if heights else 0],
            "n_points": sum(t.get("n_points", 0) for t in trees_data),
        }

        # 使用决策引擎生成报告
        return self.decision_engine.generate_report(
            trees_params=trees_data,
            scene_stats=scene_stats,
            report_type=report_type,
        )

    # ---- 端点：/recommend-params → 参数推荐 ----

    def recommend_params(self, file_path: str) -> Dict:
        """
        根据点云场景自动推荐最佳处理参数

        Args:
            file_path: 点云文件路径

        Returns:
            Dict: 推荐参数 + 场景描述 + 置信度
        """
        points = load_point_cloud(file_path)
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        height = z_max - z_min

        tree_base_z = z_min + height * 0.15
        ground_count = int((z < tree_base_z).sum())
        tree_count = int((z >= tree_base_z).sum())

        # 使用参数推荐引擎
        recommendation = self.advisor.recommend_from_point_cloud(
            n_points=len(points),
            tree_count=tree_count,
            ground_count=ground_count,
            height_range=(z_min, z_max),
        )

        # 推荐林分类型和处理策略
        scene_type = recommendation.get("scene_type", "normal_forest")
        avg_height = height
        avg_dbh = min(20 + tree_count / 500, 80)  # 临时估算
        strategy = self.advisor.recommend_forest_type(avg_height, avg_dbh, 1)

        return {
            "success": True,
            "scene_type": str(scene_type),
            "dbscan_params": _sanitize_dict(recommendation["recommended_params"]),
            "biomass_formula": str(recommendation["biomass_formula"]),
            "description": str(recommendation["description"]),
            "warnings": list(recommendation.get("warnings", [])),
            "growth_stage": str(strategy["stage"]),
            "strategy_focus": str(strategy["strategy_focus"]),
            "recommended_actions": list(strategy["recommended_actions"]),
        }

    # ---- LLM调用 ----

    def _llm_describe(
        self,
        params: Dict,
        semantic_interp: Dict,
        pointllm_analysis: Dict,
        ground_count: int,
        tree_count: int,
    ) -> str:
        """无问题时：生成综合分析描述"""
        if not self.llm_client:
            return self._template_describe(params, semantic_interp)

        semantic = semantic_interp.get("semantic_interpretation", {})
        metrics = semantic_interp.get("derived_metrics", {})

        prompt = f"""你是资深林业调查专家，20年野外工作经验。

## 点云分析结果（传统几何算法提取，精确可靠）

### 精确测量参数
- 树高：{params['height']:.2f} m
- 胸径DBH：{params['dbh']:.1f} cm
- 冠幅：{params['crown_width']:.2f} m
- 树冠体积：{params['crown_volume']:.2f} m³
- 树干体积：{params['stem_volume']:.3f} m³
- 碳储量：{params['carbon_stock']:.1f} kg

### 点云统计
- 树木点数：{tree_count}（占比{tree_count/(ground_count+tree_count)*100:.0f}%）
- 地面点数：{ground_count}（占比{ground_count/(ground_count+tree_count)*100:.0f}%）

### 语义解读（基于精确参数）
- 生长阶段：{semantic.get('growth_stage', '未知')}
- 健康状态：{semantic.get('health_status', '未知')} - {semantic.get('health_description', '')}
- 风险等级：{semantic.get('risk_level', '低')} - {', '.join(semantic.get('risk_reasons', ['无明显风险'])) if semantic.get('risk_reasons') else '无明显风险'}
- 高径比：{metrics.get('height_diameter_ratio', 0):.1f}（超过70:1视为高风险）

### 管理建议
{'; '.join(semantic_interp.get('management_suggestions', ['保持常规养护']))}

请基于以上精确参数生成一段专业的分析报告，用中文，格式清晰。"""

        try:
            resp = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "你是资深林业调查专家，20年野外工作经验，擅长将精确测量数据转化为专业评估结论。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[TreeAnalysisService] LLM describe failed: {e}")
            return self._template_describe(params, semantic_interp)

    def _llm_answer(
        self,
        question: str,
        intent: Intent,
        trees_params: List[Dict],
        scene_stats: Dict,
    ) -> str:
        """有问题时：意图分类 + 定向回答"""
        return self.decision_engine.answer_question(
            question=question,
            trees_params=trees_params,
            scene_stats=scene_stats,
            intent=intent.value,
        )

    def _template_describe(self, params: Dict, semantic_interp: Dict) -> str:
        """无LLM时的模板描述"""
        semantic = semantic_interp.get("semantic_interpretation", {})
        metrics = semantic_interp.get("derived_metrics", {})
        carbon_val = semantic_interp.get("carbon_value", {})

        lines = [
            f"## 树木分析报告",
            "",
            f"### 精确测量参数",
            f"- **树高**：{params['height']:.2f} m（可靠度高）",
            f"- **胸径DBH**：{params['dbh']:.1f} cm",
            f"- **冠幅**：{params['crown_width']:.2f} m",
            f"- **碳储量**：{params['carbon_stock']:.1f} kg",
            "",
            f"### 语义解读",
            f"- 生长阶段：{semantic.get('growth_stage', '未知')}",
            f"- 健康状态：{semantic.get('health_status', '未知')} — {semantic.get('health_description', '')}",
            f"- 风险等级：{semantic.get('risk_level', '低')}",
            f"- 高径比：{metrics.get('height_diameter_ratio', 0):.1f} {'（⚠️ 超过70:1，倒伏风险较高）' if metrics.get('slenderness_flag') else ''}",
            "",
            f"### 碳汇价值",
            f"- 碳储量：{carbon_val.get('carbon_stock_kg', 0):.1f} kg（{carbon_val.get('carbon_stock_ton', 0):.3f} 吨）",
            f"- 碳交易价值（参考）：约 ¥{carbon_val.get('carbon_value_yuan', 0):.2f}",
            "",
            f"### 管理建议",
            *[f"- {s}" for s in semantic_interp.get("management_suggestions", [])],
        ]
        return "\n".join(lines)

    # ---- 工具方法 ----

    def get_cache_status(self) -> Dict:
        """获取缓存状态"""
        return {
            "cached_files": self.cache.size(),
            "keys": self.cache.keys(),
        }

    def clear_cache(self) -> Dict:
        """清空缓存"""
        self.cache.clear()
        return {"success": True, "message": "缓存已清空"}