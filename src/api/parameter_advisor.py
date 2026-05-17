"""
参数推荐引擎 - 根据场景特征自动推荐最佳处理参数
解决痛点4+5：参数调优依赖经验 + 跨场景泛化差

使用规则+LLM双重判断：
1. 规则层：基于点云统计量的确定性推荐
2. LLM层：复杂场景下的经验判断
"""

from typing import Dict, List, Tuple, Optional, Any


class ParameterAdvisor:
    """
    参数推荐引擎

    功能：
    - 根据点云场景特征推荐最佳聚类参数
    - 推荐生物量公式
    - 推荐处理策略
    """

    # DBSCAN参数场景库
    DBSCAN_PROFILES = {
        "dense_forest": {
            "eps": 0.3,
            "min_samples": 80,
            "description": "密林场景（树木间距<0.5m），需要较小eps避免错误合并",
            "typical_species": ["热带雨林", "天然林", "人工桉树林"],
        },
        "normal_forest": {
            "eps": 0.5,
            "min_samples": 50,
            "description": "一般人工林场景（树木间距0.5-1.5m）",
            "typical_species": ["杉木林", "马尾松林", "一般用材林"],
        },
        "sparse_forest": {
            "eps": 0.8,
            "min_samples": 30,
            "description": "疏林或单棵散生（树木间距>1.5m）",
            "typical_species": ["防护林", "疏林地", "四旁树"],
        },
        "urban_trees": {
            "eps": 1.0,
            "min_samples": 20,
            "description": "城市行道树（间距大，地面硬质）",
            "typical_species": ["行道树", "景观树", "庭院树"],
        },
        "mature_forest": {
            "eps": 0.4,
            "min_samples": 60,
            "description": "成熟林/过熟林（树冠大，点密度高）",
            "typical_species": ["天然林", "原始林", "成熟用材林"],
        },
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def recommend_from_point_cloud(
        self,
        n_points: int,
        tree_count: int,
        ground_count: int,
        height_range: Tuple[float, float],
        scene_stats: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        根据点云统计量推荐参数

        Args:
            n_points: 总点数
            tree_count: 树木点数
            ground_count: 地面点数
            height_range: (min_z, max_z) 高度范围
            scene_stats: 额外场景统计量

        Returns:
            Dict: 推荐参数和建议
        """
        # 1. 计算场景指标
        height = height_range[1] - height_range[0]
        tree_ratio = tree_count / n_points if n_points > 0 else 0

        # 估计树木间距（基于点数密度）
        estimated_area = 100  # 假设覆盖100m²
        tree_density = tree_count / estimated_area  # 点/m²

        # 2. 判断场景类型
        scene_type = self._classify_scene(
            tree_count=tree_count,
            tree_ratio=tree_ratio,
            height=height,
        )

        # 3. 获取推荐参数
        profile = self.DBSCAN_PROFILES.get(scene_type, self.DBSCAN_PROFILES["normal_forest"])

        # 4. 精细调整（基于高度范围）
        if height > 20:  # 高大树木
            profile["eps"] = min(profile["eps"] + 0.2, 1.5)  # 稍微放大
            profile["description"] += "（已调整：高大树木树冠大，适当放大eps）"

        # 5. 风险检查
        warnings = []
        if tree_ratio < 0.3:
            warnings.append("树木点比例偏低，可能存在地面点误分类，建议检查点云质量")
        if height < 2:
            warnings.append("点云高度范围过小，可能不是完整的单棵树，建议使用包含完整树冠的点云")

        return {
            "scene_type": scene_type,
            "recommended_params": {
                "eps": profile["eps"],
                "min_samples": profile["min_samples"],
                "height_threshold": round(height_range[0] + height * 0.15, 3),
            },
            "biomass_formula": self._recommend_biomass_formula(scene_type),
            "description": profile["description"],
            "warnings": warnings,
            "confidence": "高" if scene_type in ["urban_trees", "sparse_forest", "normal_forest"] else "中",
        }

    def recommend_forest_type(
        self,
        avg_height: float,
        avg_dbh: float,
        n_trees: int,
    ) -> Dict[str, Any]:
        """
        推荐林分类型和处理策略

        Args:
            avg_height: 平均树高
            avg_dbh: 平均胸径
            n_trees: 树木数量

        Returns:
            Dict: 林分类型和处理建议
        """
        # 基于参数判断林分类型
        if avg_height < 5:
            stage = "幼龄林"
        elif avg_height < 15:
            stage = "中龄林"
        elif avg_height < 25:
            stage = "成熟林"
        else:
            stage = "过熟林"

        # 处理策略
        strategies = {
            "幼龄林": {
                "focus": "促进生长",
                "actions": ["除草松土", "补植", "定株抚育"],
                "monitoring": "每半年复查生长量",
            },
            "中龄林": {
                "focus": "优化结构",
                "actions": ["间伐", "修枝", "病虫害防治"],
                "monitoring": "每年监测林分密度",
            },
            "成熟林": {
                "focus": "规划利用",
                "actions": ["确定采伐时间", "评估碳汇价值", "制定采伐方案"],
                "monitoring": "每季度评估生长状态",
            },
            "过熟林": {
                "focus": "优先保护或采伐",
                "actions": ["评估碳汇价值", "决定保护或采伐", "更新造林规划"],
                "monitoring": "关注倒伏风险",
            },
        }

        strategy = strategies.get(stage, strategies["中龄林"])

        return {
            "stage": stage,
            "strategy_focus": strategy["focus"],
            "recommended_actions": strategy["actions"],
            "monitoring_plan": strategy["monitoring"],
            "priority": "高" if stage in ["幼龄林", "过熟林"] else "中",
        }

    def build_recommendation_prompt(
        self,
        scene_stats: Dict,
        tree_params: List[Dict],
    ) -> str:
        """
        构建LLM参数推荐prompt（用于复杂场景）

        当规则无法覆盖时，调用LLM做经验判断
        """
        prompt = f"""你是林业数据处理专家。用户上传了一片点云数据，需要你推荐最佳处理参数。

场景信息：
- 树木数量：{scene_stats.get('n_trees', len(tree_params))} 棵
- 平均树高：{scene_stats.get('avg_height', 0):.2f} m
- 平均胸径：{scene_stats.get('avg_dbh', 0):.1f} cm
- 高度范围：{scene_stats.get('height_range', [0, 0])} m
- 点总数：{scene_stats.get('n_points', 0):,}
- 树木点/地面点比：{scene_stats.get('tree_count', 0)}/{scene_stats.get('ground_count', 0)}

请推荐：
1. 最佳的DBSCAN聚类参数（eps和min_samples）
2. 适合的生物量估算公式
3. 是否需要特殊处理（如处理地面点干扰、修剪树冠点等）

直接给出推荐值，格式：
DBSCAN eps: X.X
DBSCAN min_samples: XX
生物量公式: XXX
特殊处理: XXX"""

        return prompt

    # ---- 内部方法 ----

    def _classify_scene(
        self,
        tree_count: int,
        tree_ratio: float,
        height: float,
    ) -> str:
        """基于统计量分类场景类型"""
        # 城市树木：高差小，树木点比例高
        if height < 10 and tree_ratio > 0.7:
            return "urban_trees"

        # 疏林
        if tree_ratio < 0.4 and height < 15:
            return "sparse_forest"

        # 密林
        if tree_ratio > 0.8 and height > 10:
            return "dense_forest"

        # 成熟林
        if height > 20:
            return "mature_forest"

        # 默认
        return "normal_forest"

    def _recommend_biomass_formula(self, scene_type: str) -> str:
        """推荐生物量公式"""
        formula_map = {
            "urban_trees": "城市树木生物量方程（简化版）",
            "sparse_forest": "稀疏立木生物量方程",
            "dense_forest": "密林生物量方程（考虑竞争）",
            "mature_forest": "成熟林生物量方程",
            "normal_forest": "通用立木生物量方程",
        }
        return formula_map.get(scene_type, "通用立木生物量方程")


# 快速参数推荐（无需实例化）
def quick_recommend(n_points: int, tree_count: int, height_range: Tuple[float, float]) -> Dict[str, Any]:
    """
    快速推荐参数（静态方法，无LLM）

    用于：用户上传点云后立即给出推荐，无需等待LLM响应
    """
    height = height_range[1] - height_range[0]

    # 基础场景判断
    if height < 10 and tree_count / n_points > 0.7:
        eps, min_s, scene = 1.0, 20, "城市行道树"
    elif tree_count / n_points > 0.8 and height > 10:
        eps, min_s, scene = 0.3, 80, "密林"
    elif height > 20:
        eps, min_s, scene = 0.4, 60, "成熟林"
    else:
        eps, min_s, scene = 0.5, 50, "一般林分"

    return {
        "scene": scene,
        "eps": eps,
        "min_samples": min_s,
        "confidence": "高",
        "note": f"基于点云统计量自动判断：高度{height:.2f}m，树木点比例{tree_count/n_points*100:.0f}%",
    }