"""
树木参数语义解读器
将传统算法提取的精确参数转化为专业语义结论
解决痛点2：只输出数字 → 输出"这意味着什么"
"""

from typing import Dict, List, Optional, Any
import numpy as np

from src.data.forest_knowledge import (
    identify_forest_type,
    get_applicable_rules,
    get_scene_description,
    FOREST_TYPES,
    BIOMASS_FORMULAS,
)


class TreeParameterInterpreter:
    """
    树木参数语义解读器

    接收传统算法输出的精确参数，输出专业结论：
    - 林分类型判断
    - 健康状态评估
    - 风险识别
    - 生长阶段分析
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def interpret_single_tree(
        self,
        tree_id: str,
        params: Dict,
        pointllm_analysis: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        解读单棵树的参数

        Args:
            tree_id: 树木ID
            params: 树木参数 {height, dbh, crown_width, ...}
            pointllm_analysis: PointLLM分析结果（可选）

        Returns:
            Dict: 语义解读结果
        """
        height = params.get("height", 0)
        dbh = params.get("dbh", 0)
        crown_width = params.get("crown_width", 0)
        carbon = params.get("carbon_stock", 0)
        stem_volume = params.get("stem_volume", 0)

        # 1. 林分类型判断
        forest_type = identify_forest_type(height, dbh)

        # 2. 生长阶段判断
        growth_stage = self._judge_growth_stage(height, dbh)

        # 3. 健康状态评估（基于参数组合）
        health_status = self._assess_health(height, dbh, crown_width, pointllm_analysis)

        # 4. 风险识别
        risk_level, risk_reasons = self._assess_risk(height, dbh, crown_width)

        # 5. 高径比计算
        height_diameter_ratio = round(height / (dbh / 100), 1) if dbh > 0 else 0

        # 6. 冠幅比计算
        crown_ratio = round(crown_width / height, 3) if height > 0 else 0

        # 7. 管理建议
        management_suggestions = self._generate_suggestions(
            forest_type, growth_stage, health_status, risk_level
        )

        # 8. 碳汇价值评估
        carbon_value = self._assess_carbon_value(carbon, dbh, height)

        return {
            "tree_id": tree_id,
            # 精确参数（保留原始值）
            "precise_params": {
                "height": height,
                "dbh": dbh,
                "crown_width": crown_width,
                "carbon_stock": carbon,
                "stem_volume": stem_volume,
            },
            # 语义结论
            "semantic_interpretation": {
                "forest_type": forest_type.name,
                "growth_stage": growth_stage,
                "health_status": health_status,
                "health_description": self._get_health_description(health_status),
                "risk_level": risk_level,
                "risk_reasons": risk_reasons,
            },
            # 衍生指标
            "derived_metrics": {
                "height_diameter_ratio": height_diameter_ratio,
                "crown_ratio": crown_ratio,
                "slenderness_flag": height_diameter_ratio > 70,  # 高径比超过70视为细长
            },
            # 建议
            "management_suggestions": management_suggestions,
            "carbon_value": carbon_value,
        }

    def interpret_multiple_trees(
        self,
        trees_params: List[Dict],
        scene_stats: Dict,
    ) -> Dict[str, Any]:
        """
        解读多棵树（林分级别）

        Args:
            trees_params: 多棵树的参数列表
            scene_stats: 场景统计量

        Returns:
            Dict: 林分级别解读
        """
        if not trees_params:
            return {"error": "没有树木数据"}

        heights = [t.get("height", 0) for t in trees_params]
        dbhs = [t.get("dbh", 0) for t in trees_params]
        carbons = [t.get("carbon_stock", 0) for t in trees_params]

        # 林分统计
        avg_height = np.mean(heights)
        avg_dbh = np.mean(dbhs)
        total_carbon = sum(carbons)
        n_trees = len(trees_params)

        # 林分密度估算（假设场景面积约100m²）
        density = n_trees * 100  # 棵/公顷（临时估算，需要实际面积）

        # 林分类型判断
        forest_type = identify_forest_type(avg_height, avg_dbh, density)

        # 树高变异系数（反映林分整齐度）
        height_cv = round(np.std(heights) / avg_height * 100, 1) if avg_height > 0 else 0

        # 最高树、最低树
        max_height_tree = trees_params[np.argmax(heights)]
        min_height_tree = trees_params[np.argmin(heights)]

        # 风险树木识别
        risk_trees = []
        for i, t in enumerate(trees_params):
            h = t.get("height", 0)
            d = t.get("dbh", 1)
            if d > 0 and h / (d / 100) > 70:
                risk_trees.append({
                    "tree_id": t.get("tree_id", f"tree_{i}"),
                    "height_diameter_ratio": round(h / (d / 100), 1),
                })

        # 获取适用的管理规则
        rules = get_applicable_rules(forest_type, {
            "height": avg_height,
            "dbh": avg_dbh,
            "crown_width": np.mean([t.get("crown_width", 0) for t in trees_params]),
        })

        return {
            "n_trees": n_trees,
            "forest_type": forest_type.name,
            "growth_stage": self._judge_growth_stage(avg_height, avg_dbh),
            "scene_stats": {
                "avg_height": round(avg_height, 2),
                "avg_dbh": round(avg_dbh, 1),
                "total_carbon": round(total_carbon, 1),
                "density_estimate": density,
                "height_cv": height_cv,
                "height_range": [min(heights), max(heights)],
            },
            "stand_quality": self._judge_stand_quality(avg_height, avg_dbh, height_cv),
            "risk_trees": risk_trees,
            "applicable_rules": [
                {"id": r.id, "name": r.name, "action": r.action, "priority": r.priority}
                for r in rules
            ],
            "management_advice": self._generate_stand_advice(
                forest_type, avg_height, avg_dbh, density, len(risk_trees)
            ),
        }

    # ---- 内部方法 ----

    def _judge_growth_stage(self, height: float, dbh: float) -> str:
        """判断生长阶段"""
        if height < 5 and dbh < 10:
            return "幼龄林"
        elif height < 15 and dbh < 30:
            return "中龄林"
        elif height < 25 and dbh < 50:
            return "成熟林"
        else:
            return "过熟林"

    def _assess_health(
        self,
        height: float,
        dbh: float,
        crown_width: float,
        pointllm_analysis: Optional[Dict],
    ) -> str:
        """评估健康状态"""
        if height <= 0:
            return "未知"

        # 冠幅比判断
        crown_ratio = crown_width / height if height > 0 else 0
        if crown_ratio > 0.4:
            return "健康"
        elif crown_ratio > 0.2:
            return "一般"
        elif crown_ratio > 0.1:
            return "偏弱"
        else:
            return "衰弱"

    def _get_health_description(self, status: str) -> str:
        """获取健康状态描述"""
        descriptions = {
            "健康": "树冠饱满，生长旺盛，光合作用效率高，预计未来生长趋势良好",
            "一般": "树冠正常，生长稳定，处于正常生长状态",
            "偏弱": "树冠受压或偏小，可能存在营养空间不足或竞争压力，建议关注",
            "衰弱": "树冠严重受压，生长空间受限，需及时抚育干预或检查病虫害",
            "未知": "数据不足，无法判断健康状态",
        }
        return descriptions.get(status, "无法评估")

    def _assess_risk(
        self,
        height: float,
        dbh: float,
        crown_width: float,
    ) -> tuple:
        """
        评估倒伏风险

        Returns:
            (risk_level: str, risk_reasons: List[str])
        """
        risks = []
        level = "低"

        if dbh > 0:
            # 高径比
            h_d_ratio = height / (dbh / 100)
            if h_d_ratio > 100:
                risks.append(f"高径比{h_d_ratio:.0f}:1，严重头重脚轻，倒伏风险极高")
                level = "极高"
            elif h_d_ratio > 70:
                risks.append(f"高径比{h_d_ratio:.0f}:1，倒伏风险较高")
                level = "高" if level == "低" else level

        # 冠幅比
        if height > 0:
            crown_ratio = crown_width / height
            if crown_ratio < 0.15:
                risks.append(f"冠幅比{crown_ratio:.2f}，树冠严重受压，机械稳定性差")
                if level == "低":
                    level = "中"

        # 冠幅不对称（如果有PointLLM数据）
        if height < 3:
            risks.append("树体较小但存在生长空间受限风险")

        if not risks:
            return "低", ["树木形态正常，倒伏风险低"]

        return level, risks

    def _generate_suggestions(
        self,
        forest_type,
        growth_stage: str,
        health_status: str,
        risk_level: str,
    ) -> List[str]:
        """生成管理建议"""
        suggestions = []

        # 健康状态建议
        if health_status == "衰弱":
            suggestions.append("建议进行现场检查，确认是否存在病虫害或营养缺乏")
        elif health_status == "偏弱":
            suggestions.append("建议加强水肥管理，或进行适度修剪改善树冠")

        # 风险建议
        if risk_level in ["高", "极高"]:
            suggestions.append("该树木存在倒伏风险，建议优先处理或加固支撑")
        elif risk_level == "中":
            suggestions.append("建议定期监测，必要时进行修剪降低重心")

        # 生长阶段建议
        if growth_stage == "中龄林":
            suggestions.append("处于中龄林阶段，建议关注林分密度，适时进行抚育间伐")
        elif growth_stage == "成熟林":
            suggestions.append("已达成熟期，可根据经营目标规划采伐时间")

        if not suggestions:
            suggestions.append("树木生长正常，建议保持常规养护")

        return suggestions

    def _assess_carbon_value(
        self,
        carbon: float,
        dbh: float,
        height: float,
    ) -> Dict[str, Any]:
        """评估碳汇价值"""
        if carbon <= 0:
            return {"carbon_stock_kg": 0, "carbon_value_yuan": 0, "level": "无"}

        # 碳交易价格参考（2024年国内自愿减排市场约¥50-100/吨）
        price_per_kg = 0.08  # ¥0.08/kg ≈ ¥80/吨
        value = round(carbon * price_per_kg, 2)

        if carbon > 500:
            level = "高"
        elif carbon > 100:
            level = "中"
        else:
            level = "低"

        return {
            "carbon_stock_kg": round(carbon, 1),
            "carbon_stock_ton": round(carbon / 1000, 3),
            "carbon_value_yuan": value,
            "level": level,
            "note": "碳汇价值基于当前自愿减排市场价格估算，实际交易价格可能波动",
        }

    def _judge_stand_quality(
        self,
        avg_height: float,
        avg_dbh: float,
        height_cv: float,
    ) -> str:
        """判断林分质量等级"""
        # 基于树高变异系数判断整齐度
        if height_cv < 15:
           整齐度 = "整齐"
        elif height_cv < 30:
           整齐度 = "一般"
        else:
           整齐度 = "参差"

        # 综合评价
        if avg_height > 15 and avg_dbh > 25 and height_cv < 20:
            quality = f"优（林分{整齐度}，生长良好）"
        elif avg_height > 8 and avg_dbh > 15:
            quality = f"良（林分{整齐度}，生长正常）"
        elif avg_height > 5:
            quality = f"中（林分{整齐度}，需加强管理）"
        else:
            quality = "差（需全面抚育）"

        return quality

    def _generate_stand_advice(
        self,
        forest_type,
        avg_height: float,
        avg_dbh: float,
        density: float,
        n_risk_trees: int,
    ) -> str:
        """生成林分级别管理建议"""
        advices = []

        # 密度建议
        if density > 1500:
            advices.append(f"林分密度偏大（约{density:.0f}棵/公顷），建议进行间伐，强度15-25%")
        elif density < 300:
            advices.append(f"林分密度偏低（约{density:.0f}棵/公顷），建议补植或天然更新")

        # 风险树木建议
        if n_risk_trees > 0:
            advices.append(f"发现{n_risk_trees}棵高风险树木（高径比>70:1），建议优先处理")

        # 整体建议
        if avg_height < 5:
            advices.append("林分处于幼龄阶段，重点做好除草松土等抚育工作")
        elif avg_height < 15:
            advices.append("林分处于中龄林阶段，建议每3-5年进行一次抚育间伐")
        else:
            advices.append("林分接近或达到成熟期，可根据经营目标规划主伐")

        return "; ".join(advices) if advices else "建议保持常规养护管理"