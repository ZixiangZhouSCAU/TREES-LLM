"""
决策推理引擎 - 从点到决策的完整链条
解决痛点6：打通点云→参数→语义→决策→报告的完整链路
"""

from typing import Dict, List, Optional, Any
import numpy as np

from src.api.interpreters import TreeParameterInterpreter
from src.api.parameter_advisor import ParameterAdvisor
from src.data.forest_knowledge import (
    build_knowledge_context,
    get_scene_description,
    MANAGEMENT_RULES,
)


class DecisionEngine:
    """
    林业决策推理引擎（v0.4 — 三层编码 + RAG）

    功能：
    - 接收精确参数 + 语义理解 → 输出可执行决策建议
    - 生成完整的调查报告
    - 提供多场景决策支持
    - RAG 知识库检索增强
    - 三层编码器指令路由（Object/Relationship/Scene）
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.interpreter = TreeParameterInterpreter(llm_client)
        self.advisor = ParameterAdvisor(llm_client)

        # RAG 系统（惰性初始化）
        self._rag = None
        self._rag_available = False

    def init_rag(self):
        """初始化 RAG 系统（延迟加载，避免阻塞服务器启动）"""
        if self._rag is not None:
            return
        try:
            from src.api.rag import get_rag_system
            self._rag = get_rag_system()
            self._rag_available = True
            print("[DecisionEngine] RAG system initialized")
        except Exception as e:
            print(f"[DecisionEngine] RAG not available: {e}")
            self._rag_available = False

    def answer_with_rag(
        self,
        question: str,
        trees_params: List[Dict],
        scene_stats: Dict,
        intent: str,
    ) -> str:
        """
        使用 RAG 检索增强回答问题（优先级高于普通 answer_question）

        Args:
            question: 用户问题
            trees_params: 树木参数列表
            scene_stats: 场景统计
            intent: 意图类型

        Returns:
            str: 自然语言回答
        """
        if self._rag is None:
            self.init_rag()

        if not self._rag_available or self._rag is None:
            return self.answer_question(question, trees_params, scene_stats, intent)

        try:
            prompts = self._rag.build_rag_prompt(
                question=question,
                trees_params=trees_params,
                scene_stats=scene_stats,
                intent=intent,
            )
        except Exception as e:
            print(f"[DecisionEngine] RAG prompt build failed: {e}, fallback to normal")
            return self.answer_question(question, trees_params, scene_stats, intent)

        if not self.llm_client:
            return self._template_answer(question, trees_params, scene_stats, intent)

        try:
            resp = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[DecisionEngine] RAG LLM answer failed: {e}")
            return self.answer_question(question, trees_params, scene_stats, intent)

    def generate_report(
        self,
        trees_params: List[Dict],
        scene_stats: Dict,
        report_type: str = "standard",
    ) -> Dict[str, Any]:
        """
        生成完整调查报告

        Args:
            trees_params: 树木参数列表
            scene_stats: 场景统计量
            report_type: standard | detailed | carbon

        Returns:
            Dict: 报告内容 {success, report_text, summary, recommendations}
        """
        if not trees_params:
            return {"success": False, "error": "没有树木数据"}

        # 1. 林分级别解读
        stand_interpretation = self.interpreter.interpret_multiple_trees(
            trees_params, scene_stats
        )

        # 2. 单棵树解读
        tree_interpretations = []
        for i, params in enumerate(trees_params):
            interp = self.interpreter.interpret_single_tree(
                tree_id=params.get("tree_id", f"tree_{i}"),
                params=params,
            )
            tree_interpretations.append(interp)

        # 3. 生成报告内容
        if self.llm_client:
            report_text = self._llm_generate_report(
                stand_interpretation,
                tree_interpretations,
                scene_stats,
                report_type,
            )
        else:
            report_text = self._template_generate_report(
                stand_interpretation,
                tree_interpretations,
                scene_stats,
                report_type,
            )

        # 4. 构建摘要
        summary = {
            "total_trees": len(trees_params),
            "forest_type": stand_interpretation.get("forest_type", "未知"),
            "growth_stage": stand_interpretation.get("growth_stage", "未知"),
            "avg_height": scene_stats.get("avg_height", 0),
            "avg_dbh": scene_stats.get("avg_dbh", 0),
            "total_carbon": scene_stats.get("total_carbon", 0),
            "stand_quality": stand_interpretation.get("stand_quality", ""),
            "n_risk_trees": len(stand_interpretation.get("risk_trees", [])),
        }

        return {
            "success": True,
            "report_text": report_text,
            "summary": summary,
            "recommendations": stand_interpretation.get("management_advice", ""),
            "trees_count": len(trees_params),
        }

    def answer_question(
        self,
        question: str,
        trees_params: List[Dict],
        scene_stats: Dict,
        intent: str,
    ) -> str:
        """
        回答用户问题（基于意图路由）

        Args:
            question: 用户问题
            trees_params: 树木参数列表
            scene_stats: 场景统计量
            intent: 意图类型

        Returns:
            str: 自然语言回答
        """
        if not self.llm_client:
            return self._template_answer(question, trees_params, scene_stats, intent)

        # LLM驱动的回答
        knowledge_context = build_knowledge_context()
        scene_desc = get_scene_description(scene_stats)

        # 构建参数摘要
        params_summary = self._build_params_summary(trees_params)

        # 意图对应的system prompt
        intent_prompts = {
            "analyze": "你是一位资深林业调查专家。用户上传了点云数据并请求分析。请基于树木参数给出专业的整体评估。",
            "health_check": "你是一位林业健康评估专家。用户想了解林分的健康状况。请基于树木参数评估健康状态并给出建议。",
            "risk_check": "你是一位林业风险评估专家。用户想知道哪些树木存在倒伏风险。请识别高风险树木并给出处理建议。",
            "carbon_assess": "你是碳汇评估专家。用户想了解林分的碳汇价值。请基于树木参数计算碳储量并评估经济价值。",
            "management_suggest": "你是森林经营规划专家。用户想了解如何管理这片林分。请基于林分特征给出具体的经营建议。",
            "general": "你是林业智能助手。用户提出了一个关于林业的问题。请基于提供的数据给出专业、准确的回答。",
        }

        system_prompt = intent_prompts.get(intent, intent_prompts["general"])
        system_prompt += f"\n\n## 林业知识库\n{knowledge_context}"

        user_prompt = f"""## 场景信息\n{scene_desc}\n\n## 树木参数\n{params_summary}\n\n## 用户问题\n{question}\n\n请给出专业、准确、有实际指导意义的回答。用中文回答。"""

        try:
            resp = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[DecisionEngine] LLM answer failed: {e}")
            return self._template_answer(question, trees_params, scene_stats, intent)

    def stream_answer(
        self,
        question: str,
        trees_params: List[Dict],
        scene_stats: Dict,
        intent: str,
    ):
        """
        流式回答用户问题，yield每个token片段

        Yields:
            str: LLM输出的token片段
        """
        if not self.llm_client:
            yield self._template_answer(question, trees_params, scene_stats, intent)
            return

        knowledge_context = build_knowledge_context()
        scene_desc = get_scene_description(scene_stats)
        params_summary = self._build_params_summary(trees_params)

        intent_prompts = {
            "analyze": "你是一位资深林业调查专家。用户上传了点云数据并请求分析。请基于树木参数给出专业的整体评估。",
            "health_check": "你是一位林业健康评估专家。用户想了解林分的健康状况。请基于树木参数评估健康状态并给出建议。",
            "risk_check": "你是一位林业风险评估专家。用户想知道哪些树木存在倒伏风险。请识别高风险树木并给出处理建议。",
            "carbon_assess": "你是碳汇评估专家。用户想了解林分的碳汇价值。请基于树木参数计算碳储量并评估经济价值。",
            "management_suggest": "你是森林经营规划专家。用户想了解如何管理这片林分。请基于林分特征给出具体的经营建议。",
            "general": "你是林业智能助手。用户提出了一个关于林业的问题。请基于提供的数据给出专业、准确的回答。",
        }

        system_prompt = intent_prompts.get(intent, intent_prompts["general"])
        system_prompt += f"\n\n## 林业知识库\n{knowledge_context}"

        user_prompt = f"""## 场景信息\n{scene_desc}\n\n## 树木参数\n{params_summary}\n\n## 用户问题\n{question}\n\n请给出专业、准确、有实际指导意义的回答。用中文回答。"""

        try:
            resp = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                stream=True,
            )
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            print(f"[DecisionEngine] LLM stream failed: {e}")
            yield self._template_answer(question, trees_params, scene_stats, intent)

    # ---- 内部方法 ----

    def _build_params_summary(self, trees_params: List[Dict]) -> str:
        """构建参数摘要文本"""
        if not trees_params:
            return "无树木数据"

        lines = []
        for i, p in enumerate(trees_params[:20]):  # 最多20棵
            lines.append(
                f"- {p.get('tree_id', f'tree_{i}')}: "
                f"树高{p.get('height', 0):.2f}m, "
                f"胸径{p.get('dbh', 0):.1f}cm, "
                f"冠幅{p.get('crown_width', 0):.2f}m, "
                f"碳储量{p.get('carbon_stock', 0):.1f}kg"
            )

        if len(trees_params) > 20:
            lines.append(f"...（共{len(trees_params)}棵，仅显示前20棵）")

        return "\n".join(lines)

    def _llm_generate_report(
        self,
        stand_interpretation: Dict,
        tree_interpretations: List[Dict],
        scene_stats: Dict,
        report_type: str,
    ) -> str:
        """使用LLM生成完整报告"""
        knowledge_context = build_knowledge_context()
        scene_desc = get_scene_description(scene_stats)

        # 树木详情摘要
        tree_details = []
        for interp in tree_interpretations[:50]:
            params = interp["precise_params"]
            semantic = interp["semantic_interpretation"]
            tree_details.append(
                f"### {interp['tree_id']}\n"
                f"- 树高: {params['height']:.2f}m | 胸径: {params['dbh']:.1f}cm | "
                f"冠幅: {params['crown_width']:.2f}m | 碳储量: {params['carbon_stock']:.1f}kg\n"
                f"- 生长阶段: {semantic['growth_stage']} | 健康状态: {semantic['health_status']} | "
                f"风险等级: {semantic['risk_level']}\n"
                f"- 高径比: {interp['derived_metrics']['height_diameter_ratio']:.1f} | "
                f"管理建议: {'; '.join(interp['management_suggestions'][:2])}"
            )

        # 风险树木列表
        risk_trees = stand_interpretation.get("risk_trees", [])
        risk_section = ""
        if risk_trees:
            risk_section = "\n## 风险树木清单\n"
            for rt in risk_trees:
                risk_section += f"- **{rt['tree_id']}**: 高径比 {rt['height_diameter_ratio']:.0f}:1，需优先处理\n"

        prompt = f"""你是一位资深林业调查专家。请基于以下数据生成一份完整的树木调查报告。

## 林业知识库
{knowledge_context}

## 场景统计
{scene_desc}

## 林分评估结论
- 林分类型: {stand_interpretation.get('forest_type', '未知')}
- 生长阶段: {stand_interpretation.get('growth_stage', '未知')}
- 林分质量: {stand_interpretation.get('stand_quality', '未知')}
- 适用管理规则: {stand_interpretation.get('management_advice', '')}

## 树木详情
"""
        prompt += "\n\n".join(tree_details)

        if report_type == "carbon":
            prompt += "\n## 报告要求（碳汇专项）\n请在报告中增加：1) 各树碳储量排名 2) 碳汇总价值评估 3) 碳汇管理建议"
        elif report_type == "detailed":
            prompt += "\n## 报告要求（详细版）\n请在报告中增加：1) 每棵树的详细分析 2) 健康状况分布图描述 3) 管理优先级排序"

        prompt += f"\n\n{risk_section}\n\n请生成一份专业、完整、有实际指导意义的调查报告。格式清晰，使用markdown。"

        try:
            resp = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "你是资深林业调查专家，负责生成专业的树木调查报告。请以客观、专业的方式撰写报告。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[DecisionEngine] LLM report generation failed: {e}")
            return self._template_generate_report(stand_interpretation, tree_interpretations, scene_stats, report_type)

    def _template_generate_report(
        self,
        stand_interpretation: Dict,
        tree_interpretations: List[Dict],
        scene_stats: Dict,
        report_type: str,
    ) -> str:
        """使用模板生成报告（无LLM时fallback）"""
        lines = [
            "# 林业点云智能分析报告",
            "",
            "## 一、林分概览",
            f"- 树木总数：{len(tree_interpretations)} 棵",
            f"- 林分类型：{stand_interpretation.get('forest_type', '未知')}",
            f"- 生长阶段：{stand_interpretation.get('growth_stage', '未知')}",
            f"- 林分质量：{stand_interpretation.get('stand_quality', '')}",
            "",
            "## 二、统计摘要",
        ]

        scene = stand_interpretation.get("scene_stats", {})
        lines.append(f"- 平均树高：{scene.get('avg_height', 0):.2f} m")
        lines.append(f"- 平均胸径：{scene.get('avg_dbh', 0):.1f} cm")
        lines.append(f"- 总碳储量：{scene.get('total_carbon', 0):.1f} kg")
        lines.append(f"- 林分密度（估）：{scene.get('density_estimate', 0):.0f} 棵/公顷")
        lines.append(f"- 树高变异系数：{scene.get('height_cv', 0):.1f}%")

        # 风险树木
        risk_trees = stand_interpretation.get("risk_trees", [])
        if risk_trees:
            lines.append("")
            lines.append("## 三、风险树木")
            for rt in risk_trees:
                lines.append(f"- **{rt['tree_id']}**: 高径比 {rt['height_diameter_ratio']:.0f}:1，倒伏风险高")
        else:
            lines.append("")
            lines.append("## 三、风险评估：未发现高风险树木")

        # 管理建议
        lines.append("")
        lines.append("## 四、管理建议")
        advice = stand_interpretation.get("management_advice", "建议保持常规养护")
        lines.append(advice)

        if report_type in ["detailed", "carbon"]:
            lines.append("")
            lines.append("## 五、树木详情表")
            lines.append("| 编号 | 树高(m) | 胸径(cm) | 冠幅(m) | 碳储量(kg) | 健康状态 | 风险等级 |")
            lines.append("|------|---------|---------|---------|-----------|---------|---------|")
            for interp in tree_interpretations:
                params = interp["precise_params"]
                sem = interp["semantic_interpretation"]
                lines.append(
                    f"| {interp['tree_id']} | "
                    f"{params['height']:.2f} | "
                    f"{params['dbh']:.1f} | "
                    f"{params['crown_width']:.2f} | "
                    f"{params['carbon_stock']:.1f} | "
                    f"{sem['health_status']} | "
                    f"{sem['risk_level']} |"
                )

        if report_type == "carbon":
            lines.append("")
            lines.append("## 六、碳汇价值评估")
            total_carbon = sum(t["precise_params"]["carbon_stock"] for t in tree_interpretations)
            lines.append(f"- 碳储量总计：{total_carbon:.1f} kg ({total_carbon/1000:.3f} 吨）")
            lines.append(f"- 碳交易价值（参考）：约 ¥{total_carbon * 0.08:.2f}（按 ¥80/吨估算）")
            lines.append("- 说明：碳汇价值随市场波动，本估算仅供参考")

        lines.append("")
        lines.append("---")
        lines.append("*本报告由 TREES-LLM 智能分析系统自动生成*")

        return "\n".join(lines)

    def _template_answer(
        self,
        question: str,
        trees_params: List[Dict],
        scene_stats: Dict,
        intent: str,
    ) -> str:
        """使用模板回答问题（无LLM时fallback）"""
        if not trees_params:
            return "暂无树木数据，请先上传点云进行分析。"

        if intent == "health_check":
            health_counts = {}
            for t in trees_params:
                # 简单判断
                h = t.get("height", 0)
                d = t.get("dbh", 1)
                if d > 0:
                    ratio = h / (d / 100)
                    if ratio > 70:
                        status = "偏弱"
                    else:
                        status = "健康"
                    health_counts[status] = health_counts.get(status, 0) + 1
            return f"健康评估结果：{health_counts.get('健康', 0)}棵健康，{health_counts.get('偏弱', 0)}棵偏弱。整体林分健康状况{'良好' if health_counts.get('健康', 0) > len(trees_params) * 0.7 else '一般，建议关注'}。"

        elif intent == "risk_check":
            risks = []
            for t in trees_params:
                h = t.get("height", 0)
                d = t.get("dbh", 1)
                if d > 0 and h / (d / 100) > 70:
                    risks.append(f"{t.get('tree_id', 'tree')}: 高径比{h/(d/100):.0f}:1")
            if risks:
                return f"发现 {len(risks)} 棵高风险树木：{'; '.join(risks)}。建议优先处理倒伏风险。"
            return "未发现明显倒伏风险的树木，林分稳定性良好。"

        elif intent == "carbon_assess":
            total = sum(t.get("carbon_stock", 0) for t in trees_params)
            return f"碳储量总计：{total:.1f} kg（{total/1000:.3f} 吨）。按当前碳交易价格（约¥80/吨）估算，碳汇价值约 ¥{total*0.08:.2f}。"

        elif intent == "analyze":
            avg_h = sum(t.get("height", 0) for t in trees_params) / len(trees_params)
            avg_d = sum(t.get("dbh", 0) for t in trees_params) / len(trees_params)
            return f"林分包含 {len(trees_params)} 棵树木，平均树高 {avg_h:.2f}m，平均胸径 {avg_d:.1f}cm。整体处于{'中龄林' if avg_h < 15 else '成熟林'}阶段。"

        return f"林分共 {len(trees_params)} 棵树木，平均树高 {sum(t.get('height',0) for t in trees_params)/len(trees_params):.2f}m。"