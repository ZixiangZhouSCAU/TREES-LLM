"""
意图分类器 - 识别用户自然语言意图并路由到对应handler
解决痛点1：自然语言操作代替10步GUI
"""

from enum import Enum
from typing import List, Dict, Optional
import re


class Intent(str, Enum):
    """用户意图枚举"""
    ANALYZE = "analyze"           # 分析整片林分
    ASK_PARAMS = "ask_params"     # 询问具体参数
    HEALTH_CHECK = "health_check" # 健康状况评估
    GENERATE_REPORT = "generate_report"  # 生成报告
    RISK_CHECK = "risk_check"     # 风险识别
    CARBON_ASSESS = "carbon_assess"  # 碳汇评估
    MANAGEMENT_SUGGEST = "management_suggest"  # 管理建议
    COMPARE_TREES = "compare_trees"  # 树木对比
    SCENE_UNDERSTAND = "scene_understand"  # 场景理解
    RELATIONSHIP_QUERY = "relationship_query"  # 邻树关系查询
    GENERAL = "general"           # 通用问答


# 三层编码路由映射（3DCITY-LLM 指令驱动路由）
INTENT_TO_LAYER = {
    # Scene 层：需要林分级全局分析
    Intent.SCENE_UNDERSTAND: "scene",
    Intent.MANAGEMENT_SUGGEST: "scene",
    Intent.CARBON_ASSESS: "scene",
    Intent.ANALYZE: "scene",
    Intent.GENERATE_REPORT: "scene",
    # Relationship 层：需要树间关系
    Intent.COMPARE_TREES: "relationship",
    Intent.RISK_CHECK: "relationship",
    Intent.RELATIONSHIP_QUERY: "relationship",
    # Object 层：单树快速查询
    Intent.ASK_PARAMS: "object",
    Intent.HEALTH_CHECK: "object",
    Intent.GENERAL: "object",
}


def route_to_encoding_layer(intent: Intent) -> str:
    """根据意图路由到对应的三层编码分支"""
    return INTENT_TO_LAYER.get(intent, "object")


# 意图关键词映射
INTENT_PATTERNS: Dict[Intent, List[str]] = {
    Intent.ANALYZE: [
        "分析", "帮我看看", "这片林", "这片树", "这片区域",
        "整体情况", "概览", "总体", "整体评估",
    ],
    Intent.ASK_PARAMS: [
        "树高", "胸径", "冠幅", "参数", "高度", "宽度",
        "体积", "多少米", "多大", "数值",
    ],
    Intent.HEALTH_CHECK: [
        "健康", "枯死", "病虫害", "生长状况", "长势",
        "是否有病", "健康状态", "生长状态",
    ],
    Intent.GENERATE_REPORT: [
        "生成报告", "出报告", "写报告", "调查报告",
        "导出报告", "生成文档", "生成文档",
    ],
    Intent.RISK_CHECK: [
        "风险", "倒伏", "枯倒", "危险", "安全隐患",
        "哪棵最危险", "风险最高", "优先砍", "应急处理",
    ],
    Intent.CARBON_ASSESS: [
        "碳汇", "碳储量", "碳排放", "碳中和", "二氧化碳",
        "碳核算", "碳交易", "碳信用",
    ],
    Intent.MANAGEMENT_SUGGEST: [
        "管理建议", "怎么处理", "怎么经营", "砍多少",
        "间伐", "抚育", "采伐", "养护", "建议",
    ],
    Intent.COMPARE_TREES: [
        "对比", "哪棵最好", "哪棵最差", "排名",
        "比较", "最高", "最大", "最小",
    ],
    Intent.SCENE_UNDERSTAND: [
        "场景", "属于什么林", "什么类型的", "场景识别",
        "林分类型", "属于哪类", "分布情况",
    ],
}


def classify_intent(question: str) -> Intent:
    """
    根据用户问题识别意图

    Args:
        question: 用户输入的自然语言问题

    Returns:
        Intent: 识别到的意图类型
    """
    question_lower = question.lower()

    # 精确匹配优先
    # 报告生成 - 最高优先级
    if any(kw in question for kw in INTENT_PATTERNS[Intent.GENERATE_REPORT]):
        return Intent.GENERATE_REPORT

    # 风险识别 - 高优先级
    if any(kw in question for kw in INTENT_PATTERNS[Intent.RISK_CHECK]):
        return Intent.RISK_CHECK

    # 管理建议
    if any(kw in question for kw in INTENT_PATTERNS[Intent.MANAGEMENT_SUGGEST]):
        return Intent.MANAGEMENT_SUGGEST

    # 碳汇评估
    if any(kw in question for kw in INTENT_PATTERNS[Intent.CARBON_ASSESS]):
        return Intent.CARBON_ASSESS

    # 健康检查
    if any(kw in question for kw in INTENT_PATTERNS[Intent.HEALTH_CHECK]):
        return Intent.HEALTH_CHECK

    # 树木对比
    if any(kw in question for kw in INTENT_PATTERNS[Intent.COMPARE_TREES]):
        return Intent.COMPARE_TREES

    # 场景理解
    if any(kw in question for kw in INTENT_PATTERNS[Intent.SCENE_UNDERSTAND]):
        return Intent.SCENE_UNDERSTAND

    # 询问具体参数
    if any(kw in question for kw in INTENT_PATTERNS[Intent.ASK_PARAMS]):
        return Intent.ASK_PARAMS

    # 整体分析
    if any(kw in question for kw in INTENT_PATTERNS[Intent.ANALYZE]):
        return Intent.ANALYZE

    # 默认通用
    return Intent.GENERAL


def get_intent_description(intent: Intent) -> str:
    """获取意图的中文描述"""
    descriptions = {
        Intent.ANALYZE: "整体分析",
        Intent.ASK_PARAMS: "参数查询",
        Intent.HEALTH_CHECK: "健康评估",
        Intent.GENERATE_REPORT: "报告生成",
        Intent.RISK_CHECK: "风险识别",
        Intent.CARBON_ASSESS: "碳汇评估",
        Intent.MANAGEMENT_SUGGEST: "管理建议",
        Intent.COMPARE_TREES: "树木对比",
        Intent.SCENE_UNDERSTAND: "场景理解",
        Intent.GENERAL: "通用问答",
    }
    return descriptions.get(intent, "未知")


# 快捷问题预设（前端显示用）
QUICK_QUESTIONS: Dict[str, str] = {
    "帮我分析这片林子": "analyze",
    "这片林子健康吗": "health_check",
    "生成调查报告": "generate_report",
    "哪棵树风险最高": "risk_check",
    "碳汇量有多少": "carbon_assess",
    "应该怎么处理": "management_suggest",
    "对比一下所有树": "compare_trees",
    "这是什么类型的林分": "scene_understand",
}