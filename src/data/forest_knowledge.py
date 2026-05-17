"""
林业知识库 - 管理规程和生物量公式
LLM读取这些知识来做语义理解和生成管理建议
解决痛点2、3、6：参数→结论、报告生成、决策推理
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ForestType:
    """林分类型定义"""
    name: str                          # 如"人工用材林"
    tree_height_range: tuple           # 树高范围 (min, max) 单位m
    dbh_range: tuple                   # 胸径范围 (min, max) 单位cm
    density_range: tuple               # 密度范围 (min, max) 棵/公顷
    typical_species: List[str]         # 典型树种
    growth_stage: str                  # 所处阶段
    biomass_formula: str               # 推荐的生物量公式名称
    management_priority: str           # 经营优先级


@dataclass
class ManagementRule:
    """林业管理规则"""
    id: str
    name: str
    applicable_scene: str              # 适用场景
    condition: str                     # 触发条件
    action: str                        # 建议动作
    priority: int                      # 优先级 1=高 2=中 3=低
    description: str                   # 规则描述


@dataclass
class BiomassFormula:
    """生物量公式"""
    name: str
    species: str                       # 适用树种 "general"=通用
    formula: str                       # 公式文本
    variables: List[str]               # 变量列表
    unit: str                          # 输出单位
    source: str                        # 来源（国家标准等）


# 预定义林分类型
FOREST_TYPES: List[ForestType] = [
    ForestType(
        name="幼龄林",
        tree_height_range=(0, 5),
        dbh_range=(0, 10),
        density_range=(1000, 5000),
        typical_species=["杉木", "马尾松", "桉树"],
        growth_stage="幼龄林",
        biomass_formula="一般立木生物量方程",
        management_priority="中",
    ),
    ForestType(
        name="中龄林",
        tree_height_range=(5, 15),
        dbh_range=(10, 30),
        density_range=(400, 1500),
        typical_species=["杉木", "马尾松", "湿地松", "桉树"],
        growth_stage="中龄林",
        biomass_formula="一般立木生物量方程",
        management_priority="高",
    ),
    ForestType(
        name="成熟林",
        tree_height_range=(15, 30),
        dbh_range=(30, 60),
        density_range=(200, 600),
        typical_species=["杉木", "马尾松", "阔叶树"],
        growth_stage="成熟林",
        biomass_formula="成熟林生物量方程",
        management_priority="中",
    ),
    ForestType(
        name="过熟林",
        tree_height_range=(25, 50),
        dbh_range=(50, 100),
        density_range=(100, 400),
        typical_species=["天然林", "原始林"],
        growth_stage="过熟林",
        biomass_formula="老林生物量方程",
        management_priority="高",
    ),
    ForestType(
        name="城市行道树",
        tree_height_range=(3, 20),
        dbh_range=(10, 60),
        density_range=(50, 200),
        typical_species=["榕树", "樟树", "芒果", "木棉"],
        growth_stage="景观林",
        biomass_formula="城市树木生物量方程",
        management_priority="中",
    ),
    ForestType(
        name="热带雨林",
        tree_height_range=(20, 60),
        dbh_range=(20, 200),
        density_range=(400, 1200),
        typical_species=["热带乔木"],
        growth_stage="顶级群落",
        biomass_formula="热带雨林生物量方程",
        management_priority="低",
    ),
]


# 预定义管理规则
MANAGEMENT_RULES: List[ManagementRule] = [
    ManagementRule(
        id="rule_001",
        name="中龄林间伐规则",
        applicable_scene="中龄林",
        condition="林分密度 > 800 棵/公顷 且 平均胸径 18-30cm",
        action="建议近期（1-3年）进行抚育间伐，强度15-25%，优先伐除枯死木和病虫害木",
        priority=1,
        description="中龄林树冠开始郁闭，光合作用效率下降，需要间伐改善生长空间",
    ),
    ManagementRule(
        id="rule_002",
        name="高风险枯倒木规则",
        applicable_scene="所有林分",
        condition="高径比（树高/胸径）> 70:1",
        action="该树木倒伏风险极高，建议优先采伐处理，避免安全事故",
        priority=1,
        description="高径比大于70:1的树木机械稳定性差，风倒风险显著增加",
    ),
    ManagementRule(
        id="rule_003",
        name="胸径达标采伐规则",
        applicable_scene="用材林",
        condition="平均胸径 > 50cm",
        action="已达到主要用材树种采伐标准，可规划主伐利用",
        priority=2,
        description="胸径50cm以上为优质用材，建议在森林经营方案中规划采伐时间",
    ),
    ManagementRule(
        id="rule_004",
        name="碳汇优先保护规则",
        applicable_scene="成熟林/过熟林",
        condition="单棵碳储量 > 500kg",
        action="该树木碳汇价值高，建议长期保护，不纳入近期采伐计划",
        priority=2,
        description="成熟林和过熟林单株碳储量通常较大，保护性经营可维持碳汇功能",
    ),
    ManagementRule(
        id="rule_005",
        name="密度过低补充规则",
        applicable_scene="人工林",
        condition="密度 < 300 棵/公顷",
        action="林分密度偏低，建议补植或天然更新，提高林分覆盖率",
        priority=2,
        description="密度过低导致林地利用率不足，应通过补植提高林分质量",
    ),
    ManagementRule(
        id="rule_006",
        name="冠幅过窄预警规则",
        applicable_scene="所有林分",
        condition="冠幅/树高 < 0.2（树冠严重受压）",
        action="树木生长空间受限，建议间伐周围竞争木，改善营养空间",
        priority=1,
        description="冠幅过窄表明树冠受到严重挤压，如不及时处理将影响树木正常生长",
    ),
    ManagementRule(
        id="rule_007",
        name="城市行道树修剪规则",
        applicable_scene="城市行道树",
        condition="树高 > 15m 且 位于电线下方",
        action="建议定期修剪，防止枝条接触电线引发安全事故",
        priority=1,
        description="城市行道树需平衡景观功能与安全，城市树木管理规程要求定期修剪",
    ),
    ManagementRule(
        id="rule_008",
        name="幼龄林抚育规则",
        applicable_scene="幼龄林",
        condition="树高 < 5m 且 密度 > 2000 棵/公顷",
        action="林分过密，建议进行定株抚育，每亩保留60-80株",
        priority=1,
        description="幼龄林需要通过定株抚育调整密度，优化林木生长空间",
    ),
]


# 生物量公式库
BIOMASS_FORMULAS: List[BiomassFormula] = [
    BiomassFormula(
        name="通用立木生物量方程",
        species="general",
        formula="B = 0.5 × (H × DBH² / 10000) × WD",
        variables=["B=生物量(kg)", "H=树高(m)", "DBH=胸径(cm)", "WD=木材密度(kg/m³,一般取500)"],
        unit="kg",
        source="通用估算公式（林业行业标准）",
    ),
    BiomassFormula(
        name="碳储量估算公式",
        species="general",
        formula="C = B × 0.5",
        variables=["C=碳储量(kg)", "B=生物量(kg)", "木材含碳率约50%"],
        unit="kg",
        source="IPCC碳汇估算标准",
    ),
    BiomassFormula(
        name="杉木生物量方程",
        species="杉木",
        formula="B = 0.041 × DBH^{2.558} × H^{0.742}",
        variables=["B=生物量(kg)", "DBH=胸径(cm)", "H=树高(m)"],
        unit="kg",
        source="LY/T 2124-2013 杉木立木生物量模型",
    ),
    BiomassFormula(
        name="桉树生物量方程",
        species="桉树",
        formula="B = 0.118 × DBH^{2.386}",
        variables=["B=生物量(kg)", "DBH=胸径(cm)"],
        unit="kg",
        source="华南桉树人工林生物量估算",
    ),
]


def identify_forest_type(tree_height: float, dbh: float, density: Optional[float] = None) -> ForestType:
    """
    根据树高和胸径判断林分类型

    Args:
        tree_height: 平均树高(m)
        dbh: 平均胸径(cm)
        density: 密度（棵/公顷），可选

    Returns:
        ForestType: 识别到的林分类型
    """
    # 基于树高和胸径的判断逻辑
    if tree_height < 5:
        stage = "幼龄林"
    elif tree_height < 15:
        stage = "中龄林"
    elif tree_height < 25:
        stage = "成熟林"
    else:
        stage = "过熟林"

    # 匹配最适合的林分类型
    for ft in FOREST_TYPES:
        if stage in ft.name or stage == ft.growth_stage:
            return ft

    # 默认返回中龄林
    return FOREST_TYPES[1]


def get_applicable_rules(forest_type: ForestType, tree_params: Dict) -> List[ManagementRule]:
    """
    获取适用的管理规则

    Args:
        forest_type: 识别的林分类型
        tree_params: 树木参数

    Returns:
        List[ManagementRule]: 适用的规则列表（按优先级排序）
    """
    applicable = []

    # 高径比规则（所有场景适用）
    height = tree_params.get("height", 0)
    dbh = tree_params.get("dbh", 1)
    if dbh > 0:
        ratio = height / (dbh / 100)  # 树高/胸径(m/cm -> m/m)
        if ratio > 70:
            for rule in MANAGEMENT_RULES:
                if "高径比" in rule.name or rule.condition.startswith("高径比"):
                    applicable.append(rule)

    # 冠幅比规则
    crown_width = tree_params.get("crown_width", 0)
    if height > 0 and crown_width / height < 0.2:
        for rule in MANAGEMENT_RULES:
            if "冠幅" in rule.name:
                applicable.append(rule)

    # 林分类型特定规则
    for rule in MANAGEMENT_RULES:
        if forest_type.name in rule.applicable_scene or rule.applicable_scene in forest_type.name:
            if rule not in applicable:
                applicable.append(rule)

    # 按优先级排序
    applicable.sort(key=lambda r: r.priority)
    return applicable[:5]  # 最多返回5条


def get_scene_description(scene_stats: Dict) -> str:
    """
    根据场景统计量生成场景描述（用于发送给LLM）

    Args:
        scene_stats: 场景统计量字典

    Returns:
        str: 场景描述文本
    """
    total_trees = scene_stats.get("total_trees", 1)
    avg_height = scene_stats.get("avg_height", 0)
    avg_dbh = scene_stats.get("avg_dbh", 0)
    height_range = scene_stats.get("height_range", [0, 0])
    if not isinstance(height_range, (list, tuple)):
        height_range = [height_range, height_range]

    return f"""场景统计信息：
- 树木总数：{total_trees} 棵
- 平均树高：{avg_height:.2f} m（范围：{height_range[0]:.2f}~{height_range[1]:.2f} m）
- 平均胸径：{avg_dbh:.1f} cm
- 总点数：{scene_stats.get('n_points', 0):,}（地面点：{scene_stats.get('ground_count', 0):,}，树木点：{scene_stats.get('tree_count', 0):,}）"""


# 导出知识库用于prompt构建
def build_knowledge_context() -> str:
    """构建知识库上下文（供LLM system prompt使用）"""
    lines = ["## 林业知识库", ""]

    lines.append("### 林分类型判断标准")
    for ft in FOREST_TYPES:
        lines.append(f"- {ft.name}：树高{ft.tree_height_range[0]}-{ft.tree_height_range[1]}m，胸径{ft.dbh_range[0]}-{ft.dbh_range[1]}cm")
    lines.append("")

    lines.append("### 管理规则摘要")
    for rule in MANAGEMENT_RULES[:6]:  # 只取前6条最重要规则
        lines.append(f"- {rule.name}（优先级{rule.priority}）：{rule.description}")
    lines.append("")

    lines.append("### 生物量公式")
    for bf in BIOMASS_FORMULAS[:2]:  # 只取通用公式
        lines.append(f"- {bf.name}（{bf.source}）：{bf.formula}")

    return "\n".join(lines)