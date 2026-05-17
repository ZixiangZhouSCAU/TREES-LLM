"""
API数据模型定义（Rule 8: Schema Validation）
所有端点入参/出参在API边界处校验
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# ============ 请求模型 ============

class AnalyzeRequest(BaseModel):
    """分析请求（无body，file通过UploadFile传递）"""
    pass


class AskRequest(BaseModel):
    """自然语言问答请求"""
    question: str = Field(min_length=1, max_length=1000, description="用户问题")
    file_key: Optional[str] = Field(default=None, description="缓存key（可选，默认用最新的）")


class MultiAnalyzeRequest(BaseModel):
    """多树分割分析请求"""
    pass  # file通过UploadFile传递, eps和min_samples通过Form


class ReportRequest(BaseModel):
    """报告生成请求"""
    trees_data: Optional[List["TreeParamsInput"]] = Field(default=None, description="树木参数列表（可选，从缓存获取）")
    report_type: str = Field(default="standard", description="报告类型: standard | detailed | carbon")
    file_key: Optional[str] = Field(default=None, description="缓存key（可选）")


class RecommendParamsRequest(BaseModel):
    """参数推荐请求"""
    pass  # file通过UploadFile传递


# ============ 输入模型 ============

class TreeParamsInput(BaseModel):
    """单棵树参数（输入）"""
    tree_id: str
    height: float = Field(ge=0, description="树高(米)")
    dbh: float = Field(ge=0, description="胸径(厘米)")
    crown_width: float = Field(ge=0, description="冠幅(米)")
    crown_volume: float = Field(ge=0, default=0, description="树冠体积(立方米)")
    stem_volume: float = Field(ge=0, default=0, description="树干体积(立方米)")
    carbon_stock: float = Field(ge=0, default=0, description="碳储量(千克)")


class TreeParams(TreeParamsInput):
    """单棵树参数（输出）"""
    pass


# ============ 响应模型 ============

class AnalyzeResponse(BaseModel):
    """分析响应"""
    success: bool = True
    answer: str = Field(description="自然语言回答")
    params: Dict[str, Any] = Field(description="结构化树木参数")
    n_points: int = Field(ge=0, description="输入点云点数")
    ground_count: int = Field(default=0, description="地面点数")
    tree_count: int = Field(default=0, description="树木点数")
    semantic_interpretation: Dict[str, Any] = Field(default_factory=dict, description="语义解读结果")
    method: str = Field(default="traditional_geometry + llm_semantic", description="分析方法")


class AskResponse(BaseModel):
    """自然语言问答响应"""
    success: bool = True
    answer: str = Field(description="LLM生成的自然语言回答")
    intent: str = Field(description="识别到的意图")
    intent_description: str = Field(description="意图描述")
    scene_stats: Dict[str, Any] = Field(default_factory=dict, description="场景统计")
    trees_count: int = Field(default=0, description="树木数量")


class MultiAnalyzeResponse(BaseModel):
    """多树分割分析响应"""
    success: bool = True
    n_trees: int = Field(ge=0, description="检测到的树木数量")
    trees_params: List[Dict[str, Any]] = Field(default_factory=list, description="各棵树的参数")
    scene_stats: Dict[str, Any] = Field(default_factory=dict, description="场景统计")
    stand_summary: str = Field(default="", description="林分整体分析")
    params_recommendation: Dict[str, Any] = Field(default_factory=dict, description="参数推荐")
    method: str = Field(default="dbscan_segmentation + traditional_geometry", description="分析方法")


class ReportResponse(BaseModel):
    """报告生成响应"""
    success: bool = True
    report_text: str
    summary: Dict[str, Any]
    recommendations: str = Field(default="", description="管理建议")
    trees_count: int


class RecommendParamsResponse(BaseModel):
    """参数推荐响应"""
    success: bool = True
    scene_type: str
    dbscan_params: Dict[str, Any]
    biomass_formula: str
    description: str
    warnings: List[str] = Field(default_factory=list)
    growth_stage: str
    strategy_focus: str
    recommended_actions: List[str] = Field(default_factory=list)


class CacheStatusResponse(BaseModel):
    """缓存状态响应"""
    cached_files: int
    keys: List[str]


# ============ 兼容旧端点的模型（保留但标记为deprecated） ============

class TreeParamsResponse(BaseModel):
    """参数提取响应（向后兼容）"""
    success: bool = True
    tree_id: str
    params: TreeParams
    confidence: Dict[str, float] = Field(default_factory=dict)


class QuestionRequest(BaseModel):
    """问答请求（向后兼容）"""
    point_cloud_path: str
    question: str = Field(max_length=500)


class QuestionResponse(BaseModel):
    """问答响应（向后兼容）"""
    answer: str
    related_params: Optional[Dict[str, float]] = None


# Update forward references
ReportRequest.model_rebuild()