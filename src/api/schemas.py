"""
API数据模型定义
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class TreeParams(BaseModel):
    """单棵树参数"""
    tree_id: str
    height: float = Field(description="树高(米)")
    dbh: float = Field(description="胸径(厘米)")
    crown_width: float = Field(description="冠幅(米)")
    crown_volume: float = Field(description="冠幅体积(立方米)")
    stem_volume: float = Field(description="树干体积(立方米)")
    carbon_stock: float = Field(description="碳储量(千克)")


class TreeParamsResponse(BaseModel):
    """参数提取响应"""
    success: bool = True
    tree_id: str
    params: TreeParams
    confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="各参数的置信度"
    )


class QuestionRequest(BaseModel):
    """问答请求"""
    point_cloud_path: str
    question: str = Field(
        description="用户问题",
        examples=[
            "这棵树长得健不健康？",
            "它旁边那棵和它有什么关系？",
            "这棵树的生长有什么异常吗？"
        ]
    )


class QuestionResponse(BaseModel):
    """问答响应"""
    answer: str
    related_params: Optional[Dict[str, float]] = None


class ReportRequest(BaseModel):
    """报告生成请求"""
    trees_data: List[TreeParams]
    report_type: str = Field(
        default="standard",
        description="报告类型: standard | detailed | carbon"
    )


class ReportResponse(BaseModel):
    """报告生成响应"""
    success: bool = True
    report_text: str
    summary: Dict[str, Any]
    trees_count: int


class TreeNode(BaseModel):
    """树木节点"""
    tree_id: str
    position: Dict[str, float]  # x, y, z
    species: Optional[str] = None
    params: Dict[str, float]


class TreeEdge(BaseModel):
    """树木关系边"""
    source: str
    target: str
    relation: str = Field(
        description="关系类型: adjacent | mutual_shade | same_species | competition"
    )
    distance: float = Field(description="距离(米)")


class SceneGraphResponse(BaseModel):
    """空间关系图响应"""
    nodes: List[TreeNode]
    edges: List[TreeEdge]
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="图统计摘要"
    )
