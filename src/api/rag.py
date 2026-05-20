"""
RAG 检索系统 - 林业专家知识检索增强

功能：
- 基于 FAISS 的向量检索（CPU 可用）
- 将检索到的知识注入到 GLM prompt 中
- 支持三层编码器的指令路由

用法：
    rag = RAGSystem()
    rag.build_index()  # 首次运行需要构建索引
    context = rag.retrieve("如何计算一棵树的碳储量")
    # context 是检索到的知识上下文，可直接拼入 LLM prompt

    # 三层编码器路由：
    rag_context = rag.retrieve_for_intent(
        question="这片林子应该怎么处理",
        intent="management_suggest",  # → 检索管理规则相关知识
    )
"""

from pathlib import Path
from typing import Dict, List, Optional
import os

from src.api.knowledge_base import KnowledgeBase


# 按意图类型映射检索的知识文档
INTENT_KB_MAPPING: Dict[str, List[str]] = {
    # 碳汇相关 → 碳储量计算规范 + Q&A
    "carbon_assess": ["02_碳储量计算规范", "06_Q&A专家问答集"],
    # 管理建议 → 林分管理指南 + 林业法规
    "management_suggest": ["03_林分管理指南", "05_林业法规标准", "06_Q&A专家问答集"],
    # 健康检查 → 林分管理指南 + 树种数据库
    "health_check": ["03_林分管理指南", "01_树种数据库"],
    # 风险识别 → 林分管理指南
    "risk_check": ["03_林分管理指南"],
    # 场景理解 → 树种数据库 + 林分管理指南
    "scene_understand": ["01_树种数据库", "03_林分管理指南"],
    # 参数查询 → MLS参数提取规范
    "ask_params": ["04_MLS参数提取规范"],
    # 通用 → 全量检索
    "general": None,
    "analyze": None,
    "compare_trees": ["01_树种数据库", "03_林分管理指南"],
}


class RAGSystem:
    """
    RAG 林业知识检索系统

    支持：
    - 全量检索（通用问题）
    - 按意图路由检索（专业问题）
    - 自动将检索结果注入 LLM prompt
    """

    def __init__(
        self,
        knowledge_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Args:
            knowledge_dir: 知识库目录（resources/knowledge/）
            index_dir: FAISS 索引保存路径（默认 data/knowledge_index/）
            embedding_model: embedding 模型名称
        """
        project_root = Path(__file__).parent.parent.parent

        if knowledge_dir is None:
            knowledge_dir = str(project_root / "resources" / "knowledge")
        if index_dir is None:
            index_dir = str(project_root / "data" / "knowledge_index")

        self.knowledge_dir = knowledge_dir
        self.index_dir = index_dir

        self.kb = KnowledgeBase(
            knowledge_dir=knowledge_dir,
            embedding_model=embedding_model,
        )

        self._index_loaded = False

    def build_index(self, force_rebuild: bool = False) -> "RAGSystem":
        """
        构建 FAISS 索引

        优先从 index_dir 加载；若不存在则构建并保存

        Args:
            force_rebuild: 强制重新构建索引
        """
        index_path = Path(self.index_dir)

        if not force_rebuild and index_path.exists():
            try:
                self.kb.load_index(self.index_dir)
                self._index_loaded = True
                print(f"[RAGSystem] Loaded index from {self.index_dir}")
                return self
            except Exception as e:
                print(f"[RAGSystem] Failed to load index: {e}, rebuilding...")

        # 构建索引
        self.kb.load_all()
        if self.kb.chunks:
            self.kb.build_index()
            self.kb.save_index(self.index_dir)
            self._index_loaded = True
            print(f"[RAGSystem] Index built and saved to {self.index_dir}")
        else:
            print("[RAGSystem] WARNING: No knowledge chunks found. "
                  "Please add content to resources/knowledge/*.md")
            self._index_loaded = False

        return self

    def retrieve(
        self,
        question: str,
        intent: Optional[str] = None,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> Dict:
        """
        检索与问题最相关的知识

        Args:
            question: 用户问题
            intent: 意图类型（用于优先检索特定文档）
            top_k: 返回 top-k 个结果
            min_score: 最低相似度阈值

        Returns:
            Dict: {
                "answer": str,  # 拼接后的知识上下文
                "chunks": List[Dict],  # 原始检索块
                "sources": List[str],  # 来源文件列表
            }
        """
        if not self._index_loaded:
            self.build_index()

        if not self.kb.chunks:
            return {
                "answer": "",
                "chunks": [],
                "sources": [],
            }

        # 全量检索
        results = self.kb.search(question, top_k=top_k * 2)

        # 如果指定了意图，按意图过滤文档优先级
        if intent and intent in INTENT_KB_MAPPING and INTENT_KB_MAPPING[intent] is not None:
            preferred_docs = INTENT_KB_MAPPING[intent]
            # 对结果重排序：优先文档放前面
            def sort_key(r):
                for i, doc in enumerate(preferred_docs):
                    if doc in r.get("doc_id", ""):
                        return i
                return 999
            results.sort(key=sort_key)

        # 过滤低分结果
        results = [r for r in results if r["score"] >= min_score][:top_k]

        # 拼接上下文
        answer_parts = []
        sources = []
        for r in results:
            answer_parts.append(f"【{r['title']}】\n{r['content']}")
            if r["source_file"] not in sources:
                sources.append(r["source_file"])

        answer = "\n\n---\n\n".join(answer_parts)

        return {
            "answer": answer,
            "chunks": results,
            "sources": sources,
        }

    def retrieve_for_intent(
        self,
        question: str,
        intent: str,
        top_k: int = 3,
    ) -> str:
        """
        检索并直接返回上下文字符串（用于拼入 prompt）

        这是最常用的方法：
            context = rag.retrieve_for_intent(
                question="这片林子应如何管理",
                intent="management_suggest",
            )
        """
        result = self.retrieve(question, intent=intent, top_k=top_k)
        return result["answer"]

    def build_rag_prompt(
        self,
        question: str,
        trees_params: List[Dict],
        scene_stats: Dict,
        intent: str,
    ) -> Dict[str, str]:
        """
        构建带 RAG 上下文的完整 prompt

        Args:
            question: 用户问题
            trees_params: 树木参数列表
            scene_stats: 场景统计
            intent: 意图类型

        Returns:
            Dict: {"system": str, "user": str} 完整 prompt
        """
        # 1. RAG 检索知识上下文
        rag_context = self.retrieve_for_intent(question, intent=intent, top_k=3)

        # 2. 构建 system prompt
        intent_roles = {
            "analyze": "资深林业调查专家",
            "health_check": "林业健康评估专家",
            "risk_check": "林业风险评估专家",
            "carbon_assess": "碳汇评估专家",
            "management_suggest": "森林经营规划专家",
            "scene_understand": "林业分类专家",
            "compare_trees": "林业比较分析专家",
            "ask_params": "林业测量专家",
            "general": "林业智能助手",
        }
        role = intent_roles.get(intent, "林业智能助手")

        system_prompt = f"你是{role}。"

        # 3. 如果有 RAG 上下文，加入 system prompt
        if rag_context:
            system_prompt += f"\n\n## 参考知识\n以下是从林业知识库检索到的相关信息，可作为回答的参考依据：\n\n{rag_context}"

        # 4. 构建 user prompt（与 decision_engine 格式一致）
        scene_desc = self._build_scene_desc(scene_stats)
        params_summary = self._build_params_summary(trees_params)

        user_prompt = f"""## 场景信息\n{scene_desc}\n\n## 树木参数\n{params_summary}\n\n## 用户问题\n{question}\n\n请给出专业、准确、有实际指导意义的回答。用中文回答。如果参考知识中有相关内容，请结合使用。"""

        return {
            "system": system_prompt,
            "user": user_prompt,
        }

    def _build_scene_desc(self, scene_stats: Dict) -> str:
        """构建场景描述"""
        if not scene_stats:
            return "暂无点云数据，为纯对话模式。"

        total_trees = scene_stats.get("total_trees", 1)
        avg_height = scene_stats.get("avg_height", 0)
        avg_dbh = scene_stats.get("avg_dbh", 0)

        return (f"场景统计：共{total_trees}棵树木，"
                f"平均树高{avg_height:.2f}m，平均胸径{avg_dbh:.1f}cm。")

    def _build_params_summary(self, trees_params: List[Dict]) -> str:
        """构建参数摘要"""
        if not trees_params:
            return "无树木数据。"

        lines = []
        for i, p in enumerate(trees_params[:10]):
            lines.append(
                f"- {p.get('tree_id', f'tree_{i}')}: "
                f"树高{p.get('height', 0):.2f}m, "
                f"胸径{p.get('dbh', 0):.1f}cm, "
                f"冠幅{p.get('crown_width', 0):.2f}m, "
                f"碳储量{p.get('carbon_stock', 0):.1f}kg"
            )

        if len(trees_params) > 10:
            lines.append(f"...（共{len(trees_params)}棵，仅显示前10棵）")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """获取 RAG 系统统计"""
        return {
            **self.kb.get_stats(),
            "index_dir": self.index_dir,
        }


# 全局单例（惰性初始化）
_rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """获取全局 RAG 系统单例"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
        # 尝试加载已有索引，不存在则静默跳过
        try:
            _rag_system.build_index(force_rebuild=False)
        except Exception:
            pass
    return _rag_system


def test_rag():
    """测试 RAG 系统"""
    print("[RAG Test] Creating test knowledge files...")

    kb_dir = Path("resources/knowledge")
    kb_dir.mkdir(parents=True, exist_ok=True)

    # 创建测试文档
    (kb_dir / "01_树种数据库.md").write_text("""# 树种数据库

## 杉木
杉木是中国南方重要的用材树种。生长速度快，适应性强。
生物量公式: B = 0.041 × DBH^2.558 × H^0.742
适用于树高5-25m，胸径10-50cm的杉木林。

## 桉树
华南地区广泛种植的速生树种。
生物量公式: B = 0.118 × DBH^2.386
生长迅速，但需注意水肥管理。
""", encoding="utf-8")

    (kb_dir / "02_碳储量计算规范.md").write_text("""# 碳储量计算规范

## 碳储量估算方法
碳储量 = 生物量 × 含碳率（通用值 0.5）

## IPCC 碳汇标准
根据IPCC指南，森林碳汇估算应使用生物量扩展因子法。

## 生物量估算
通用公式: B = 0.5 × (H × DBH² / 10000) × WD
其中 WD 为木材密度，一般阔叶树取 0.45-0.55 g/cm³。
""", encoding="utf-8")

    # 测试
    rag = RAGSystem()
    rag.build_index(force_rebuild=True)

    # 测试检索
    result = rag.retrieve("如何计算一棵树的碳储量", intent="carbon_assess")
    print(f"\n检索结果（intent=carbon_assess）:")
    print(f"  来源: {result['sources']}")
    print(f"  chunks: {len(result['chunks'])}")
    for chunk in result['chunks']:
        print(f"  [{chunk['score']:.3f}] {chunk['title']}: {chunk['content'][:60]}...")

    # 测试 prompt 构建
    prompt = rag.build_rag_prompt(
        question="如何计算碳储量",
        trees_params=[{"tree_id": "A01", "height": 12.3, "dbh": 35.2, "crown_width": 4.5, "carbon_stock": 52.1}],
        scene_stats={"total_trees": 1, "avg_height": 12.3, "avg_dbh": 35.2},
        intent="carbon_assess",
    )
    print(f"\nSystem prompt 长度: {len(prompt['system'])} chars")
    print(f"User prompt 长度: {len(prompt['user'])} chars")

    print("\n[OK] RAG test passed")


if __name__ == "__main__":
    test_rag()