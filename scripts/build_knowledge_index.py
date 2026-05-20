"""
构建 RAG 知识库 FAISS 索引

用法：
    # 构建索引
    python scripts/build_knowledge_index.py

    # 强制重建
    python scripts/build_knowledge_index.py --rebuild
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.rag import RAGSystem


def main():
    import argparse
    parser = argparse.ArgumentParser(description="构建 RAG 知识库 FAISS 索引")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新构建索引")
    parser.add_argument("--knowledge-dir", type=str, default=None,
                        help="知识库目录路径")
    args = parser.parse_args()

    rag = RAGSystem()

    if args.knowledge_dir:
        rag = RAGSystem(knowledge_dir=args.knowledge_dir)

    print("[build_knowledge_index] Building FAISS index...")
    rag.build_index(force_rebuild=args.rebuild)

    stats = rag.get_stats()
    print(f"\n索引统计：")
    print(f"  知识库目录: {stats['knowledge_dir']}")
    print(f"  文档块数量: {stats['total_chunks']}")
    print(f"  索引已构建: {stats['index_built']}")
    print(f"  Embedding 模型: {stats['embedding_model']}")
    print(f"  索引保存路径: {rag.index_dir}")


if __name__ == "__main__":
    main()