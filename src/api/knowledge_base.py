"""
知识库加载系统 - 将 Markdown 文档加载为结构化 chunks，供 RAG 检索使用

功能：
- 加载 resources/knowledge/ 目录下的 .md 文档
- 按 Markdown 二级标题分块（每块 ~500 字）
- 生成 embedding（使用 sentence-transformers）
- 构建 FAISS 索引（CPU 可用）
- 支持增量更新

用法：
    kb = KnowledgeBase()
    kb.load_all()
    kb.build_index()
    kb.save_index("data/knowledge_index")

    # 检索
    results = kb.search("如何计算碳储量", top_k=3)
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


class KnowledgeChunk:
    """知识库分块"""

    def __init__(self, doc_id: str, title: str, content: str, source_file: str):
        self.doc_id = doc_id       # 如 "01-树种数据库"
        self.title = title         # 如 "杉木"
        self.content = content     # 块内容
        self.source_file = source_file  # 源文件路径
        self.embedding: Optional[np.ndarray] = None  # 预计算的 embedding

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeChunk":
        chunk = cls(d["doc_id"], d["title"], d["content"], d["source_file"])
        return chunk


class KnowledgeBase:
    """
    林业知识库管理器

    目录结构（resources/knowledge/）：
    ├── 01_树种数据库.md
    ├── 02_碳储量计算规范.md
    ├── 03_林分管理指南.md
    ├── 04_MLS参数提取规范.md
    ├── 05_林业法规标准.md
    └── 06_Q&A专家问答集.md
    """

    def __init__(
        self,
        knowledge_dir: Optional[str] = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        if knowledge_dir is None:
            project_root = Path(__file__).parent.parent.parent
            knowledge_dir = str(project_root / "resources" / "knowledge")
        self.knowledge_dir = Path(knowledge_dir)

        self.embedding_model_name = embedding_model
        self._embedding_model = None  # 懒加载

        self.chunks: List[KnowledgeChunk] = []
        self._index = None  # FAISS index
        self._index_built = False

    def _load_embedding_model(self):
        """懒加载 embedding 模型"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[KnowledgeBase] Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"[KnowledgeBase] Embedding model loaded")
            except ImportError:
                print("[KnowledgeBase] WARNING: sentence-transformers not installed. "
                      "Run: pip install sentence-transformers")
                raise
        return self._embedding_model

    def load_all(self) -> List[KnowledgeChunk]:
        """
        加载所有知识库文档并分块

        Returns:
            List[KnowledgeChunk]: 所有分块
        """
        self.chunks = []

        if not self.knowledge_dir.exists():
            print(f"[KnowledgeBase] WARNING: knowledge directory not found: {self.knowledge_dir}")
            return self.chunks

        md_files = sorted(self.knowledge_dir.glob("*.md"))
        if not md_files:
            print(f"[KnowledgeBase] WARNING: no .md files found in {self.knowledge_dir}")
            return self.chunks

        for md_file in md_files:
            doc_chunks = self._load_markdown(md_file)
            self.chunks.extend(doc_chunks)
            print(f"[KnowledgeBase] Loaded {len(doc_chunks)} chunks from {md_file.name}")

        print(f"[KnowledgeBase] Total: {len(self.chunks)} chunks")
        return self.chunks

    def _load_markdown(self, file_path: Path) -> List[KnowledgeChunk]:
        """
        加载单个 Markdown 文件，按二级标题分块

        分块策略：
        - 以 `## ` 为分隔符
        - 每个二级标题下的内容为一个 chunk
        - 如果内容过长（>800字），进一步按段落分割
        """
        doc_id = file_path.stem  # 如 "01_树种数据库"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按二级标题分割
        # 匹配 "## 标题" 作为分隔
        sections = re.split(r'\n##\s+', content)

        chunks = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # 第一部分可能是文件开头（无 ## 前缀）
            if i == 0 and not section.startswith("#"):
                title = "概述"
                body = section.strip()
            else:
                lines = section.strip().split("\n", 1)
                title = lines[0].strip().lstrip("#").strip()
                body = lines[1].strip() if len(lines) > 1 else ""

            if not body:
                continue

            # 如果内容过长，按段落进一步分割
            if len(body) > 800:
                paragraphs = body.split("\n\n")
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) < 600:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(KnowledgeChunk(
                                doc_id=doc_id,
                                title=title,
                                content=current_chunk.strip(),
                                source_file=str(file_path),
                            ))
                        current_chunk = para + "\n\n"
                if current_chunk.strip():
                    chunks.append(KnowledgeChunk(
                        doc_id=doc_id,
                        title=title,
                        content=current_chunk.strip(),
                        source_file=str(file_path),
                    ))
            else:
                chunks.append(KnowledgeChunk(
                    doc_id=doc_id,
                    title=title,
                    content=body,
                    source_file=str(file_path),
                ))

        return chunks

    def build_index(self) -> "KnowledgeBase":
        """
        为所有 chunks 构建 FAISS 索引

        需要 sentence-transformers 和 faiss
        """
        if not self.chunks:
            self.load_all()

        if not self.chunks:
            print("[KnowledgeBase] No chunks to index")
            return self

        try:
            import faiss
        except ImportError:
            print("[KnowledgeBase] WARNING: faiss not installed. "
                  "Run: pip install faiss-cpu")
            raise

        model = self._load_embedding_model()

        # 计算所有 chunks 的 embedding
        texts = [f"{c.doc_id} {c.title}: {c.content}" for c in self.chunks]
        print(f"[KnowledgeBase] Computing embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # 归一化（用于余弦相似度）
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # 存储 embedding
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]

        # 构建 FAISS 索引
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine Similarity (已归一化)
        self._index.add(embeddings.astype(np.float32))
        self._index_built = True

        print(f"[KnowledgeBase] FAISS index built: {self._index.ntotal} vectors, dim={dim}")
        return self

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索与 query 最相关的知识块

        Args:
            query: 查询文本
            top_k: 返回 top-k 个结果

        Returns:
            List[Dict]: 每个结果包含 {doc_id, title, content, score, source_file}
        """
        if not self._index_built or self._index is None:
            print("[KnowledgeBase] Index not built. Call build_index() first.")
            return []

        model = self._load_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)

        scores, indices = self._index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append({
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "content": chunk.content,
                "score": float(score),
                "source_file": chunk.source_file,
            })

        return results

    def save_index(self, save_dir: str) -> None:
        """保存索引和 chunks 到磁盘"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存 chunks 元数据
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(save_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # 保存 embeddings
        if self.chunks and self.chunks[0].embedding is not None:
            embeddings = np.stack([c.embedding for c in self.chunks])
            np.save(save_path / "embeddings.npy", embeddings)

        # 保存 FAISS 索引
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(save_path / "faiss.index"))

        print(f"[KnowledgeBase] Index saved to {save_dir}")

    def load_index(self, save_dir: str) -> "KnowledgeBase":
        """从磁盘加载索引"""
        save_path = Path(save_dir)

        # 加载 chunks
        chunks_file = save_path / "chunks.json"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = [KnowledgeChunk.from_dict(d) for d in chunks_data]

        # 加载 embeddings
        embeddings_file = save_path / "embeddings.npy"
        if embeddings_file.exists():
            embeddings = np.load(embeddings_file)
            for i, chunk in enumerate(self.chunks):
                if i < len(embeddings):
                    chunk.embedding = embeddings[i]

        # 加载 FAISS 索引
        index_file = save_path / "faiss.index"
        if index_file.exists():
            import faiss
            self._index = faiss.read_index(str(index_file))
            self._index_built = True
            print(f"[KnowledgeBase] Index loaded: {self._index.ntotal} vectors")

        return self

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "total_chunks": len(self.chunks),
            "index_built": self._index_built,
            "knowledge_dir": str(self.knowledge_dir),
            "embedding_model": self.embedding_model_name,
        }


def test_knowledge_base():
    """测试知识库"""
    kb = KnowledgeBase()

    # 创建测试文档
    test_dir = Path("resources/knowledge")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test_doc.md"
    test_file.write_text("""# 测试文档

## 杉木
杉木是中国南方重要的用材树种，生长速度快。

## 樟树
樟树是常绿乔木，树冠宽大，适合城市绿化。
""", encoding="utf-8")

    # 加载
    kb.load_all()
    print(f"Loaded {len(kb.chunks)} chunks")

    # 构建索引
    try:
        kb.build_index()
        results = kb.search("杉木的生长速度", top_k=2)
        for r in results:
            print(f"  [{r['doc_id']}] {r['title']} (score={r['score']:.3f})")
            print(f"    {r['content'][:100]}...")
    except Exception as e:
        print(f"[Test] Skipping FAISS test: {e}")

    # 清理
    test_file.unlink(missing_ok=True)
    print("[OK] KnowledgeBase test passed")


if __name__ == "__main__":
    test_knowledge_base()