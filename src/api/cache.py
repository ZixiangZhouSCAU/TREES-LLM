"""
分析结果缓存模块
上传点云时预先执行分析并缓存结果，用户提问时直接响应
解决痛点1：消除用户等待时间
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class CachedAnalysis:
    """缓存的单次分析结果"""
    file_name: str
    timestamp: str
    n_points: int
    ground_count: int
    tree_count: int
    # 单棵树参数
    params: Dict[str, Any]
    # 多棵树参数（分割后）
    trees_params: List[Dict[str, Any]]
    # 场景统计量（用于快速回答）
    scene_stats: Dict[str, Any]
    # PointLLM编码结果
    pointllm_analysis: Dict[str, Any]
    # LLM回答
    llm_answer: str


class AnalysisCache:
    """
    分析结果缓存（内存）
    线程安全，支持多用户并发
    """

    def __init__(self, max_entries: int = 50):
        self._cache: Dict[str, CachedAnalysis] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._access_order: List[str] = []  # LRU顺序

    def set(self, file_key: str, analysis: CachedAnalysis) -> None:
        """存储分析结果"""
        with self._lock:
            # LRU淘汰
            if len(self._cache) >= self._max_entries and file_key not in self._cache:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)

            self._cache[file_key] = analysis
            if file_key in self._access_order:
                self._access_order.remove(file_key)
            self._access_order.append(file_key)

    def get(self, file_key: str) -> Optional[CachedAnalysis]:
        """获取缓存结果（命中时更新LRU）"""
        with self._lock:
            if file_key in self._cache:
                self._access_order.remove(file_key)
                self._access_order.append(file_key)
                return self._cache[file_key]
            return None

    def has(self, file_key: str) -> bool:
        """检查是否已缓存"""
        return file_key in self._cache

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def size(self) -> int:
        """当前缓存条目数"""
        return len(self._cache)

    def keys(self) -> List[str]:
        """所有缓存的key"""
        return list(self._cache.keys())


# 全局缓存实例
_global_cache = AnalysisCache(max_entries=50)


def get_cache() -> AnalysisCache:
    """获取全局缓存实例"""
    return _global_cache


def make_file_key(file_name: str, file_size: int) -> str:
    """根据文件名和大小生成缓存key"""
    import hashlib
    key = f"{file_name}_{file_size}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _to_py(val):
    """Convert numpy types to JSON-serializable Python types"""
    if hasattr(val, 'item'):
        return val.item()
    return val


def _sanitize(v):
    """Recursively clean any value for JSON serialization"""
    if isinstance(v, dict):
        return {k: _sanitize(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_sanitize(item) for item in v]
    return _to_py(v)


def build_cached_analysis(
    file_name: str,
    n_points: int,
    ground_count: int,
    tree_count: int,
    params: Dict,
    trees_params: List[Dict],
    pointllm_analysis: Dict,
    llm_answer: str,
    scene_stats: Optional[Dict] = None,
) -> CachedAnalysis:
    """Build cached analysis result with sanitized types"""
    if scene_stats is not None:
        # Use caller-provided scene_stats directly
        scene_stats = _sanitize(scene_stats)
    else:
        # Fallback: reconstruct from pointllm_analysis
        z = pointllm_analysis.get("geometry", {})
        layers = pointllm_analysis.get("layers", {})
        height_range_val = _sanitize(z.get("z_min", 0))
        scene_stats = _sanitize({
            "n_points": int(n_points),
            "ground_count": int(ground_count),
            "tree_count": int(tree_count),
            "total_trees": len(trees_params) if trees_params else 1,
            "avg_height": float(round(sum(t.get("height", 0) for t in trees_params) / max(len(trees_params), 1), 2)) if trees_params else float(params.get("height", 0)),
            "avg_dbh": float(round(sum(t.get("dbh", 0) for t in trees_params) / max(len(trees_params), 1), 1)) if trees_params else float(params.get("dbh", 0)),
            "height_range": [height_range_val, height_range_val],
            "layers": _sanitize(layers),
        })

    return CachedAnalysis(
        file_name=str(file_name),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_points=int(n_points),
        ground_count=int(ground_count),
        tree_count=int(tree_count),
        params=_sanitize(params),
        trees_params=_sanitize(trees_params),
        scene_stats=_sanitize(scene_stats),
        pointllm_analysis={},  # Don't cache PointLLM analysis to avoid numpy issues
        llm_answer=str(llm_answer),
    )