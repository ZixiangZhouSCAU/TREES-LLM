"""
点云预处理模块（已弃用）
⚠️ DEPRECATED: 本模块不再被任何活跃端点使用，仅保留历史参考。
当前数据流：PointLLM.encode() → VQ tokenizer → token文本化 → GLM
本模块的 ground filter / clustering / 几何参数计算已不再在关键路径中。
"""

import numpy as np
from typing import Tuple, List, Optional, Callable

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


def estimate_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """估计点云法向量"""
    if not HAS_O3D:
        raise ImportError("open3d required")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k))
    return np.asarray(pcd.normals)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """体素下采样，返回下采样后的点和每个体素的点数"""
    if not HAS_O3D:
        raise ImportError("open3d required")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points), np.array([1] * len(down_pcd.points))


def remove_ground(points: np.ndarray, method: str = "cloth") -> np.ndarray:
    """
    滤除地面点

    Args:
        points: (N, 3) 点云
        method: "cloth" (布料模拟) 或 "height" (高度阈值)
    """
    if method == "height":
        ground_height = np.percentile(points[:, 2], 5)
        mask = points[:, 2] > ground_height + 0.3
        return points[mask]

    elif method == "cloth" and HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
        mask = np.ones(len(points), dtype=bool)
        mask[inliers] = False
        return points[mask]

    else:
        ground_height = np.percentile(points[:, 2], 10)
        return points[points[:, 2] > ground_height + 0.5]


def remove_outliers(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """离群点去除"""
    if not HAS_O3D:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return np.asarray(cl.points)


def cluster_trees(
    points: np.ndarray,
    eps: float = 0.5,
    min_points: int = 50,
) -> List[np.ndarray]:
    """
    基于欧氏聚类分割单棵树

    Args:
        points: 非地面点云 (N, 3)
        eps: 邻域半径 (米)
        min_points: 最小点数

    Returns:
        List of tree point clouds, each (Mi, 3)
    """
    if not HAS_O3D:
        return cluster_trees_numpy(points, eps, min_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps, min_points, False))

    trees = []
    tree_ids = set(labels[labels >= 0])
    for tree_id in tree_ids:
        mask = labels == tree_id
        trees.append(points[mask])
    return trees


def cluster_trees_numpy(points: np.ndarray, eps: float = 0.5, min_points: int = 50) -> List[np.ndarray]:
    """纯NumPy实现的欧氏聚类（无Open3D依赖）"""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    labels = np.full(len(points), -1, dtype=np.int32)
    current_label = 0

    for i in range(len(points)):
        if labels[i] != -1:
            continue
        indices = [i]
        labels[i] = current_label

        queue = [i]
        while queue:
            idx = queue.pop()
            neighbors = tree.query_ball_point(points[idx], eps)
            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = current_label
                    queue.append(nb)

        if len(indices) >= min_points:
            current_label += 1
        else:
            labels[labels == current_label] = -1

    trees = []
    for label in set(labels[labels >= 0]):
        trees.append(points[labels == label])
    return trees


def compute_tree_params(points: np.ndarray) -> dict:
    """
    从单棵树点云计算基础参数

    Returns:
        dict with: height, dbh_estimate, crown_width, n_points
    """
    xyz = points[:, :3] if points.shape[1] >= 3 else points

    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    height = z_max - z_min

    # 树冠宽度（XY平面投影的直径）
    centroid = xyz[:, :2].mean(axis=0)
    distances = np.linalg.norm(xyz[:, :2] - centroid, axis=1)
    crown_width = distances.max() * 2

    # DBH 估计（1.3m 高度附近）
    dbh_height = 1.3
    mask_dbh = np.abs(xyz[:, 2] - dbh_height) < 0.2
    if mask_dbh.sum() > 10:
        cross_section = xyz[mask_dbh, :2]
        cross_centroid = cross_section.mean(axis=0)
        dbh_estimate = 2 * np.median(np.linalg.norm(cross_section - cross_centroid, axis=1))
    else:
        dbh_estimate = 0.0

    return {
        "height": float(height),
        "dbh_estimate": float(dbh_estimate),
        "crown_width": float(crown_width),
        "n_points": len(points),
        "centroid": centroid.tolist(),
    }


class PointCloudPreprocessor:
    """
    点云预处理器 pipeline

    用法：
        preprocessor = PointCloudPreprocessor()
        result = preprocessor.process(raw_points)
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        remove_ground_method: str = "height",
        ground_threshold: float = 0.5,
        outlier_nb: int = 20,
        outlier_std: float = 2.0,
        cluster_eps: float = 0.6,
        cluster_min: int = 100,
    ):
        self.voxel_size = voxel_size
        self.remove_ground_method = remove_ground_method
        self.ground_threshold = ground_threshold
        self.outlier_nb = outlier_nb
        self.outlier_std = outlier_std
        self.cluster_eps = cluster_eps
        self.cluster_min = cluster_min

    def process(self, points: np.ndarray) -> List[dict]:
        """
        完整预处理流程

        Args:
            points: (N, 3) or (N, 6) 原始点云

        Returns:
            List of dicts, each containing tree point cloud and computed parameters
        """
        xyz = points[:, :3]

        # 1. 离群点去除
        xyz_clean = remove_outliers(xyz, self.outlier_nb, self.outlier_std)

        # 2. 地面滤除
        xyz_above_ground = remove_ground(xyz_clean, self.remove_ground_method)

        # 3. 体素下采样
        xyz_down, _ = voxel_downsample(xyz_above_ground, self.voxel_size)

        # 4. 树木聚类分割
        tree_clouds = cluster_trees(xyz_down, self.cluster_eps, self.cluster_min)

        # 5. 计算每棵树的参数
        results = []
        for i, tree_pts in enumerate(tree_clouds):
            params = compute_tree_params(tree_pts)
            results.append({
                "tree_id": i,
                "points": tree_pts,
                "params": params,
            })

        return results

    def save_trees(self, trees: List[dict], output_dir: str):
        """将分割后的树木保存为 npy 文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        for tree in trees:
            out_path = os.path.join(output_dir, f"tree_{tree['tree_id']:04d}.npy")
            np.save(out_path, tree["points"])
