"""
可视化工具
"""

import numpy as np
from typing import Optional, List


def plot_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    point_size: float = 1.0,
):
    """使用Open3D可视化点云"""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=title, point_show_normal=False)
    except ImportError:
        print("open3d not available, skipping visualization")


def plot_tree_parameters(
    tree_ids: List[str],
    heights: List[float],
    dbhs: List[float],
    save_path: Optional[str] = None,
):
    """绘制树木参数对比图"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(heights)), heights, color='forestgreen', alpha=0.8)
    axes[0].set_title('树高 (m)')
    axes[0].set_xlabel('Tree ID')
    axes[0].set_ylabel('Height (m)')

    axes[1].bar(range(len(dbhs)), dbhs, color='sienna', alpha=0.8)
    axes[1].set_title('胸径 DBH (m)')
    axes[1].set_xlabel('Tree ID')
    axes[1].set_ylabel('DBH (m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
