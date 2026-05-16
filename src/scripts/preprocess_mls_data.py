"""
MLS数据预处理脚本
读取原始 .las/.laz 文件，执行地面滤除、树木分割、参数提取
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np

from src.data.preprocessing import PointCloudPreprocessor


def main():
    parser = argparse.ArgumentParser(description="MLS点云数据预处理")
    parser.add_argument("--input", type=str, required=True, help="原始数据目录 (.las/.laz)")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="体素大小(米)")
    parser.add_argument("--cluster-eps", type=float, default=0.6, help="聚类半径(米)")
    parser.add_argument("--cluster-min", type=int, default=100, help="最小聚类点数")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    trees_dir = output_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = PointCloudPreprocessor(
        voxel_size=args.voxel_size,
        cluster_eps=args.cluster_eps,
        cluster_min=args.cluster_min,
    )

    las_files = list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz"))
    print(f"Found {len(las_files)} files to process")

    all_results = []

    for las_file in tqdm(las_files, desc="Processing"):
        try:
            import laspy
            las = laspy.read(str(las_file))
            points = np.vstack([las.x, las.y, las.z]).T

            # 预处理
            trees = preprocessor.process(points)

            # 保存每棵树
            for tree in trees:
                tree_id = f"{las_file.stem}_tree_{tree['tree_id']:04d}"
                out_path = trees_dir / f"{tree_id}.npy"
                np.save(str(out_path), tree["points"])

                result = {
                    "file": las_file.name,
                    "tree_id": tree_id,
                    "n_points": tree["params"]["n_points"],
                    "height": tree["params"]["height"],
                    "dbh_estimate": tree["params"]["dbh_estimate"],
                    "crown_width": tree["params"]["crown_width"],
                    "centroid": tree["params"]["centroid"],
                }
                all_results.append(result)

        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")

    # 保存汇总
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Processed {len(all_results)} trees")
    print(f"Results saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
