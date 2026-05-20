"""
训练数据生成工具
从实测点云数据 + Ground Truth → 自动生成 JSONL 训练数据

功能：
1. 读取实测点云 + 树元数据（DBH/树高/冠幅）
2. 自动提取几何参数（复用 service.py 的算法）
3. 生成 ObjectCaption / SceneAnalysis / ScenePlanning 三类训练样本
4. 数据增强：同一棵树 × 多种提问风格 × 多种分析角度

用法：
    # 生成训练数据
    python scripts/generate_training_data.py \
        --data-dir data/collected \
        --metadata data/tree_metadata.json \
        --output data/training/tree_training_data.jsonl

    # 同时生成 tree_metadata.json 和 plot_metadata.json
    python scripts/generate_training_data.py \
        --data-dir data/collected \
        --collect-metadata \
        --output data/training/tree_training_data.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.service import load_point_cloud, extract_precise_params, segment_trees


# ============ 描述模板（Stage 1 ObjectCaption）============

OBJECT_CAPTION_TEMPLATES = [
    # 风格 1：专业测量报告
    (
        "这棵{tree_id}树高{h}米，胸径{dbh}厘米，"
        "冠幅约{cw}米。整体生长{health}，"
        "处于{growth_stage}阶段。"
    ),
    # 风格 2：简洁描述
    (
        "{tree_id}：高{h}米，DBH={dbh}厘米，"
        "冠幅{cw}米，属于{growth_stage}。"
    ),
    # 风格 3：详细评估
    (
        "对这棵编号{tree_id}的分析表明："
        "树高{h}米，胸径{dbh}厘米，"
        "树冠体积约{cv}立方米，"
        "估算碳储量约{cs}千克。"
    ),
    # 风格 4：养护建议
    (
        "{tree_id}树高约{h}米，胸径{dbh}厘米。"
        "根据{health}状况判断，"
        "{suggest}。"
    ),
    # 风格 5：林学分析
    (
        "从林学角度分析："
        "该{tree_id}树高{h}米，"
        "高径比约{ratio}，"
        "属于{growth_stage}的{growth_type}。"
        "{comment}"
    ),
]

# Stage 2 分析任务模板
SCENE_ANALYSIS_TEMPLATES = [
    (
        "样地{plot_id}共{n}棵树木，"
        "平均树高{avg_h}米，平均胸径{avg_dbh}厘米，"
        "总碳储量约{total_c}千克。"
        "林分类型为{forest_type}，"
        "整体{growth_stage}。"
    ),
    (
        "{plot_id}调查结果显示："
        "共测量{n}棵树木，"
        "树高范围{h_range}米，"
        "总碳汇量约{total_c}千克。"
        "{suggest}"
    ),
]

MANAGEMENT_TEMPLATES = [
    (
        "建议{action}。"
        "优先级：{priority}。"
        "理由：{reason}。"
    ),
    (
        "基于本次调查结果，"
        "对{plot_id}提出以下管理建议："
        "{actions}。"
    ),
]

# 生长阶段映射
GROWTH_STAGES = {
    (0, 5): "幼龄林",
    (5, 15): "中龄林",
    (15, 25): "成熟林",
    (25, 999): "过熟林",
}
GROWTH_TYPES = {
    (0, 5): "速生期",
    (5, 15): "旺盛期",
    (15, 25): "稳定期",
    (25, 999): "衰老期",
}


def get_growth_stage(height: float) -> tuple:
    for (lo, hi), stage in GROWTH_STAGES.items():
        if lo <= height < hi:
            return stage
    return "成熟林"


def get_growth_type(height: float) -> str:
    for (lo, hi), gtype in GROWTH_TYPES.items():
        if lo <= height < hi:
            return gtype
    return "稳定期"


def format_height(h: float) -> str:
    return f"{h:.1f}"


def format_dbh(d: float) -> str:
    return f"{d:.1f}"


def format_cw(cw: float) -> str:
    return f"{cw:.1f}"


def generate_object_caption(
    tree_id: str,
    params: Dict,
    template_id: Optional[int] = None,
) -> str:
    """生成单树描述（ObjectCaption 样本）"""
    height = params.get("height", 0)
    dbh = params.get("dbh", 0)
    crown_width = params.get("crown_width", 0)
    crown_volume = params.get("crown_volume", 0)
    carbon = params.get("carbon_stock", 0)
    center_xy = params.get("center_xy", [0, 0])

    stage = get_growth_stage(height)
    gtype = get_growth_type(height)

    if dbh > 0:
        ratio = height / (dbh / 100)
    else:
        ratio = 0

    # 判断健康状态
    if crown_width / max(height, 0.1) > 0.4:
        health = "良好"
        suggest = "建议保持常规养护"
        comment = "冠层发育正常。"
    elif crown_width / max(height, 0.1) > 0.2:
        health = "一般"
        suggest = "注意观察生长状况"
        comment = "冠层略有收缩，需关注。"
    else:
        health = "偏弱"
        suggest = "建议检查土壤和营养状况"
        comment = "树冠受压明显，建议加强管理。"

    # 森林类型推断
    if height < 5:
        forest_type = "幼龄林"
    elif height < 15:
        forest_type = "中龄林"
    else:
        forest_type = "成熟林"

    # 渲染模板
    if template_id is None:
        template_id = random.randint(0, len(OBJECT_CAPTION_TEMPLATES) - 1)

    template = OBJECT_CAPTION_TEMPLATES[template_id]
    return template.format(
        tree_id=tree_id,
        h=format_height(height),
        dbh=format_dbh(dbh),
        cw=format_cw(crown_width),
        cv=f"{crown_volume:.2f}",
        cs=f"{carbon:.1f}",
        ratio=f"{ratio:.0f}",
        health=health,
        growth_stage=stage,
        growth_type=gtype,
        suggest=suggest,
        comment=comment,
        forest_type=forest_type,
    )


def generate_scene_analysis(
    plot_id: str,
    trees_params: List[Dict],
    scene_stats: Dict,
) -> str:
    """生成场景分析描述（SceneAnalysis 样本）"""
    n = len(trees_params)
    heights = [p.get("height", 0) for p in trees_params if p.get("height", 0) > 0]
    dbhs = [p.get("dbh", 0) for p in trees_params if p.get("dbh", 0) > 0]
    carbons = [p.get("carbon_stock", 0) for p in trees_params]

    avg_h = np.mean(heights) if heights else 0
    avg_dbh = np.mean(dbhs) if dbhs else 0
    total_c = sum(carbons)

    h_min = min(heights) if heights else 0
    h_max = max(heights) if heights else 0

    stage = get_growth_stage(avg_h)

    if avg_h < 5:
        forest_type = "幼龄林"
        suggest = "建议加强幼林抚育，定期除草松土。"
    elif avg_h < 15:
        forest_type = "中龄林"
        suggest = "建议进行适度间伐，改善林木生长空间。"
    else:
        forest_type = "成熟林"
        suggest = "已达采伐年龄，可根据经营目标规划采伐。"

    template = random.choice(SCENE_ANALYSIS_TEMPLATES)
    return template.format(
        plot_id=plot_id,
        n=n,
        avg_h=format_height(avg_h),
        avg_dbh=format_dbh(avg_dbh),
        h_range=f"{format_height(h_min)}-{format_height(h_max)}",
        total_c=f"{total_c:.1f}",
        forest_type=forest_type,
        growth_stage=stage,
        suggest=suggest,
    )


def generate_management_suggestion(
    plot_id: str,
    trees_params: List[Dict],
    scene_stats: Dict,
) -> str:
    """生成管理建议（ScenePlanning 样本）"""
    n = len(trees_params)
    heights = [p.get("height", 0) for p in trees_params if p.get("height", 0) > 0]
    avg_h = np.mean(heights) if heights else 0

    # 识别风险树木
    risk_trees = []
    for p in trees_params:
        h = p.get("height", 0)
        d = p.get("dbh", 1)
        if d > 0 and h / (d / 100) > 70:
            risk_trees.append(p.get("tree_id", "?"))

    actions = []
    if avg_h < 5:
        actions.append("定株抚育，每亩保留60-80株")
        priority = "高"
        reason = "幼龄林密度过高影响生长"
    elif avg_h < 15:
        actions.append("抚育间伐，强度15-25%")
        actions.append("优先伐除枯死木和病虫害木")
        priority = "高"
        reason = "中龄林树冠开始郁闭"
    else:
        actions.append("规划主伐利用")
        actions.append("采伐前进行碳汇评估")
        priority = "中"
        reason = "成熟林已达采伐标准"

    if risk_trees:
        actions.append(f"优先处理风险树木：{', '.join(risk_trees)}")

    action_str = "；".join(actions)

    template = random.choice(MANAGEMENT_TEMPLATES)
    return template.format(
        action=action_str,
        priority=priority,
        reason=reason,
        plot_id=plot_id,
        actions=action_str,
    )


def augment_single_tree(
    tree_id: str,
    params: Dict,
    n_variations: int = 3,
) -> List[Dict]:
    """
    对单棵树生成多条数据增强样本

    同一棵树，用不同模板、不同角度生成多条 ObjectCaption
    """
    samples = []
    for i in range(min(n_variations, len(OBJECT_CAPTION_TEMPLATES))):
        caption = generate_object_caption(tree_id, params, template_id=i)
        samples.append({
            "task": "object_caption",
            "tree_id": tree_id,
            "height": params.get("height", 0),
            "dbh": params.get("dbh", 0),
            "crown_width": params.get("crown_width", 0),
            "carbon_stock": params.get("carbon_stock", 0),
            "center_xy": params.get("center_xy", [0, 0]),
            "caption": caption,
        })
    return samples


def load_metadata(metadata_path: str) -> tuple:
    """加载树元数据和样地元数据"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tree_meta = metadata.get("trees", {})
    plot_meta = metadata.get("plots", {})
    return tree_meta, plot_meta


def generate_from_directory(
    data_dir: str,
    metadata_path: Optional[str] = None,
    output_path: str = "data/training/tree_training_data.jsonl",
) -> List[Dict]:
    """
    从数据目录生成训练数据

    目录结构：
        data_dir/
        ├── plot_A/
        │   ├── tree_A01.ply
        │   ├── tree_A02.ply
        │   └── ...
        ├── plot_B/
        │   └── ...
        └── ...

    metadata.json 格式：
        {
            "trees": {
                "A01": {"species": "玉兰", "dbh": 35.2, "height": 12.3, "crown_width": 4.5},
                ...
            },
            "plots": {
                "plot_A": {"area_m2": 200, "location": "教学楼6旁"},
                ...
            }
        }
    """
    data_path = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载元数据
    tree_meta, plot_meta = ({}, {})
    if metadata_path and Path(metadata_path).exists():
        tree_meta, plot_meta = load_metadata(metadata_path)

    all_samples = []

    # 遍历每个样地
    for plot_dir in sorted(data_path.iterdir()):
        if not plot_dir.is_dir():
            continue

        plot_id = plot_dir.name
        plot_info = plot_meta.get(plot_id, {})
        area_m2 = plot_info.get("area_m2", 100.0)

        # 读取该样地所有点云
        ply_files = sorted(plot_dir.glob("*.ply"))
        if not ply_files:
            continue

        # 如果是单文件（整块样地），先做 DBSCAN 分割
        if len(ply_files) == 1:
            try:
                points = load_point_cloud(str(ply_files[0]))
                trees = segment_trees(points, eps=0.5, min_samples=50)
            except Exception as e:
                print(f"[WARN] Failed to segment {ply_files[0]}: {e}")
                trees = []
        else:
            # 多文件，每文件一棵
            trees = []
            for pf in ply_files:
                try:
                    pts = load_point_cloud(str(pf))
                    trees.append(pts)
                except Exception as e:
                    print(f"[WARN] Failed to load {pf}: {e}")

        if not trees:
            continue

        print(f"[{plot_id}] Found {len(trees)} trees")

        # 逐树提取参数 + 生成 ObjectCaption 样本
        trees_params = []
        for i, tree_pts in enumerate(trees):
            params = extract_precise_params(tree_pts)

            # 推断 tree_id（从文件名或编号）
            if len(ply_files) > 1:
                tree_id = ply_files[i].stem.replace("tree_", "").replace("Tree", "")
            else:
                tree_id = f"{plot_id}_{i+1}"

            # 合并元数据（优先用实测值）
            if tree_id in tree_meta:
                meta = tree_meta[tree_id]
                params.setdefault("species", meta.get("species"))
                # 实测值更准确，优先使用

            params["tree_id"] = tree_id
            params["center_xy"] = [
                float(tree_pts[:, 0].mean()),
                float(tree_pts[:, 1].mean()),
            ]
            trees_params.append(params)

            # ObjectCaption 样本（每棵 3 条）
            obj_samples = augment_single_tree(tree_id, params, n_variations=3)
            all_samples.extend(obj_samples)

        # SceneAnalysis 样本（每个样地 5 条）
        scene_stats = {
            "total_trees": len(trees_params),
            "avg_height": np.mean([p["height"] for p in trees_params]),
            "avg_dbh": np.mean([p["dbh"] for p in trees_params]),
            "total_carbon": sum(p["carbon_stock"] for p in trees_params),
        }

        for _ in range(5):
            scene_desc = generate_scene_analysis(plot_id, trees_params, scene_stats)
            all_samples.append({
                "task": "scene_analysis",
                "plot_id": plot_id,
                "n_trees": len(trees_params),
                "scene_stats": scene_stats,
                "question": f"请分析{plot_id}的整体情况。",
                "answer": scene_desc,
            })

        # ScenePlanning 样本（每个样地 2 条）
        for _ in range(2):
            mgmt = generate_management_suggestion(plot_id, trees_params, scene_stats)
            all_samples.append({
                "task": "scene_planning",
                "plot_id": plot_id,
                "n_trees": len(trees_params),
                "scene_stats": scene_stats,
                "question": f"对{plot_id}应采取什么管理措施？",
                "answer": mgmt,
            })

    # 保存为 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n生成完成：共 {len(all_samples)} 条样本 → {output_path}")

    # 统计
    task_counts = {}
    for s in all_samples:
        task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1
    print("各任务分布：")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} 条")

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="生成训练数据")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="数据目录（包含各样地子目录）")
    parser.add_argument("--metadata", type=str, default=None,
                        help="树元数据 JSON 文件路径")
    parser.add_argument("--output", type=str,
                        default="data/training/tree_training_data.jsonl",
                        help="输出 JSONL 文件路径")
    parser.add_argument("--n-variations", type=int, default=3,
                        help="每棵树生成多少条 ObjectCaption 变体")

    args = parser.parse_args()

    generate_from_directory(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()