"""
推理脚本 - 快速体验完整pipeline
不需要训练，可直接使用 Claude API
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import PointCloudPreprocessor, compute_tree_params
from src.api.services import TreesService


def main():
    parser = argparse.ArgumentParser(description="TREES-LLM 推理")
    parser.add_argument("--input", type=str, required=True, help="点云文件 (.las/.laz/.npy)")
    parser.add_argument("--question", type=str, default=None, help="问答模式的问题")
    parser.add_argument("--report", action="store_true", help="生成报告模式")
    parser.add_argument("--config", type=str, default="src/configs/inference.yaml")
    args = parser.parse_args()

    service = TreesService(config_path=args.config)

    if args.question:
        # 问答模式
        print(f"\n问题: {args.question}")
        print("-" * 40)
        import asyncio
        answer = asyncio.run(service.answer_question(args.input, args.question))
        print(f"回答: {answer}")

    elif args.report:
        # 报告模式
        print("正在提取参数并生成报告...")
        print("-" * 40)

        import asyncio
        tree_data = asyncio.run(service.extract_params(args.input))

        if tree_data.get("success"):
            report = asyncio.run(
                service.generate_report([tree_data["params"]], report_type="standard")
            )
            print(report["report_text"])
        else:
            print(f"错误: {tree_data.get('error')}")

    else:
        # 参数提取模式
        print("正在提取树木参数...")
        print("-" * 40)

        import asyncio
        result = asyncio.run(service.extract_params(args.input))

        if result.get("success"):
            params = result["params"]
            print(f"树高:     {params['height']} m")
            print(f"胸径(DBH): {params['dbh']} cm")
            print(f"冠幅:     {params['crown_width']} m")
            print(f"碳储量:   {params['carbon_stock']} kg")
            print("-" * 40)
            print("置信度:")
            for k, v in result["confidence"].items():
                print(f"  {k}: {v:.0%}")
        else:
            print(f"错误: {result.get('error')}")


if __name__ == "__main__":
    main()
