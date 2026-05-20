"""
推理脚本 - 快速体验完整 pipeline
不需要训练，直接使用 GLM-4-Flash API
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.service import TreeAnalysisService


def main():
    parser = argparse.ArgumentParser(description="TREES-LLM 推理")
    parser.add_argument("--input", type=str, required=True, help="点云文件 (.ply/.las/.laz/.npy)")
    parser.add_argument("--question", type=str, default=None, help="问答模式的问题")
    parser.add_argument("--report", action="store_true", help="生成报告模式")
    args = parser.parse_args()

    service = TreeAnalysisService()

    if args.question:
        print(f"\n问题: {args.question}")
        print("-" * 40)
        result = service.analyze(args.input, question=args.question)
        if result.get("success"):
            print(f"回答: {result['answer']}")
        else:
            print(f"错误: {result.get('error', '未知错误')}")

    elif args.report:
        print("正在提取参数并生成报告...")
        print("-" * 40)
        result = service.multi_analyze(args.input)
        if result.get("success"):
            report = service.generate_report(
                trees_data=result.get("trees_params"),
                report_type="standard",
            )
            print(report.get("report_text", "报告生成失败"))
        else:
            print(f"错误: {result.get('error', '未知错误')}")

    else:
        print("正在提取树木参数...")
        print("-" * 40)
        result = service.analyze(args.input)
        if result.get("success"):
            params = result["params"]
            print(f"树高:       {params['height']} m")
            print(f"胸径(DBH):  {params['dbh']} cm")
            print(f"冠幅:       {params['crown_width']} m")
            print(f"碳储量:     {params['carbon_stock']} kg")
            print(f"碳汇价值:   ¥{params['carbon_value']}")
            print("-" * 40)
            print(f"方法: {result['method']}")
        else:
            print(f"错误: {result.get('error', '未知错误')}")


if __name__ == "__main__":
    main()