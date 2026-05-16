#!/usr/bin/env python3
"""
sync_zotero.py - Zotero 文献同步脚本

功能：
  1. 从 Zotero Web API 读取文献元数据
  2. 对齐 PDF 文件到项目目录
  3. 生成 library_index.md

使用方法：
  python sync_zotero.py

配置：
  在脚本中设置 ZOTERO_LIBRARY_ID 和 ZOTERO_API_KEY
  或设置环境变量 ZOTERO_LIBRARY_ID 和 ZOTERO_API_KEY
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from pyzotero import zotero
except ImportError:
    print("错误：需要安装 pyzotero 库")
    print("运行: pip install pyzotero")
    sys.exit(1)


# ============ 配置 ============
# Zotero 库信息（请替换为你的实际值）
ZOTERO_LIBRARY_ID = "16973901"
ZOTERO_API_KEY = "ceKz8ZshyMok2W6ROwrJtfeJ"

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # 从 _docs 到根目录
PAPER_DIR = PROJECT_ROOT / "001literature" / "pdf"
BACKUP_DIR = PROJECT_ROOT / "001literature" / "zotero_backup"
INDEX_FILE = PROJECT_ROOT / "_docs" / "library_index.md"

# ============ 分类映射 ============
# 根据关键词自动分类论文
KEYWORD_CATEGORIES = {
    "树木检测与聚类": ["tree detection", "clustering", "instance segmentation", "street tree"],
    "树干参数提取": ["stem", "dbh", "diameter", "volume", "height", "crown"],
    "Leaf-Wood 分离": ["wood-leaf", "leaf-wood", "wood/leaf", "leaf separation"],
    "语义分割": ["semantic segmentation", "point cloud", "lidar"],
    "树种分类": ["species classification", "species identification"],
    "多源融合": ["fusion", "multispectral", "UAV", "TLS", "MLS"],
}


def get_zotero_items():
    """从 Zotero Web API 读取文献条目"""
    if ZOTERO_LIBRARY_ID == "YOUR_LIBRARY_ID":
        print("警告：未配置 Zotero 库信息，使用手动模式")
        return []

    try:
        zot = zotero.Zotero(ZOTERO_LIBRARY_ID, "user", ZOTERO_API_KEY)
        items = zot.everything(zot.items(item_type="journalArticle"))
        return items
    except Exception as e:
        print(f"从 Zotero 读取失败: {e}")
        return []


def extract_item_metadata(item: dict) -> dict:
    """提取单条文献的元数据"""
    data = item.get("data", {})

    # 提取作者
    authors = []
    for creator in data.get("creators", []):
        if "name" in creator:
            authors.append(creator["name"])
        elif "lastName" in creator and "firstName" in creator:
            authors.append(f"{creator['lastName']}, {creator['firstName']}")

    # 提取标签
    tags = [tag.get("tag", "") for tag in data.get("tags", [])]

    # 提取 PDF 附件信息
    attachment = None
    for att in data.get("attachments", []):
        if att.get("linkMode") == "imported_file" and att.get("mimetype") == "application/pdf":
            attachment = {
                "filename": att.get("filename", ""),
                "key": att.get("parentItem", ""),
            }
            break

    return {
        "key": item.get("key", ""),
        "title": data.get("title", "Untitled"),
        "authors": "; ".join(authors) if authors else "Unknown",
        "year": data.get("date", "")[:4] if data.get("date") else "N/A",
        "venue": data.get("publicationTitle", data.get("bookTitle", "N/A")),
        "doi": data.get("DOI", "N/A"),
        "abstract": data.get("abstractNote", ""),
        "tags": tags,
        "url": data.get("url", ""),
        "attachment": attachment,
    }


def auto_classify(item: dict) -> str:
    """根据标题和标签自动分类论文"""
    text = (item.get("title", "") + " " + " ".join(item.get("tags", []))).lower()

    for category, keywords in KEYWORD_CATEGORIES.items():
        if any(kw.lower() in text for kw in keywords):
            return category
    return "其他"


def load_existing_papers() -> list:
    """从 PDF 目录加载已存在的论文信息（手动模式）"""
    papers = []
    if not PAPER_DIR.exists():
        return papers

    for pdf_file in PAPER_DIR.glob("*.pdf"):
        # 从文件名提取基本信息（简化处理）
        title = pdf_file.stem.replace("_", " ")
        papers.append({
            "title": title,
            "filename": pdf_file.name,
            "path": str(pdf_file.relative_to(PROJECT_ROOT)),
            "year": "N/A",
            "authors": "待补充",
            "doi": "N/A",
            "category": "其他",
        })
    return papers


def generate_index(papers: list) -> str:
    """生成 library_index.md 内容"""
    # 按分类组织
    categories = {cat: [] for cat in KEYWORD_CATEGORIES.keys()}
    categories["其他"] = []

    for paper in papers:
        cat = paper.get("category", "其他")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(paper)

    lines = [
        "# 论文文献索引\n",
        f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"> 数据来源：Zotero 库 | 共 {len(papers)} 篇\n",
        "---\n\n",
    ]

    for cat, cat_papers in categories.items():
        if not cat_papers:
            continue
        lines.append(f"## {cat}\n\n")
        for i, p in enumerate(cat_papers, 1):
            lines.append(f"### {i}. {p.get('title', 'Untitled')}\n")
            lines.append(f"- **作者：** {p.get('authors', 'Unknown')}\n")
            lines.append(f"- **年份：** {p.get('year', 'N/A')}\n")
            lines.append(f"- **发表地：** {p.get('venue', 'N/A')}\n")
            if p.get('doi') and p['doi'] != 'N/A':
                lines.append(f"- **DOI：** [{p['doi']}](https://doi.org/{p['doi']})\n")
            if p.get('tags'):
                lines.append(f"- **关键词：** {', '.join(p.get('tags', []))}\n")
            if p.get('path'):
                lines.append(f"- **PDF：** `{p['path']}`\n")
            lines.append("\n")

    return "".join(lines)


def main():
    print("=" * 60)
    print("Zotero 文献同步脚本")
    print("=" * 60)
    print()

    # 确保目录存在
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # 获取 Zotero 条目
    items = get_zotero_items()

    if items:
        print(f"从 Zotero 读取到 {len(items)} 个条目")

        # 提取元数据
        papers = []
        for item in items:
            meta = extract_item_metadata(item)
            meta["category"] = auto_classify(meta)
            papers.append(meta)

        # 生成索引
        index_content = generate_index(papers)
        INDEX_FILE.write_text(index_content, encoding="utf-8")
        print(f"library_index.md 已更新，共 {len(papers)} 篇论文")

        # 记录同步清单
        manifest = {
            "last_sync": datetime.now().isoformat(),
            "total_items": len(items),
            "zotero_mode": True,
        }
    else:
        # 手动模式：从 PDF 目录加载
        print("使用手动模式：从 PDF 目录加载论文信息")
        papers = load_existing_papers()

        index_content = generate_index(papers)
        INDEX_FILE.write_text(index_content, encoding="utf-8")
        print(f"library_index.md 已更新，共 {len(papers)} 篇论文")

        manifest = {
            "last_sync": datetime.now().isoformat(),
            "total_items": len(papers),
            "zotero_mode": False,
        }

    # 保存同步清单
    manifest_file = BACKUP_DIR / "sync_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"同步清单已保存到: {manifest_file}")

    print()
    print("同步完成！")
    print(f"文献索引: {INDEX_FILE}")
    print(f"提示：查看 library_index.md 了解论文分类情况")


if __name__ == "__main__":
    main()
