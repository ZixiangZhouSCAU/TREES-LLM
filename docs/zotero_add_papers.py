#!/usr/bin/env python3
"""
add_papers_to_zotero.py
根据 DOI 批量添加到 Zotero

使用方法：
  python add_papers_to_zotero.py

在 PAPERS 列表中添加论文的 DOI，脚本会自动获取元数据并添加到 Zotero。
"""

import os
import sys

try:
    from pyzotero import zotero
except ImportError:
    print("错误：需要安装 pyzotero")
    print("运行: pip install pyzotero")
    sys.exit(1)


# ============ 配置 ============
ZOTERO_LIBRARY_ID = "16973901"
ZOTERO_API_KEY = "ceKz8ZshyMok2W6ROwrJtfeJ"
TARGET_COLLECTION = "ForC-MLS"  # 目标文库名称

# ============ 要添加的论文（按 DOI）============

PAPERS = [
    # === 核心相关论文 (★★★★★) ===
    "10.5194/isprs-annals-X-4-W1-2022-721-2023",  # PointNet++ MLS树木提取
    "10.3389/frsen.2026.1774149",  # 低成本MLS算法对比
    "10.3390/f130801245",  # 街道树木分割几何特征
    "10.1016/j.foreco.2022.120065",  # Ground-UAV LiDAR QSM融合
    "10.3390/f15081375",  # YOLOTree RGB+LiDAR冠幅估算
    "10.3390/rs17162805",  # 语义感知跨模态迁移
    "10.3390/rs17121996",  # MTCDNet多模态特征融合
    "10.1016/j.isprsjprs.2023.01.012",  # 3D CNN多光谱-LiDAR融合
    "10.3390/rs17213515",  # DA-GSGTNet图Transformer分割
    "10.1016/j.rse.2024.113162",  # DSM融合3D光合性状

    # === 高相关论文 (★★★★) ===
    "10.3390/s25010188",  # 拓扑检查街道树木分割
    "10.1109/JSTARS.2021.3126542",  # 自动树木分割参数提取
    "10.3390/f15061043",  # UAV-LiDAR混合方法
    "10.3390/rs16040608",  # 区域生长+超体素分割
    "10.1016/j.ophoto.2023.100039",  # 高密度多光谱激光扫描
    "10.3390/rs17132245",  # PointNet++ vs XGBoost对比
    "10.3390/rs17010093",  # 多光谱点云构建
    "10.3389/fenvs.2022.960083",  # UAV-LiDAR树分支重建
    "10.3390/drones8120744",  # 无人机LiDAR-多光谱集成
    "10.1007/s41064-025-00369-4",  # 多光谱LiDAR 3D深度学习
    "10.1016/j.ophoto.2023.100037",  # Mask R-CNN+DETR实例分割

    # === 树干参数提取相关 ===
    "10.1371/journal.pone.0196004",  # Mobile LiDAR城市树木参数提取
    "10.3390/rs14122753",  # UAV LiDAR树干点提取
    "10.3390/f15040590",  # MLS胸径提取
    "10.3390/rs17233888",  # 低成本设备DBH估算对比
    "10.1016/j.srs.2022.100050",  # ALS树干曲线体积
    "10.1016/j.rse.2026.115246",  # MLS立木体积估算
    "10.3390/rs14143344",  # 背包MLS树木表型
    "10.5194/isprs-annals-X-2-W2-2025-49-2025",  # UAV图像DBH蒙特卡洛
    "10.5194/isprs-archives-XLVIII-2-W11-2025-55-2025",  # 冠层下UAV路径规划
    "10.1016/j.plaphe.2026.100171",  # UAV RGB-LiDAR多模态树种参数

    # === 综述与趋势 ===
    "10.3390/rs17213557",  # UAV vs TLS精度对比
    "10.3389/fsci.2024.1383659",  # 数字孪生地球
    "10.1111/gcb.17473",  # TLS树高异速生长
]


def add_paper_by_doi(doi: str, collection_key: str = None) -> dict:
    """通过 DOI 添加论文到 Zotero"""
    import requests

    # 清理 DOI
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    if doi.startswith("doi:"):
        doi = doi.replace("doi:", "")

    # 通过 CrossRef API 获取元数据
    try:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"Accept": "application/json"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()["message"]

        # 提取关键字段
        title = data.get("title", [""])[0] if data.get("title") else "Unknown"
        authors = []
        for creator in data.get("author", []):
            name = f"{creator.get('given', '')} {creator.get('family', '')}".strip()
            if name:
                authors.append({"creatorType": "author", "name": name})

        year = str(data.get("published-print", data.get("published-online", {})).get("date-parts", [[None]])[0][0] or "")

        journal = data.get("container-title", [""])[0] if data.get("container-title") else ""

        item = {
            "itemType": "journalArticle",
            "title": title,
            "creators": authors,
            "date": year,
            "publicationTitle": journal,
            "DOI": doi,
            "url": f"https://doi.org/{doi}",
            "abstractNote": data.get("abstract", "").replace("<jats:p>", "").replace("</jats:p>", ""),
            "collections": [collection_key],  # 创建时直接添加到文库
        }

        return item

    except Exception as e:
        print(f"  获取元数据失败: {doi} - {e}")
        return None


def find_or_create_collection(zot, name: str) -> str:
    """查找或创建指定名称的文库"""
    collections = zot.collections()
    for col in collections:
        if col["data"]["name"] == name:
            print(f"找到文库: {name} (key: {col['key']})")
            return col["key"]

    # 创建新文库
    print(f"创建新文库: {name}")
    new_col = zot.create_collections([{"name": name, "parentCollection": False}])
    return new_col[0]["key"]


def main():
    print("=" * 60)
    print("批量添加论文到 Zotero")
    print("=" * 60)
    print(f"目标文库: {TARGET_COLLECTION}")
    print(f"将添加 {len(PAPERS)} 篇论文")
    print()

    # 连接到 Zotero
    zot = zotero.Zotero(ZOTERO_LIBRARY_ID, "user", ZOTERO_API_KEY)

    # 查找或创建目标文库
    collection_key = find_or_create_collection(zot, TARGET_COLLECTION)

    # 检查文库中已存在的论文（避免重复）
    try:
        collection_items = zot.collection_items(collection_key, item_type="journalArticle")
        existing_dois = {item["data"].get("DOI", "").lower() for item in collection_items}
        print(f"文库中已有 {len(existing_dois)} 篇期刊论文")
    except Exception:
        existing_dois = set()
        print("无法获取文库已有论文，将全部添加")

    # 添加论文
    added = 0
    skipped = 0
    failed = []

    for i, doi in enumerate(PAPERS, 1):
        doi_clean = doi.strip().lower()
        print(f"\n[{i}/{len(PAPERS)}] 处理: {doi}")

        # 检查是否已存在
        if doi_clean in existing_dois:
            print(f"  已存在，跳过")
            skipped += 1
            continue

        # 获取元数据
        item_data = add_paper_by_doi(doi, collection_key)
        if item_data is None:
            failed.append(doi)
            continue

        # 添加到 Zotero
        try:
            result = zot.create_items([item_data])
            item_key = list(result["successful"].keys())[0]
            print(f"  添加成功: {item_data['title'][:60]}...")
            added += 1
        except Exception as e:
            print(f"  添加失败: {e}")
            failed.append(doi)

    print()
    print("=" * 60)
    print(f"完成！新增 {added} 篇，跳过 {skipped} 篇（已存在），失败 {len(failed)} 篇")
    if failed:
        print(f"失败的 DOI: {failed}")
    print("=" * 60)

    # 提示运行同步
    print("\n建议运行 python sync_zotero.py 更新本地索引")


if __name__ == "__main__":
    main()
