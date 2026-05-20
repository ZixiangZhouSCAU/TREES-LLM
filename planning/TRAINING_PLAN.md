# 训练计划 — 路线 A：架构对齐

> 文档日期：2026-05-20
> 参考论文：3DCITY-LLM (CVPR 2024) 粗到细编码 + 指令驱动路由

---

## 一、训练目标

训练 3 个 FeatureProjector（Object / Relationship / Scene），让点云特征对齐到 LLM 文本空间。

**不训练 LLM 本体**（GLM-4-Flash 冻结），只训练投影层（约 6M 参数）。

---

## 二、训练架构

```
输入：树木点云 + 几何参数
    │
    ├── ObjectProjector (864→512)    # 768(PointBERT) + 64(几何) + 32(物种)
    ├── RelationshipProjector (48→256)  # 16(距离) + 16(高差) + 16(竞争)
    └── SceneProjector (960→512)     # 768(PointBERT) + 64(密度) + 64(高度) + 64(郁闭度)
              │
    只更新这 3 个 Projector（约 6M 参数）
    PointBERT 和 GLM-4-Flash 冻结
```

---

## 三、训练数据

### 数据来源

| 来源 | 预估数量 | 用途 |
|------|---------|------|
| 实测 50 棵树（样地 A/B/C） | ~750 条 | 核心训练集 |
| TreeLearn 公开数据集 | ~3000 条 | 数据增强 |
| **总计** | **~3750 条** | Stage 1 + Stage 2 |

### 数据格式（JSONL）

```json
// Stage 1: ObjectCaption
{"task": "object_caption", "tree_id": "A01", "height": 12.3, "dbh": 35.2, "crown_width": 4.5, "carbon_stock": 52.1, "caption": "这棵玉兰树高12.3米..."}

// Stage 2: SceneAnalysis
{"task": "scene_analysis", "plot_id": "PlotB", "n_trees": 17, "scene_stats": {...}, "question": "这片样地情况如何？", "answer": "样地B共17棵..."}

// Stage 2: ScenePlanning
{"task": "scene_planning", "plot_id": "PlotC", "n_trees": 25, "question": "怎么管理？", "answer": "建议间伐..."}
```

### 数据生成

```bash
# 从实测数据自动生成训练数据
python scripts/generate_training_data.py \
    --data-dir data/collected \
    --metadata data/tree_metadata.json \
    --output data/training/tree_training_data.jsonl
```

---

## 四、两阶段训练

### Stage 1：特征对齐（Feature Alignment）

- **数据**：ObjectCaption（单树描述）
- **目标**：让 Projector 学会将点云特征映射为有意义的描述
- **损失**：Caption Loss（交叉熵）
- **参数**：只更新 3 个 Projector

```bash
python src/training/trainer.py --stage 1 --epochs 10 --lr 1e-3 --batch-size 8
```

### Stage 2：指令微调（Instruction Tuning）

- **数据**：SceneAnalysis + ScenePlanning
- **目标**：让 Projector 支持专业林业分析和管理规划
- **损失**：Analysis Loss
- **参数**：Projector（降低学习率）

```bash
python src/training/trainer.py --stage 2 --epochs 5 --lr 5e-4 --batch-size 4
```

---

## 五、训练环境

| 项目 | 配置 |
|------|------|
| GPU | RTX 4060 Laptop (8GB) |
| PyTorch | 2.6.0+cu124 |
| 预计训练时间 | Stage 1: ~30min, Stage 2: ~20min |
| 预计显存占用 | ~3GB（PointBERT encoder + projector） |

---

## 六、评估方式

### 定量评估
- BLEU-4（描述质量）
- ROUGE-L（分析准确度）
- METEOR（语义匹配度）

### 定性评估
- 人工检查生成的描述是否准确反映树木参数
- 对比训练前后的 LLM 回答质量

---

## 七、时间线

| 时间 | 工作 |
|------|------|
| 5月20-23日 | 数据采集 |
| 5月24-26日 | 生成训练数据 + 构建 RAG 知识库 |
| 5月27-28日 | Stage 1 训练 |
| 5月29-30日 | Stage 2 训练 + 评估 |
| 6月1日后 | 端到端集成测试 |

---

## 八、注意事项

1. **数据采集时必须记录 Ground Truth**：每棵树的 DBH/树高/冠幅（人工测量）
2. **训练数据生成脚本** 会自动从点云提取参数 + 生成描述，无需手写
3. **RAG 知识库** 是独立模块，不需要训练，只需整理好文档后构建索引
4. **如果训练数据不够**，可以先不训练 Projector，直接使用随机初始化的投影层，系统仍可工作（只是投影特征是随机的）