# TREES-LLM 开发计划

> **目标**: 竞赛/课程展示 — 2个月内完成端到端林业点云智能分析系统
> **更新日期**: 2026-05-17

---

## 项目概述

**TREES-LLM** (Tree Understanding and Reasoning Enhanced System) — 基于 PointLLM 架构的林业点云智能分析平台。核心技术路线：**ULIP-2 PointBERT 编码器 → VQ-VAE 量化 → GLM-4-Flash 推理**，零训练，调用云端 LLM API。

### 技术栈

| 层级 | 技术 |
|------|------|
| 点云编码器 | ULIP-2 PointBERT (Objaverse 预训练, 768-dim) |
| 量化器 | VQ-VAE (2048 codebook, 128-dim) |
| 语言模型 | GLM-4-Flash (智谱AI 云端 API) |
| 后端 | FastAPI + uvicorn |
| 前端 | Three.js (原生 JS, 无构建工具) |
| 环境 | Python 3.12, PyTorch 2.6+CUDA12.4 |

---

## 8周详细计划

### 第1周：端到端跑通

**目标**: 点云上传 → API 返回 → 前端显示，整条链路跑通

| Day | 任务 | 状态 |
|-----|------|------|
| 1 | 修复前端 API 端口 bug (`8001` → `8000`)，设置 API Key 环境变量 | ✅ |
| 2-3 | 用 `points3D.ply` 测试 `/analyze` 端点，确认 PointLLM encoder 加载 | ✅ |
| 3-4 | 验证 ULIP-2 checkpoint 加载是否成功，确认是 pretrained 还是 random fallback | ✅ |
| 5 | 用 curl 测试 `/report` 端点 | ✅ |
| 6-7 | 修复所有链路报错 | ✅ |

**关键验证命令**:
```bash
set ZHIPUAI_API_KEY=32871b74afe147af83edfe74281edaaf.EyDpmMOAPjS85vJI
python src/api/main.py

# 另开终端测试
curl -X POST "http://localhost:8000/analyze" -F "file=@data/tree2/colmap_runbasicsfm/sparse/0/points3D.ply" -F "question=这棵树有多高"
```

---

### 第2周：林业 Prompt 工程

**目标**: 让 GLM 输出可靠的林业参数估计值

**改进方向**:
- 增强 system prompt，精确引导 LLM 从几何统计量推断参数
- 树高（Z轴范围）→ 直接读数，高置信度
- 胸径（DBH）→ 从点密度分布推算，中置信度
- 冠幅（X/Y扩散范围）→ 从水平投影估算，中置信度
- 碳储量 → 生物量公式推算，低置信度

**测试要求**: 准备 2-3 个不同点云样本，验证 prompt 泛化性

---

### 第3周：多树分割能力

**目标**: 从单棵树扩展到场景级多树检测

**方案**: 基于 DBSCAN 的 XY 坐标聚类（推荐，2周可完成）

```python
from sklearn.cluster import DBSCAN

def _cluster_trees(points: np.ndarray) -> List[np.ndarray]:
    """基于 XY 坐标聚类，分割不同树木"""
    clustering = DBSCAN(eps=0.5, min_samples=50).fit(points[:, :2])
    labels = clustering.labels_
    return [points[labels == i] for i in set(labels) if i >= 0]
```

**更新 `/analyze` 返回格式**: 支持多棵树列表

---

### 第4周：可视化增强

**目标**: 前端展示树木分割结果和参数对比

- [ ] 完善多树列表：点击树木卡片高亮对应 3D 点云
- [ ] 每棵树显示：树高、胸径、冠幅、碳储量
- [ ] Ground Truth 对比面板（可选，加分项）
- [ ] 参数雷达图多棵树对比

---

### 第5周：报告生成完善

**目标**: 生成可提交的调查报告文档

**报告结构**:
```
# 某路段树木调查报告
- 调查日期、总树木数
- 统计摘要（平均树高、平均胸径、总碳储量、林分蓄积量）
- 树木详情表（编号/树高/胸径/冠幅/碳储量/健康状态）
- 技术说明（PointLLM 技术路线）
```

---

### 第6周：性能优化

**目标**: 处理速度 < 10秒/文件

- [ ] 批量推理缓存（避免重复初始化 encoder）
- [ ] GPU 加速（确认使用 CUDA 而非 CPU）
- [ ] 前端实时进度反馈
- [ ] 大文件处理稳定性测试（50万+点）

---

### 第7周：测试与演示准备

**目标**: 准备 2 个演示场景 + 完整测试用例

**演示场景 A**: 单文件快速分析（3分钟）
```
上传 points3D.ply → 分析 → 展示3D可视化 → 聊天问答 → GLM回答
```

**演示场景 B**: 多树场景分析（5分钟）
```
上传多树MLS点云 → 自动分割 → 参数列表 → 生成调查报告
```

**测试用例集**:
| 文件 | 预期结果 |
|------|---------|
| `points3D.ply` | 单棵树，约15m高 |
| 人工合成矮树 | 验证高度估算准确性 |
| 密林场景 | 多树分割 |

---

### 第8周：文档与比赛提交

**目标**: 完整技术文档 + 演示材料

- [ ] 技术报告（15-20页）
  - 背景：林业点云智能分析需求
  - 方法：PointLLM 技术路线
  - 实现：工程细节
  - 实验：定量评测结果
  - 讨论：局限性与未来工作

- [ ] 演示 PPT（10页）
  - 封面、技术方案、系统演示截图、实验结果、总结

- [ ] 代码仓库整理
  - README.md、requirements.txt、示例数据

---

## 技术风险与应对

| 风险 | 可能性 | 应对 |
|------|--------|------|
| ULIP-2 checkpoint 加载失败 | 中 | 已有 random fallback；确认 checkpoint 文件是否存在 |
| LLM 参数估计精度差 | 高 | 用"估算"而非"测量"，重点展示语义理解能力 |
| 多树分割效果差 | 高 | 优先调试单树场景，多树作为加分项 |
| GLM API 超时/限流 | 低 | 添加超时重试逻辑，缓存结果 |
| 大点云文件内存爆炸 | 中 | 限制处理点数上限，显示友好错误 |

---

## 日常开发习惯

```bash
# 每天早上
git pull
python src/api/main.py  # 启动后端
# 测试昨日改动是否破坏已有功能

# 每天结束前
git add .
git commit -m "Day X: 完成YYY，修复ZZZ"
```

---

## 文件结构

```
H:\1reserch\03TREES-LLM\
├── web/index.html              # Three.js 前端
├── src/
│   ├── models/
│   │   ├── pretrained_encoder.py   # ULIP-2 PointBERT 编码器
│   │   ├── tokenizer.py             # VQ-VAE 量化器
│   │   └── point_llm.py             # 主模型封装
│   ├── api/
│   │   ├── main.py             # FastAPI 入口
│   │   ├── service.py          # 唯一业务服务
│   │   └── schemas.py          # Pydantic 模型
│   └── scripts/
│       └── inference.py        # 推理脚本
├── planning/                   # 本目录 — 项目规划
├── literature/                # 论文文献
└── resources/prompts/         # Prompt 模板
```

---

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查，版本信息 |
| `/health` | GET | 服务健康状态 |
| `/web` | GET | serve Web 前端 |
| `/analyze` | POST | 核心端点：点云+问题 → 分析结果 |
| `/report` | POST | 树木参数 → 调查报告 |