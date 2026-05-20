# 研究路线：基于 MLS 点云的树木参数自动提取与碳汇计量系统

> **所属项目**：TREES-LLM（Tree Understanding and Reasoning Enhanced System）
> **参赛赛道**：全国大学生测绘学科创新创业智能大赛 —— 创新设计比赛
> **团队成员**：何健仁（队长）、周子翔、陈梓薇、王睿
> **文档日期**：2026-05-18

---

## 一、研究背景与问题定位

### 1.1 竞赛背景

本项目属于测绘学科创新创业智能大赛——创新设计比赛赛道，**移动激光雷达（MLS）+ 深度学习 + 大语言模型（LLM）** 三大测绘遥感与人工智能技术的交叉应用。

### 1.2 行业痛点与核心问题

现有研究在三个层次上存在局限：

| 层次 | 传统方法局限 | 本研究创新切入点 |
|------|------------|----------------|
| **单木分割** | 依赖人工阈值或传统聚类，泛化能力差，遮挡区域精度骤降 | PointNet++ 监督学习 + 多模态纹理特征 + 不确定性感知输出 |
| **参数提取** | 需野外人工测量 DBH/树高，效率低、破坏性大 | 几何算法自动计算 + 误差传播链不确定性量化 |
| **碳储量估算** | 异速生长公式依赖经验系数，与实测点云数据脱节 | 点云体积法与材积-密度法双轨比对校验 |

> PPT 中已明确指出："**很难做得出彩，需要更进一步的创新**，解决传统方法的局限性。本研究以 LLM 语义层作为核心创新增量，实现从点云数据到自然语言报告的端到端。

### 1.3 研究目标

开发基于 MLS 的端到端系统，实现：
- 点云预处理与单木自动分割
- DBH、树高、冠幅、材积等参数自动提取
- 碳储量估算与不确定性量化
- LLM 驱动的个性化林业专家报告

---

## 二、技术路线总览

```
数据获取 ──→ 点云预处理 ──→ 单木分割 ──→ 参数提取 ──→ 碳储量估算 ──→ LLM报告
LiDAR+      去噪/配准/      PointNet++   DBH/树高/    AGB×含碳率      自然语言
UAV融合      地面分离       监督学习      冠幅/材积     双轨验证        个性化报告
```

### 核心技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| 点云编码器 | ULIP-2 PointBERT | 768维 Objaverse 预训练权重 |
| 点云分割 | PointNet++ | 监督学习，实例如分割 |
| 量化器 | VQ-VAE | 2048 codebook，128维嵌入 |
| 语言模型 | GLM-4-Flash | 智谱AI 云端 API，零训练 |
| 后端 | FastAPI + uvicorn | 懒加载 torch，避免阻塞 |
| 前端 | Three.js | 原生 ES modules，无需构建 |
| 点云处理 | Open3D + sklearn | DBSCAN 聚类 / 几何计算 |
| 环境 | Python 3.12, PyTorch 2.6+CUDA12.4 | RTX 4060 Laptop 8GB |

---

## 三、分阶段研究计划

### 阶段 1：数据获取与预处理

**时间**：5月19日 — 5月31日
**负责人**：何健仁 + 王睿

#### 设备清单
- 手持 LiDAR 扫描仪（高精度单木点云采集）
- 带 RTK 模块的 DJI 无人机（厘米级定位 + 影像采集）
- RTK 地面基准站
- 4070/5060 显卡（本机开发）

#### 数据采集方案

> 详细计划见 [DATA_COLLECTION_PLAN.md](./DATA_COLLECTION_PLAN.md)

**采集策略**：小规模验证，50–60棵单木，3块样地，2–3天完成

| 样地 | 场景 | 树种 | 数量 | 特点 |
|------|------|------|------|------|
| 样地 A | 孤树 | 玉兰 | **已有** | 无需重复采集，直接使用现有数据 |
| 样地 B | 道路+树木 | 行道树 | 15–20 棵 | MLS 移动扫描，街道场景，行道树间距均匀 |
| 样地 C | 树林 | 香樟/桉树 | 20–30 棵 | 密林网格采样，密度高，遮挡严重 |

**采集流程**：

```
无人机宏观扫描 → 手持 LiDAR 精细扫描 + 人工 Ground Truth 测量（并行）
    → 点云-影像配准（室内）→ 数据整理与命名
```

**Ground Truth 测量项**（每棵树）：
- DBH（胸径尺，0.1cm 精度）
- 树高（激光测距仪，0.1m 精度）
- 冠幅东西/南北（卷尺，0.1m 精度）

---

### 阶段 2：单木实例分割（PointNet++ 监督学习）

**时间**：5月20日 — 5月31日
**负责人**：周子翔 + 陈梓薇（消融实验）

#### 训练数据：TreeLearn

**来源**：[TreeLearn: A deep learning method for segmenting individual trees from forest point clouds](https://arxiv.org/abs/2309.08471)（Knowledge-Based Systems, 2024）
**GitHub**：`Weizheng-NY/TreeLearn`

TreeLearn 提供了已标注的森林点云数据集，每点标注了树干/树冠/地面的语义标签及单木实例 ID，可直接用于 PointNet++ 监督训练。

**数据集特点**：
- 森林场景 TLS/MLS 点云，包含多棵树的林分场景
- 逐点标注：语义标签（树干/枝叶/地面）+ 实例 ID（树木编号）
- 格式：PLY / LAS，兼容 Open3D / PyTorch 数据加载

**使用策略**：
1. 下载 TreeLearn 数据集，解析为 `(N, 3+label)` 格式
2. 按 8:2 划分训练/验证集
3. 用 TreeLearn 数据训练 PointNet++ baseline（Exp-A）
4. 在实测 MLS 数据上微调，验证泛化能力

#### 主干模型：PointNet++

```
输入：林分点云
        ↓
PointNet++（层次化集合抽象）
        ↓
输出：每点语义标签（树干/树冠/地面）+ 实例标签（树木编号）
```

#### 核心创新点

**创新点 A：多模态输入**
- 将 RGB/NIR 纹理作为额外特征维度（6维输入：xyz + RGB）
- 提升遮挡区域和林分密集处的分割精度

**创新点 B：不确定性感知输出**
- 在语义头之外增加 uncertainty head
- 量化分割边界的不确定性
- 作为后续参数提取的可信度权重

#### 消融实验设计

| 实验组 | 描述 | 目的 |
|--------|------|------|
| Exp-A | PointNet++ baseline（仅 xyz） | 基准性能 |
| Exp-B | PointNet++ + 纹理特征（xyz + RGB） | 验证多模态增益 |
| Exp-C | PointNet++ + uncertainty head | 验证不确定性量化价值 |

**评价指标**：mIoU（所有类别分割精度的平均值）

> 项目 `src/models/point_encoder.py` 已集成 PointBERT/PointNet++ 两种编码器路径，可直接扩展为分割模型。

---

### 阶段 3：树木参数自动提取

**负责人**：周子翔

基于阶段2输出的带标签单木点云，通过几何算法计算四类参数：

#### 参数计算方法

| 参数 | 计算方法 | 技术细节 |
|------|---------|---------|
| **胸径 DBH** | 1.3m 高度处最小二乘圆拟合 | 在 1.2m–1.4m 范围内搜索最大圆半径 |
| **树高 TH** | Z 轴极值（max_z − 地面高程） | 地面高程由地面点拟合平面确定 |
| **冠幅** | XY 平面投影外包框面积 | 多方向投影取最大值（东西×南北） |
| **材积 V** | 分段圆台求和 / 体素化 | 0.2m 分段高度，Open3D 凸包体积 |

#### 不确定性量化

每个参数输出 **均值 ± 标准差**，误差传播链：

```
点云配准误差（RTK精度 ~2cm）
        ↓
分割边界不确定性（uncertainty head 输出）
        ↓
参数计算误差（几何算法的数值误差）
        ↓
最终参数 ± 不确定性
```

> `src/api/interpreters.py` 已实现参数解读引擎，需补充几何计算模块。

---

### 阶段 4：碳储量估算（双轨验证）

**时间**：6月1日 — 6月10日
**负责人**：王睿（算法）+ 何健仁（实验验证）

#### 方法一：材积-密度法（林业标准方法，Ground Truth）

```
V = Σ(分段圆台体积)           # 阶段3已实现
AGB = V × BD                 # BD = 基本密度（阔叶树参考值 0.45–0.55 g/cm³）
C = AGB × CF                 # CF = 含碳率 = 0.5（林业通用值）
```

#### 方法二：点云体积法

```
V_stem   = 体素化树干点云    # VoxelGrid 方法（Open3D）
V_crown  = 凸包/体素化树冠   # convex hull
AGB = V_stem × BD_stem + V_crown × BD_crown
C = AGB × 0.5
```

#### 比对分析

以材积-密度法为 Ground Truth，评估点云体积法的相对误差，验证方法二的可行性。

```
林分总碳储量：C_total = Σ C_i （逐木累加）
```

> `src/data/forest_knowledge.py` 已内置林业知识库（异速生长公式、林分类型规则），`src/api/parameter_advisor.py` 负责参数推荐。

---

### 阶段 5：LLM 语义层与软件集成

**时间**：6月10日 — 提交前
**负责人**：周子翔（核心）+ 全员辅助

#### 系统架构

```
┌──────────────────────────────────────────────────────┐
│                   GLM-4-Flash LLM                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 意图分类器 │→│ 决策推理引擎  │→│ 林业专家解读器 │  │
│  │(10种意图) │  │(报告/问答/推荐)│  │(interpreters)│  │
│  └──────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────┘
                        ↓
         自然语言回答 + 个性化碳储量报告
```

#### 系统输出功能

1. **单木参数结构表**（DBH / 树高 / 冠幅 / 材积）
2. **单木识别结果可视化**（Web 端 Three.js 点云渲染）
3. **碳储量计算报表**（标准 / 详细 / 碳汇专项三种模式）
4. **个性化林业专家对话**（用户追问，自动解读参数含义）
5. **处理后点云截图导出**

#### 核心差异化

这是本研究区别于一般点云处理论文的最大创新点：普通方法只能输出参数数值，本系统可以：
- 解释"这棵樟树的碳储量在全校属于什么水平"
- 根据用户需求输出不同详细程度的报告（简要 / 标准 / 详细 / 碳汇专项）
- 回答非专业用户的林业问题

> 项目 `src/api/main.py` 已实现 `/ask/stream`（SSE 流式）、`/report`（报告生成）、`/analyze`（端到端分析）等全部端点，Web 端 `web/index.html` 已集成 Three.js 可视化。

---

## 四、竞赛评分对应策略

| 评分维度 | 对应本研究内容 | 得分策略 |
|---------|-------------|---------|
| **创新性** | PointLLM 多模态 + LLM 语义层 + 不确定性量化 | 重点展示"点云+LLM"融合的原创性 |
| **技术深度** | PointNet++ mIoU、参数提取精度、碳储量双轨验证 | 提供量化指标（mIoU > 85%，DBH 误差 < 5cm） |
| **实用性** | 端到端 Web 系统，可直接输出碳储量报告 | 现场演示 Web 界面，展示完整端到端流程 |
| **文档质量** | 技术报告 + 用户手册 + 林业知识库说明 | 结构化撰写，图文并茂 |
| **团队协作** | 分工明确（模型/数据/算法/文档），进度可控 | 保留 git 提交记录、版本迭代证明 |

---

## 五、SCI 论文投稿规划

**目标期刊**：Journal of Forestry Research（SCI, Q2，非 OA，无需版面费）

#### 论文框架

```
Title（暂定）:
"An integrated MLS-LLM system for automatic tree parameter extraction
 and carbon stock estimation in subtropical forests"

Abstract (300词):
  问题陈述 → 方法概述 → 实验结果 → 结论贡献

1. Introduction
   - 森林碳汇计量的重要性与"双碳"目标背景
   - 传统野外测量方法的局限性
   - MLS 点云在林业调查中的应用潜力
   - 本文贡献（3-4条）

2. Related Work
   - LiDAR 单木分割方法综述（PointNet++、PointMLP 等）
   - 树木参数自动提取综述
   - LLM 在遥感/林业中的应用现状

3. Methodology
   3.1 数据采集与预处理（MLS + UAV 融合）
   3.2 单木分割（PointNet++ + 不确定性感知）
   3.3 参数提取（几何算法 + 不确定性量化）
   3.4 碳储量估算（材积-密度法 vs. 点云体积法双轨）
   3.5 LLM 语义层（意图分类 + 决策引擎 + 林业知识库）

4. Experiments
   4.1 研究区域与数据集描述
   4.2 单木分割精度（mIoU 指标）
   4.3 参数提取精度（DBH/树高/冠幅 vs. 人工测量）
   4.4 碳储量对比（双轨方法 vs. 地面样地数据）
   4.5 LLM 语义层用户评估

5. Discussion
   - 方法局限性分析
   - 与现有研究横向对比
   - 应用场景与推广价值

6. Conclusion
   - 主要成果总结
   - 未来工作方向
```

---

## 六、时间线总表

```
2026年
├── 4.19  启动会，熟悉竞赛与选题，分派任务，读文献
├── 4.26  选题讨论第一次会议（相关工作汇报）
├── 5.06  选题讨论第二次会议（确定最终选题与技术路线）
│
├── 5.10  ★ 阶段1启动
│          周子翔/陈梓薇 → PointNet++ 分割模型训练框架搭建
│          何健仁/王睿  → 数据采集方案设计与实测数据采集
│
├── 5.18  ★ 阶段3启动（阶段2并行推进）
│          几何参数提取模块开发（DBH/树高/冠幅/材积）
│
├── 5.25  ★ 阶段2收尾 + 消融实验
│          PointNet++ 模型精度测试（mIoU）
│          多模态/不确定性实验组对比
│
├── 6.01  ★ 阶段4启动
│          碳储量双轨估算（材积-密度法 + 点云体积法）
│          双轨比对与误差分析
│
├── 6.10  ★ 阶段5启动
│          LLM 层集成（意图分类 + 决策引擎 + 林业知识库）
│          Web 软件界面完善
│
├── 6.20  作品提交截止
│
└── 6月内  SCI 论文投稿 Journal of Forestry Research
```

---

## 七、项目现状与待完成工作

### 已完成（TREES-LLM 现有架构）

| 模块 | 状态 | 文件位置 |
|------|------|---------|
| PointLLM 核心（PointBERT + VQ-VAE + GLM-4-Flash） | ✅ | `src/models/point_llm.py` |
| FastAPI 全端点（/analyze, /ask/stream, /report） | ✅ | `src/api/main.py` |
| Three.js Web 可视化 | ✅ | `web/index.html` |
| 林业知识库（林分类型 + 异速生长公式） | ✅ | `src/data/forest_knowledge.py` |
| 意图分类器（10种意图） | ✅ | `src/api/intent.py` |
| 决策推理引擎 + 林业专家解读器 | ✅ | `src/api/decision_engine.py`、`src/api/interpreters.py` |
| 流式 SSE 问答（/ask/stream） | ✅ | `src/api/main.py` |
| DBSCAN 多树聚类 | ✅ | `src/api/service.py` |

### 扩展功能：LLM 自然语言操控点云

> **阶段 5 扩展**（与 LLM 语义层并行推进）
> **目标**：从"LLM 解读点云"升级为"LLM 操控点云"，用户可通过自然语言指令对点云执行过滤、裁剪、变换、筛选等操作

#### 技术方案：GLM Function Calling

GLM-4-Flash 原生支持 function calling，无需微调。将点云操作封装为可调用函数，LLM 根据用户指令解析并调用：

```python
pointcloud_functions = [
    {
        "name": "filter_by_height",
        "description": "筛选指定高度范围内的点",
        "parameters": {
            "type": "object",
            "properties": {
                "min_z": {"type": "number", "description": "最小高度(m)"},
                "max_z": {"type": "number", "description": "最大高度(m)"}
            },
            "required": ["min_z", "max_z"]
        }
    },
    {
        "name": "select_trees_by_dbh",
        "description": "按胸径筛选单木，返回符合条件的树ID列表",
        "parameters": {
            "type": "object",
            "properties": {
                "min_dbh_cm": {"type": "number"},
                "max_dbh_cm": {"type": "number"}
            }
        }
    },
    {
        "name": "select_trees_by_height",
        "description": "按树高筛选单木",
        "parameters": {
            "type": "object",
            "properties": {
                "min_height_m": {"type": "number"},
                "max_height_m": {"type": "number"}
            }
        }
    },
    {
        "name": "crop_region",
        "description": "裁剪指定空间区域内的点云",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x_min, y_min, z_min, x_max, y_max, z_max]"
                }
            }
        }
    },
    {
        "name": "transform_rotate",
        "description": "旋转点云到指定角度",
        "parameters": {
            "type": "object",
            "properties": {
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "angle_deg": {"type": "number"}
            }
        }
    },
    {
        "name": "get_point_cloud_stats",
        "description": "获取当前点云或选中区域的统计信息（点数、边界框、体积等）",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "export_points",
        "description": "导出选中的点云数据为指定格式",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["ply", "xyz", "las"]},
                "tree_ids": {"type": "array", "items": {"type": "number"}, "description": "要导出的树ID列表"}
            }
        }
    }
]
```

#### 意图分类扩展

在现有 10 种意图基础上，新增**操作类意图**：

| 扩展意图 | 示例指令 | 操作类型 |
|---------|---------|---------|
| `FILTER_HEIGHT` | "只看1.3m以上的点" | 过滤 |
| `SELECT_BY_DBH` | "显示胸径大于30cm的树" | 筛选 |
| `SELECT_BY_HEIGHT` | "找出高度超过10米的树" | 筛选 |
| `CROP_REGION` | "裁剪这块区域" | 裁剪 |
| `TRANSFORM_ROTATE` | "绕Z轴旋转90度" | 变换 |
| `GET_STATS` | "这片区域有多少个点" | 查询 |
| `EXPORT_DATA` | "导出树1和树2的数据" | 导出 |

#### 前端交互：点击+语言混合（进阶）

用户可在 Three.js 3D 视图中点击选中某棵树，然后通过自然语言发出指令：

```
用户在 3D 视图中点击了树 #3
用户说："把这棵树单独显示，并计算它的碳储量"
系统解析 → {"selected_trees": [3], "operation": "show_and_analyze"}
执行 → 过滤保留树3 → 更新可视化 → 调用 analyze 端点
```

**实现方式**：前端维护 `selectedTreeIds` 状态，随请求一同发送：

```javascript
// 前端状态
selectedTreeIds: [3]

// 请求体
{
  "operation": "show_and_analyze",
  "selected_tree_ids": [3],
  "instruction": "计算这棵树的碳储量"
}
```

#### 系统工作流

```
用户指令 → LLM 意图分类
    ↓
LLM 判断：无需 function call？ → 直接回答（如"这棵树的碳储量约48kg"）
    ↓
需要调用函数？
    ↓
解析参数 → 执行点云操作（filter/select/crop/transform）
    ↓
返回操作结果 + 自然语言解释（如"已筛选出DBH>30cm的3棵树，它们的高度分布在8-12m之间"）
    ↓
前端更新可视化（高亮操作后的点云）
```

#### 预期效果示例

| 用户指令 | LLM 操作 | 返回结果 |
|---------|---------|---------|
| "把1.3m以上的点显示出来" | `filter_by_height(min_z=1.3)` | 前端显示过滤后点云 + "已筛选出 8,234 个点，主要为树冠部分" |
| "找出胸径大于30cm的树" | `select_trees_by_dbh(min_dbh_cm=30)` | 前端高亮 5 棵树 + 列表展示 |
| "对比树A和树B的冠幅" | `select_trees` + 统计计算 | "树A冠幅4.2m，树B冠幅5.8m，树B比树A大38%" |
| "裁剪校园广场区域" | `crop_region(bbox=[...])` | 返回裁剪后点云 + 统计信息 |
| "导出这3棵树的数据" | `export_points(tree_ids=[1,3,5])` | 返回 PLY 文件供下载 |

#### 改动清单

| 文件 | 改动内容 |
|------|---------|
| `src/api/pointcloud_operators.py` | **新增**，封装所有点云操作函数 |
| `src/api/intent.py` | 扩展意图分类，新增操作类意图 |
| `src/api/main.py` | 新增 `POST /operate` 端点，接收自然语言指令 |
| `src/api/service.py` | 集成 pointcloud_operators |
| `web/index.html` | 新增指令输入框 + 前端状态管理（selectedTreeIds） |
| `planning/RESEARCH_ROUTE.md` | 本章节 |