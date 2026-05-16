# 研究方向总览

> 最后更新：2026-05-12
> 当前阶段：前期调研与方案设计阶段
> 课题名称：TREES-LLM (Tree Understanding and Reasoning Enhanced System)

---

## 一、研究主题

**TREES-LLM：基于树木领域大语言模型的城市街道树木参数理解与推理**

**所属领域：** 测绘遥感 / 点云处理 / 林业信息化 / 多模态大语言模型

**核心问题：** 利用 MLS 点云数据，实现城市街道树木的自动检测、分割与参数提取（树高、胸径、冠幅、体积等），并通过大语言模型实现参数理解、空间推理与自然语言报告生成

**应用背景：** 城市林业资源调查、碳储量估算、城市绿化管理、智能问答与报告生成

---

## 二、核心研究方向

### P1（已整合）：LiDAR-UAV 特征层融合

**技术路线：** Cross-Attention 特征融合 + SFM 树冠补全

**核心创新点：**
1. **特征层交互融合**：LiDAR 点云特征与 UAV 图像特征通过 Cross-Attention 机制交互
2. **UAV 引导树干补全**：利用 UAV 图像信息辅助修复 LiDAR 遮挡区域的树干点云
3. **SFM 树冠重建**：通过 UAV 影像运动恢复结构（SfM）重建完整树冠几何

**参考方案：** `planning/plan_a_lidar_uav_fusion.md`

**状态：** 已整合到方案C，保留作为对比基线

---

### P2（已整合）：LLM 驱动的树木参数报告生成

**技术路线：** Bayesian 不确定性量化 + LLM 自然语言输出

**核心创新点：**
1. **不确定性感知**：量化树木参数估计的不确定性，以置信区间形式输出
2. **自然语言报告**：将参数估计结果和不确定性分析生成可读的文本报告
3. **交互式问答**：支持用户针对估计结果进行追问和解释

**参考方案：** `planning/plan_b_llm_bayesian.md`

**状态：** 已整合到方案C，作为报告生成子模块

---

### P3（主方向）：TREES-LLM

**技术路线：** TreeVoxel Tokenizer + 点云语义投影 + 树木领域LLM

**全称：** TREES-LLM (Tree Understanding and Reasoning Enhanced System)

**核心创新点：**
1. **TreeVoxel Tokenizer** — 树木专用3D分词器，双码本机制（几何码本G+语义码本S）
2. **TreePartAttention** — 树木部位感知语义投影（干/冠/枝分别编码）
3. **TREES-LLM** — 树木领域大语言模型，实现参数理解而非仅提取
4. **Tree Scene Graph** — 树木空间关系图构建与推理

**参考方案：** `planning/plan_c_tree3d_llm.md`

**目标参数：** 胸径（DBH）、树高、冠幅、体积、碳储量

**输出能力：** 参数提取 / 智能问答 / 报告生成 / 空间关系推理 / 生长异常检测

---

## 三、技术路线图（里程碑）

| 阶段 | 时间 | 目标 | 关键交付物 | 状态 |
|------|------|------|-----------|------|
| **Phase 0** | 第 1-5 周 | 数据采集 | 45 棵树 LiDAR + UAV 数据 | ⬜ 进行中 |
| **Phase 1** | 第 6-7 周 | 数据标注 | 点云分割 + 图像分割标注 | ⬜ 待开始 |
| **Phase 2** | 第 8-11 周 | 模型训练 | PointNet++ / TreeVoxel / TREES-LLM | ⬜ 待开始 |
| **Phase 3** | 第 12-14 周 | Pipeline 搭建 | 端到端处理系统 | ⬜ 待开始 |
| **Phase 4** | 第 15-16 周 | 精度验证 | 消融实验 + 统计检验 | ⬜ 待开始 |
| **Phase 5** | 第 17-20 周 | 论文写作 | 投稿 | ⬜ 待开始 |

---

## 四、当前最紧迫的任务

### 优先级 1：补充文献调研
- [ ] 精读 3D City-LLM (CVPR 2024) — 核心架构参考
- [ ] 精读 PointLLM (ECCV 2024) — 点云LLM对齐方法
- [ ] 精读 Point-3D LLM (Apple Research) — token结构研究
- [ ] 阅读 Tree-GPT (arXiv 2023) — 林业LLM先驱
- [ ] 阅读 QuatRoPE (arXiv 2025) — 3D空间关系编码
- [ ] 阅读 3D-LLM综述论文 — 领域全景
- [ ] 工具：参考 `resources/prompts/literature_review.txt`

### 优先级 2：建立代码框架
- [x] 创建 `src/` 子目录结构
- [ ] 编写 `src/environment.yml` 环境配置
- [ ] 搭建 PointNet++ 基线代码
- [ ] 实现 TreeVoxel Tokenizer 原型
- [ ] 搭建 TREES-LLM 推理框架

### 优先级 3：规范文献管理
- [x] 更新 `literature/literature_database.xlsx`
- [ ] 运行 `docs/zotero_sync.py` 建立索引
- [ ] 补充阅读笔记到 `docs/reading_log.md`

---

## 五、数据集信息

### 自采集数据
- **采集地点：** 华南农业大学五山路校园段
- **样本数量：** 45 棵树
- **场景类型：** 3 种不同场景
- **采集设备：** MLS（车载激光扫描）+ UAV（无人机影像）
- **采集时间：** 2026 年（待确认）

### 论文数据集参考
| 数据集 | 来源 | 树木数量 | 用途 |
|--------|------|---------|------|
| Paris-Rambuteau | 城市街道 MLS | - | 树木检测参考 |
| 论文实验数据 | 已收集论文 | 多组 | 方法对比参考 |

---

## 六、竞赛信息

- **竞赛名称：** 2026 年全国大学生测绘学科创新创业智能大赛
- **相关文件：** `competition/` 目录
  - 通知 PDF 文件
  - `competition_attachment.docx`
  - 第一次/第二次会议 PPT

---

## 七、快速导航

| 资源 | 路径 |
|------|------|
| 文献索引 | `docs/library_index.md`（运行 zotero_sync.py 后生成） |
| 阅读笔记 | `docs/reading_log.md` |
| 技术方案 | `planning/` |
| 代码库 | `src/` |
| 论文草稿 | `writing/` |
| AI 提示词模板 | `resources/prompts/` |
| Claude Code 配置 | `.claude/` |
| 文献数据库 | `literature/literature_database.xlsx` |

---

## 八、变更记录

| 日期 | 变更内容 |
|------|---------|
| 2026-04-27 | 初始化研究总览，建立项目骨架 |
| 2026-05-12 | 更新课题名称为 TREES-LLM，整合方案A/B到方案C，更新文献数据库 |
