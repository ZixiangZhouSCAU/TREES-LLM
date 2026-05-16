# TREES-LLM 研究规划

> 基于PointLLM (ECCV 2024)路线的Web端树木点云LLM

---

## 核心目标

构建一个**Web端树木点云智能分析平台**：
- 用户上传点云文件 → LLM直接理解3D场景
- 自动分割树木，提取参数（DBH、树高、冠幅、体积）
- 自然语言问答 + 生成分析报告

---

## 技术路线

**PointLLM风格**（端到端点云→LLM，无训练）：

```
上传点云(.ply/.las/.npy)
    │
PointNet++ Encoder   ← 直接处理原始3D坐标
    │                 层次化特征：局部几何→中层结构→全局面貌
    │
VQ-VAE Tokenizer     ← 离散化为token序列
    │                 8192码本，EMA更新
    │
LLM Projector        ← 256 queries聚合 → LLM语义空间
    │
GLM-4-Flash          ← 接收token描述 → 生成回答
    │
Web界面              ← Three.js可视化 + 交互式问答
```

**与旧架构的本质区别：**

| | 旧架构（LiDAR+UAV深度学习） | 新架构（PointLLM） |
|---|---|---|
| 点云处理 | 先分割再提取参数 | 直接理解整图，LLM自动推断 |
| 参数来源 | 规则算法（PCA/圆柱拟合）| LLM从token序列生成 |
| 树木识别 | DBSCAN聚类（需调参）| LLM根据几何特征理解 |
| 输出形式 | 固定参数表格 | 自然语言问答+参数 |
| 技术栈 | PointNet++ / Open3D | PointLLM + GLM |

---

## 核心模型

### `src/models/point_encoder.py` — TreePointEncoder

PointNet++层次化编码器，4个SA层 + 特征传播：

```
输入: (B, N, 3) 点云
  ├── SA1: 1024点, r=0.1, 32邻域 → 128-dim
  ├── SA2: 256点, r=0.2, 64邻域 → 256-dim
  ├── SA3: 64点, r=0.4, 64邻域 → 512-dim
  └── SA4: group_all → 1024-dim全局特征

跳跃连接特征传播:
  fp3: 256 → 256-dim
  fp2: 128 → 128-dim
  upsample_to_original: → (B, N, 512)
```

### `src/models/tokenizer.py` — TreePointTokenizer

VQ-VAE离散化编码器：

```
输入: (B, N, 512) encoder特征
  ├── Linear: 512 → 256 → 128
  ├── VectorQuantizer: 2048码本, EMA更新
  │   返回: quantized, token_indices, vq_loss
  └── decoder_proj: 128 → 256 → 512 (重建)

损耗: VQ commitment loss (β=0.25)
```

### `src/models/llm_projector.py` — LLMProjector

Q-Former风格投影层：

```
输入: (B, N, 128) quantized tokens
  ├── 256个learned query tokens
  ├── Multi-head cross attention → (B, 256, 128)
  ├── Linear: 128 → 4096 (GLM维度)
  └── 输出: (B, 256, 4096) LLM-ready embeddings
```

### `src/models/point_llm.py` — PointLLMForTrees

顶层封装，提供完整Pipeline：

```python
model = PointLLMForTrees(device="cuda")
analysis = model.analyze(points)    # → geometry + tokens + layers
prompt = model.build_prompt(analysis, question)  # → GLM prompt
```

---

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `GET /` | 健康检查 | 返回版本信息 |
| `POST /extract` | 参数提取 | 上传点云 → 树木参数 |
| `POST /pointllm` | PointLLM分析 | 上传点云 → 编码 → Token化 → GLM回答 |
| `POST /chat` | 直接问答 | 参数+问题 → GLM |
| `POST /report` | 报告生成 | 多棵树 → 调查报告 |

---

## 当前进度

### Phase 1: 骨架搭建 ✅
- [x] 删除旧研究方向的代码
- [x] 建立新目录结构
- [x] 实现PointNet++编码器
- [x] 实现VQ-VAE Tokenizer
- [x] 实现LLM Projector
- [x] 实现PointLLM主模型
- [x] 集成到FastAPI服务

### Phase 2: 完善Pipeline
- [ ] 端到端测试（points3D.ply）
- [ ] Web界面更新（集成/pointllm端点）
- [ ] Token可视化
- [ ] 多棵树场景支持

### Phase 3: 训练（可选）
- [ ] PointLLM微调（如果云端API效果不够好）
- [ ] Tokenizer预训练（MTPM任务）
- [ ] LLM LoRA微调

---

## 验证清单

- [ ] 老文件删除完成
- [ ] 新架构搭建完成
- [ ] PointLLM编码器可运行
- [ ] GLM能接收token描述
- [ ] Web界面正常显示
- [ ] 完整pipeline测试通过

---

## 参考资料

- PointLLM (ECCV 2024): ByteDance/OpenGVLab, 直接处理原始3D点云的LLM
- VQ-VAE: Vector Quantized Variational Autoencoder
- PointNet++: 层次化点云处理网络
- GLM-4-Flash: 智谱AI，零训练成本LLM

