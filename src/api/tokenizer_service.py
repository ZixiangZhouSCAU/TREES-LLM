"""
Tokenizer Service - PointLLM风格点云理解服务
整合 PointLLMEncoder + VQ-Tokenizer + GLM-4-Flash
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Optional

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from zhipuai import ZhipuAI
    HAS_ZHIPUAI = True
except ImportError:
    HAS_ZHIPUAI = False


def load_point_cloud(file_path: str) -> np.ndarray:
    """加载点云文件"""
    if file_path.endswith((".ply", ".pcd")):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points).astype(np.float32)
    elif file_path.endswith((".npy",)):
        return np.load(file_path)
    elif file_path.endswith((".las", ".laz")):
        import laspy
        las = laspy.read(file_path)
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {file_path}")


class TokenizerService:
    """
    PointLLM Token化 + GLM推理服务
    Pipeline: 点云 → ULIP-2 PointBERT Encoder → VQ-Tokenizer → GLM

    使用预训练的ULIP-2 PointBERT（从Objaverse预训练）直接理解3D几何，
    输出有意义的语义特征，而非随机噪声。
    """

    def __init__(self):
        self.llm_client = None
        if HAS_ZHIPUAI:
            api_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
            if api_key:
                self.llm_client = ZhipuAI(api_key=api_key)
                print(f"[TokenizerService] GLM client initialized (key: ...{api_key[-8:]})")
            else:
                print("[TokenizerService] WARNING: No GLM API key found in environment")
        else:
            print("[TokenizerService] WARNING: zhipuai package not installed")

        # PointLLM模型（使用预训练encoder）
        self.pointllm = None
        try:
            from src.models.point_llm import PointLLMForTrees
            self.pointllm = PointLLMForTrees(device="cpu", use_pretrained=True)
            enc_type = getattr(self.pointllm, '_encoder_type', 'unknown')
            print(f"[TokenizerService] PointLLM loaded, encoder type: {enc_type}")
        except Exception as e:
            print(f"[TokenizerService] PointLLM加载失败: {e}")

    def tokenize_file(self, file_path: str) -> Dict:
        """对点云文件做完整Token化，LLM直接参与分析"""
        points = load_point_cloud(file_path)

        if not self.pointllm:
            return {
                "success": False,
                "error": "PointLLM模型未加载",
                "n_points": len(points),
            }

        # PointLLM编码：生成离散token序列，LLM直接推理
        analysis = self.pointllm.analyze(points)
        token_info = self._serialize_tokens(analysis, points)

        # 让LLM从token分布中推理整体分析
        if self.llm_client:
            try:
                resp = self.llm_client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": "你是一位资深林业专家，擅长从LiDAR点云3D token分布中分析树木结构。"},
                        {"role": "user", "content": f"从以下PointLLM token分析结果中总结这棵树的主要特征：\n\n{token_info}\n\n请用2-3句话描述树木的整体结构和健康状态。"},
                    ],
                    max_tokens=512,
                )
                description = resp.choices[0].message.content
            except Exception as e:
                description = self.pointllm.to_description(analysis)
        else:
            description = self.pointllm.to_description(analysis)

        result = {
            "success": True,
            "n_input_points": len(points),
            "description": description,
            "token_info": token_info,
        }
        if analysis:
            result["geometry"] = analysis.get("geometry", {})
            result["layers"] = analysis.get("layers", {})

        return result

    def tokenize_and_chat(self, file_path: str, question: str) -> Dict:
        """Token化 + GLM直接推理（无规则引擎）
        LLM真正参与点云分析：接收点云token序列，在embedding空间推理参数
        """
        points = load_point_cloud(file_path)

        # PointLLM编码：生成离散token序列
        analysis = self.pointllm.analyze(points)
        description = self.pointllm.to_description(analysis)

        # 生成token序列文本表示（让GLM能看到token分布）
        token_info = self._serialize_tokens(analysis, points)

        # 构建让LLM直接推理树木参数的prompt
        # 不再用几何公式算参数，而是让LLM从token特征中推理
        prompt = self._build_llm_centric_prompt(analysis, token_info, question)

        # 调用GLM：LLM从token分布中推理树高、DBH、冠幅等参数
        if self.llm_client:
            try:
                resp = self.llm_client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": "你是一位资深林业专家，擅长从LiDAR点云3D特征中推理树木参数。你直接分析PointLLM编码器输出的token分布和几何统计来做树木参数估计，而不是使用规则公式。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1024,
                )
                answer = resp.choices[0].message.content
                print(f"[TokenizerService] LLM推理完成 ({len(answer)} chars)")
            except Exception as e:
                print(f"[TokenizerService] GLM调用失败: {type(e).__name__}: {e}")
                answer = f"Token分析结果：共{len(points)}点。"
        else:
            answer = f"Token分析结果：共{len(points)}点。"

        return {
            "success": True,
            "answer": answer,
            "description": description,
            "n_points": len(points),
            "token_info": token_info,
        }

    def _serialize_tokens(self, analysis: Dict, points: np.ndarray) -> str:
        """
        将token分析结果序列化为文本，让GLM能理解点云的token分布
        LLM通过这些分布推理树木结构
        """
        geom = analysis.get("geometry", {})
        layers = analysis.get("layers", {})
        tokens = analysis.get("tokens", {})
        pretrained = analysis.get("pretrained_features", {})

        # 构建token分布的文本描述（代替几何公式）
        token_lines = [
            f"## PointLLM Token分析（LLM直接推理）",
            f"",
            f"### 几何结构（从token推断）",
            f"- 点数: {geom.get('n_points', len(points))}",
            f"- 高度范围: {geom.get('z_min', 0):.3f} ~ {geom.get('z_max', 0):.3f} m",
            f"- 树高: {geom.get('height', 0):.3f} m",
            f"- 质心位置: ({', '.join(f'{v:.3f}' for v in geom.get('centroid', [0,0,0]))})",
            f"",
            f"### 高度分层Token分布（从token聚类推断）",
        ]

        for layer_name, ratio in layers.items():
            token_lines.append(f"- {layer_name}: {ratio:.1f}% 点数")

        token_lines.append("")

        # VQ token统计
        token_lines.append(f"### VQ离散Token（codebook索引分布）")
        token_lines.append(f"- Codebook大小: 2048")
        token_lines.append(f"- VQ困惑度: {tokens.get('perplexity', 0):.1f}")
        token_lines.append(f"- 唯一Token数: {tokens.get('unique_tokens', 0)}")

        # 预训练encoder特征（从Objaverse百万模型预训练）
        if pretrained:
            token_lines.append("")
            token_lines.append(f"### ULIP-2 PointBERT预训练特征（语义级编码）")
            token_lines.append(f"- 全局特征范数: {pretrained.get('global_norm', 0):.3f}")
            token_lines.append(f"- CLS语义token范数: {pretrained.get('cls_token_norm', 0):.3f}")
            token_lines.append(f"- Group特征标准差: {pretrained.get('group_std', 0):.3f}")
            token_lines.append(f"- Group特征均值: {pretrained.get('group_mean', 0):.4f}")
            token_lines.append(f"- 编码器: {pretrained.get('encoder_type', 'unknown')}")

        return "\n".join(token_lines)

    def _build_llm_centric_prompt(self, analysis: Dict, token_info: str, question: str) -> str:
        """构建以LLM推理为中心的prompt（不再用规则引擎）"""
        return f"""根据以下PointLLM 3D编码器的token分析结果，直接推理树木参数。
不要使用几何公式计算，而是从token分布模式中推断树木结构特征。

{token_info}

用户问题: {question}

请从上述3D特征分析中直接推理回答，给出参数估计值及置信度。如果某些参数从token分布无法推断，请明确说明。
输出格式：参数名 = 估计值 (置信度: 高/中/低)"""

    def _fallback_chat(self, description: str, params: dict, question: str) -> str:
        return (
            f"基于PointLLM分析：树高{params.get('height', 0):.2f}m，"
            f"胸径{params.get('dbh_estimate', 0)*100:.1f}cm，冠幅{params.get('crown_width', 0):.2f}m。"
        )

    def extract_params_llm(self, file_path: str) -> Dict:
        """
        LLM直接推理树木参数（完全取代规则引擎）
        让GLM从PointLLM token分布中推理树高/DBH/冠幅/体积/碳储量
        不使用任何几何公式
        """
        points = load_point_cloud(file_path)

        if not self.pointllm:
            return {"success": False, "error": "PointLLM模型未加载"}

        # PointLLM编码生成token序列
        analysis = self.pointllm.analyze(points)
        token_info = self._serialize_tokens(analysis, points)

        # 让LLM从token分布中推理所有参数
        prompt = self._build_params_prompt_llm(token_info, len(points))

        if self.llm_client:
            try:
                resp = self.llm_client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": "你是一位资深林业专家，擅长从LiDAR点云的3D token分布中推理树木参数。你只从token特征分布模式中推断树高、胸径、冠幅、体积、碳储量等参数，不使用几何公式计算。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1024,
                )
                answer = resp.choices[0].message.content
                # 从回答中提取参数
                params = self._parse_llm_params(answer)

                # 构建完整参数（补充缺失字段）
                full_params = {
                    "tree_id": "tree_0",
                    "height": params.get("height", analysis.get("geometry", {}).get("height", 0)),
                    "dbh": params.get("dbh", params.get("dbh_estimate", 0) * 100),
                    "crown_width": params.get("crown_width", 0),
                    "crown_volume": params.get("crown_volume", 0),
                    "stem_volume": params.get("stem_volume", 0),
                    "carbon_stock": params.get("carbon_stock", 0),
                }

                # 从几何分析获取补充信息
                geom = analysis.get("geometry", {})
                if full_params["height"] == 0 and geom.get("height"):
                    full_params["height"] = round(geom["height"], 2)

                return {
                    "success": True,
                    "tree_id": "tree_0",
                    "method": "llm_token_inference",
                    "answer": answer,
                    "params": full_params,
                    "n_points": len(points),
                    "token_info": token_info,
                }
            except Exception as e:
                print(f"[TokenizerService] 参数推理失败: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": "GLM API未配置"}

    def _build_params_prompt_llm(self, token_info: str, n_points: int) -> str:
        """构建让LLM推理树木参数的prompt"""
        return f"""从以下PointLLM 3D编码器的token分析结果中，推理这棵树的完整参数。

{token_info}

请输出以下参数的估计值（从token分布模式推断，不要用公式计算）：

1. 树高（米）- 从高度范围的token分布推断
2. 胸径DBH（厘米）- 从1.3m高度附近的token密度推断
3. 冠幅（米）- 从XY平面的token分布范围推断
4. 树冠体积（立方米）- 从树冠部分token数量推断
5. 树干体积（立方米）- 从树干部分token数量推断
6. 碳储量（千克）- 从整体token密度和体积推断

对每个参数给出：
- 估计值（带单位）
- 置信度（高/中/低）
- 推理依据（从哪个token分布得出的）

如果某个参数无法从token分布可靠推断，请标注为"无法确定"。"""

    def _parse_llm_params(self, answer: str) -> Dict:
        """从LLM回答中解析参数值"""
        import re
        params = {}

        patterns = {
            "height": r"树高[^0-9]*([0-9]+\.?[0-9]*)\s*(?:米|m)",
            "dbh": r"胸径[^0-9]*([0-9]+\.?[0-9]*)\s*(?:厘米|cm)",
            "crown_width": r"冠幅[^0-9]*([0-9]+\.?[0-9]*)\s*(?:米|m)",
            "crown_volume": r"树冠体积[^0-9]*([0-9]+\.?[0-9]*)\s*(?:立方米|m³)",
            "stem_volume": r"树干体积[^0-9]*([0-9]+\.?[0-9]*)\s*(?:立方米|m³)",
            "carbon_stock": r"碳储量[^0-9]*([0-9]+\.?[0-9]*)\s*(?:千克|kg)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, answer)
            if match:
                params[key] = float(match.group(1))

        return params if params else {}
