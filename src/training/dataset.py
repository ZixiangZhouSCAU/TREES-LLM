"""
训练数据集加载器

从 JSONL 文件加载训练数据，供 TwoStageTrainer 使用

Stage 1（ObjectCaption）：树特征 → 描述文本
Stage 2（SceneAnalysis / ScenePlanning）：场景特征 → 分析/规划回答

用法：
    dataset = TreeDataset("data/training/tree_training_data.jsonl", task="object_caption")
    for item in dataset:
        print(item["input_ids"], item["labels"])
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional


class TreeDataset(Dataset):
    """
    树木训练数据集

    支持三种任务：
    - object_caption: 单树描述生成
    - scene_analysis: 场景分析
    - scene_planning: 管理规划

    输入格式（JSONL）：
        {"task": "object_caption", "tree_id": "A01", "height": 12.3, "caption": "..."}
        {"task": "scene_analysis", "plot_id": "PlotB", "question": "...", "answer": "..."}
    """

    def __init__(
        self,
        jsonl_path: str,
        task: Optional[str] = None,
        tokenizer=None,
        max_length: int = 256,
    ):
        """
        Args:
            jsonl_path: JSONL 训练文件路径
            task: 筛选特定任务（None = 加载全部）
            tokenizer: 分词器（用于 tokenize 输入输出）
            max_length: 最大 token 长度
        """
        self.jsonl_path = Path(jsonl_path)
        self.task_filter = task
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples: List[Dict] = []
        self._load()

    def _load(self):
        """从 JSONL 加载数据"""
        if not self.jsonl_path.exists():
            print(f"[TreeDataset] WARNING: {self.jsonl_path} not found, returning empty dataset")
            return

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if self.task_filter is None or sample.get("task") == self.task_filter:
                    self.samples.append(sample)

        print(f"[TreeDataset] Loaded {len(self.samples)} samples from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def tokenize_sample(self, sample: Dict) -> Dict:
        """
        将样本 tokenize 为 input_ids 和 labels
        用于训练投影层
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer must be set to tokenize samples")

        if sample["task"] == "object_caption":
            # 输入：几何参数文本
            input_text = (
                f"树高{sample.get('height', 0):.1f}米，"
                f"胸径{sample.get('dbh', 0):.1f}厘米，"
                f"冠幅{sample.get('crown_width', 0):.1f}米，"
                f"碳储量{sample.get('carbon_stock', 0):.1f}千克。"
            )
            output_text = sample.get("caption", "")

        elif sample["task"] == "scene_analysis":
            stats = sample.get("scene_stats", {})
            input_text = (
                f"样地{sample.get('plot_id', '')}，"
                f"共{sample.get('n_trees', 0)}棵树木，"
                f"平均树高{stats.get('avg_height', 0):.1f}米，"
                f"平均胸径{stats.get('avg_dbh', 0):.1f}厘米，"
                f"总碳储量{stats.get('total_carbon', 0):.1f}千克。"
            )
            output_text = sample.get("answer", "")

        elif sample["task"] == "scene_planning":
            stats = sample.get("scene_stats", {})
            input_text = (
                f"样地{sample.get('plot_id', '')}共{sample.get('n_trees', 0)}棵，"
                f"平均树高{stats.get('avg_height', 0):.1f}米。"
                f"请给出管理建议。"
            )
            output_text = sample.get("answer", "")

        else:
            input_text = sample.get("question", "")
            output_text = sample.get("answer", "")

        # Tokenize
        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        output_enc = self.tokenizer(
            output_text,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": output_enc["input_ids"].squeeze(0),
        }


class MockTokenizer:
    """
    无 tokenizer 时的简易分词器（用于测试）
    简单按字分词，映射到固定词表
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        # 简单词表：0=pad, 1=unk, 2=eos
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2

    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """简易编码：按 Unicode 码点分词"""
        tokens = [ord(c) % (self.vocab_size - 3) + 3 for c in text]
        tokens = tokens[:max_length - 1] + [self.eos_token_id]
        return tokens

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(max(0, min(i - 3, 0x10FFFF - 3) + 3)) for i in ids if i > 2)


def test_dataset():
    """测试数据集"""
    import tempfile

    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for i in range(10):
            sample = {
                "task": "object_caption",
                "tree_id": f"A{i:02d}",
                "height": 10.0 + i,
                "dbh": 30.0 + i * 2,
                "crown_width": 4.0 + i * 0.2,
                "carbon_stock": 50.0 + i * 5,
                "caption": f"这是第{i+1}棵测试树的描述。"
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        temp_path = f.name

    # 测试加载
    dataset = TreeDataset(temp_path, task="object_caption")
    print(f"Loaded {len(dataset)} samples")

    # 测试 mock tokenizer
    tokenizer = MockTokenizer()
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        tokens = tokenizer.encode(f"树高{sample['height']}米，胸径{sample['dbh']}厘米。")
        print(f"  Sample {i}: tree_id={sample['tree_id']}, tokens={len(tokens)}")

    print("[OK] TreeDataset test passed")


if __name__ == "__main__":
    test_dataset()