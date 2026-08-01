"""
reward_vlm.py
=============
基于 Qwen3-VL-2B-Instruct 的 VLM Reward，替换原版的 GIT + ORM。

包含两个类：
  - VLMAttr  : 替换 GIT，负责属性绑定 reward（颜色/形状/材质/大小）
  - VLMOrm   : 替换 ORM，负责整体语义对齐 reward

接口与原版 reward 类完全兼容：
  __init__(self, args)
  load_to_device(self, load_device)
  __call__(self, prompts, images, **kwargs) -> List[float]
"""

import re
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor



class _Qwen3VLBase:
    """
    Qwen3-VL-2B 的加载基类，VLMAttr 和 VLMOrm 共享同一个模型实例。
    """

    # 类级别共享：同一进程里只加载一次模型
    _shared_model = None
    _shared_processor = None
    _shared_model_path = None

    def __init__(self, args):
        self.ckpt_path = args.vlm_ckpt_path

    def load_to_device(self, load_device):
        self.device = load_device

        if (_Qwen3VLBase._shared_model is not None and
                _Qwen3VLBase._shared_model_path == self.ckpt_path):
            self.model = _Qwen3VLBase._shared_model
            self.processor = _Qwen3VLBase._shared_processor
            print(f"[VLM Reward] Reusing shared Qwen3-VL-2B instance.")
            return

        print(f"[VLM Reward] Loading Qwen3-VL-2B (bf16) from: {self.ckpt_path}")
        processor = AutoProcessor.from_pretrained(self.ckpt_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.ckpt_path,
            dtype=torch.bfloat16,
        ).eval().to(load_device)

        for p in model.parameters():
            p.requires_grad = False

        # 存到类变量，供另一个实例复用
        _Qwen3VLBase._shared_model = model
        _Qwen3VLBase._shared_processor = processor
        _Qwen3VLBase._shared_model_path = self.ckpt_path

        self.model = model
        self.processor = processor
        print(f"[VLM Reward] Loaded (bf16). Memory: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def _vqa(self, image: Image.Image, question: str,
             max_new_tokens: int = 16) -> str:
        """
        单张图像的 VQA 推理，返回干净的文字答案。
        使用 /no_think 关闭 thinking 模式，保证输出简洁。
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",
                     "text": question + " /no_think"},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
        response = self.processor.decode(
            trimmed, skip_special_tokens=True
        ).strip().lower()

        return response


class VLMAttr(_Qwen3VLBase):
    """
    属性绑定 Reward，替换原版 GIT。

    原版 GIT 的逻辑：
      - 跳过 spatial 和 numeracy 任务
      - 对 attr_nouns（如 "red cube"）构造 VQA 问题，用 yes/no logits 打分

    本类的改进：
      - 使用更强的 Qwen3-VL-2B 做 open-ended VQA
      - 分别询问颜色/形状/材质，做模糊匹配
      - 保持对 spatial/numeracy 任务返回 0 的兼容行为
    """

    @property
    def __name__(self):
        return 'VLMAttr'

    def __call__(self, prompts: List[str], images: List[Image.Image],
                 **kwargs) -> List[float]:
        """
        Args:
            prompts    : List[str]，原始文字 prompt
            images     : List[PIL.Image]，生成的图像
            kwargs     :
              task_type  (List[str])            : 任务类型
              attr_nouns (List[Optional[List]]) : 属性名词对，如 [["red cube"], ...]
              nouns      (List[Optional[List]]) : 纯名词列表

        Returns:
            List[float]，每张图的属性匹配分数 [0.0, 1.0]
        """
        scores = []

        for i, (prompt, image) in enumerate(zip(prompts, images)):

            task_type  = kwargs.get('task_type',  [None] * len(prompts))[i]
            attr_nouns = kwargs.get('attr_nouns', [None] * len(prompts))[i]
            nouns      = kwargs.get('nouns',      [None] * len(prompts))[i]

            # ── 与原版 GIT 保持一致：跳过 spatial 和 numeracy ──────────────
            if task_type in ['spatial', 'numeracy']:
                scores.append(0.0)
                continue

            # ── 确定要检查的词列表 ─────────────────────────────────────────
            if attr_nouns is not None and len(attr_nouns) > 0:
                check_list = attr_nouns      # 优先用 attr_nouns
            elif nouns is not None and len(nouns) > 0:
                check_list = nouns
            else:
                scores.append(1.0)           # 无名词可检查，与原版一致给 1
                continue

            # ── 对每个 attr_noun 做属性 VQA ───────────────────────────────
            item_scores = []
            for attr_noun in check_list:
                attr_noun_str = attr_noun if isinstance(attr_noun, str) \
                                else " ".join(attr_noun)

                # 解析属性词和名词
                # attr_noun 通常形如 "red cube" / "large blue sphere"
                words = attr_noun_str.lower().split()
                obj   = words[-1]            # 最后一个词当名词
                attrs = words[:-1]           # 前面的词当属性

                if not attrs:
                    # 没有属性词，只检查物体是否存在
                    q = (f"Is there a {obj} in this image? "
                         f"Reply with yes or no only.")
                    ans = self._vqa(image, q, max_new_tokens=4)
                    item_scores.append(1.0 if "yes" in ans else 0.0)
                    continue

                # 对每个属性词做独立检查
                for attr_word in attrs:
                    attr_type = self._guess_attr_type(attr_word)
                    q = (f"What is the {attr_type} of the {obj} "
                         f"in this image? "
                         f"Reply with exactly one word, nothing else.")
                    ans = self._vqa(image, q, max_new_tokens=8)

                    # 模糊匹配：期望词出现在答案里，或答案出现在期望词里
                    match = (attr_word in ans) or (ans in attr_word)
                    item_scores.append(1.0 if match else 0.0)

            scores.append(float(np.mean(item_scores)) if item_scores else 0.0)

        return scores

    @staticmethod
    def _guess_attr_type(attr_word: str) -> str:
        """根据属性词推断属性类型，用于构造更自然的问题。"""
        COLOR_WORDS = {
            "red","blue","green","yellow","orange","purple","pink",
            "black","white","gray","grey","brown","cyan","magenta",
            "gold","silver","beige","teal","violet","indigo",
        }
        SIZE_WORDS  = {"large","small","big","tiny","huge","giant","mini","tall","short"}
        SHAPE_WORDS = {"round","square","circular","rectangular","triangular",
                       "oval","cubic","spherical","cylindrical","flat"}

        w = attr_word.lower()
        if w in COLOR_WORDS:  return "color"
        if w in SIZE_WORDS:   return "size"
        if w in SHAPE_WORDS:  return "shape"
        return "appearance"   # fallback


# ── VLMOrm：替换 ORM ──────────────────────────────────────────────────────────

class VLMOrm(_Qwen3VLBase):
    """
    整体语义对齐 Reward，替换原版 ORM（LLaVA-7B）。

    原版 ORM 的逻辑：
      - 构造 "Does this image accurately represent the prompt? yes/no"
      - 取 yes/(yes+no) 的概率作为分数

    本类的改进：
      - 用 0-10 评分替代 yes/no，梯度信号更细腻
      - 对 spatial/numeracy 任务使用专门的问题模板，更有针对性
    """

    @property
    def __name__(self):
        return 'VLMOrm'

    # 不同任务类型的问题模板
    _QUESTION_TEMPLATES = {
        'attribute': (
            "Does this image accurately depict: \"{prompt}\"? "
            "Focus on whether the object attributes (color, shape, size) are correct. "
            "Rate from 0 to 10. Reply with a single integer only."
        ),
        'spatial': (
            "Does this image accurately depict: \"{prompt}\"? "
            "Focus on whether the spatial relationships (left/right/above/below) are correct. "
            "Rate from 0 to 10. Reply with a single integer only."
        ),
        'numeracy': (
            "Does this image accurately depict: \"{prompt}\"? "
            "Focus on whether the number of objects is correct. "
            "Rate from 0 to 10. Reply with a single integer only."
        ),
        'non-spatial': (
            "Does this image accurately depict: \"{prompt}\"? "
            "Rate from 0 to 10. Reply with a single integer only."
        ),
        'default': (
            "Does this image accurately depict: \"{prompt}\"? "
            "Rate from 0 to 10. Reply with a single integer only."
        ),
    }

    @property
    def __name__(self):
        return 'VLMOrm'

    def __call__(self, prompts: List[str], images: List[Image.Image],
                 **kwargs) -> List[float]:
        """
        Returns:
            List[float]，每张图的整体语义对齐分数 [0.0, 1.0]
        """
        scores = []

        for i, (prompt, image) in enumerate(zip(prompts, images)):
            task_type = kwargs.get('task_type', [None] * len(prompts))[i]

            # 选择对应的问题模板
            template = self._QUESTION_TEMPLATES.get(
                task_type, self._QUESTION_TEMPLATES['default']
            )
            question = template.format(prompt=prompt)

            raw_answer = self._vqa(image, question, max_new_tokens=8)
            score = self._parse_score(raw_answer)
            scores.append(score)

        return scores

    @staticmethod
    def _parse_score(answer: str) -> float:
        """
        从模型输出中解析 0-10 的整数评分，归一化到 [0.0, 1.0]。
        解析失败时返回 0.5（中性分）。
        """
        nums = re.findall(r'\b(\d+)\b', answer)
        if nums:
            score = int(nums[0])
            score = max(0, min(10, score))   # 裁剪到合法范围
            return score / 10.0
        return 0.5   # fallback
