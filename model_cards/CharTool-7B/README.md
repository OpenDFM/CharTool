---
base_model: Qwen/Qwen2.5-VL-7B-Instruct
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - multimodal
  - chart-understanding
  - visual-reasoning
  - tool-use
---

# CharTool-7B

[Paper](https://arxiv.org/abs/2604.02794) | [Code](https://github.com/OpenDFM/CharTool) | [CharTool-3B](https://huggingface.co/OpenDFM/CharTool-3B)

CharTool-7B is a tool-integrated multimodal agent for fine-grained chart perception and accurate numerical reasoning. It can use image cropping for localized visual perception and Python code execution for numerical computation.

## 📋 Model Details

- **Base model:** [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Training:** Cold-start supervised fine-tuning followed by agentic reinforcement learning on DuoChart
- **Parameters:** 7B

## 🚀 Usage

CharTool relies on a tool-integrated inference loop and a code sandbox. Please follow the [evaluation instructions](https://github.com/OpenDFM/CharTool#evaluation) in the CharTool repository.

Use this Hugging Face repository as the model path:

```bash
python src/generate.py \
    --model_name chartool \
    --split val \
    --mode reasoning \
    --model_path OpenDFM/CharTool-7B
```

## 📝 Citation

```bibtex
@article{zhang2026chartool,
  title   = {CharTool: Tool-Integrated Visual Reasoning for Chart Understanding},
  author  = {Zhang, Situo and Zhang, Yifan and Zhu, Zichen and Ma, Da and Pan, Lei and Zhang, Danyang and Zhao, Zihan and Chen, Lu and Yu, Kai},
  journal = {arXiv preprint arXiv:2604.02794},
  year    = {2026}
}
```
