# vlm_vanguard
# Fine-tuning SmolVLM on the ChartLlama Dataset using LoRA/QLoRA

This repository contains code and instructions for fine-tuning the **SmolVLM** models from Hugging Face, such as `HuggingFaceTB/SmolVLM-256M-Instruct`, on the [ChartLlama dataset](https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/tree/main). The fine-tuning process leverages **Parameter-Efficient Fine-Tuning (PEFT)** methods—specifically **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)**—via the Hugging Face `transformers`, `peft`, and `trl` libraries. Since the complete ChartLlama dataset is not publicly accessible, we fine-tuned the model using the publicly available subset obtained from the above link, which includes 980 chart images paired with corresponding question–answer dialogues.

This repository contains the code and resources for a project that aims to explore and enhance the capabilities of Vision Language Models through fine-tuning techniques. The project is part of a Deep Learning course at the Georgia Institute of Technology (CS7643 in Spring 2025). 

## Overview

The primary objective of this project is to improve the SmolVLM model’s ability to interpret and answer questions about various types of charts (e.g., line charts, bar charts). Our goal was to evaluate how effectively a small model like `SmolVLM-256M-Instruct` could be fine-tuned on the ChartLlama dataset. The process includes the following steps:

1.  **Loading the ChartLlama dataset:** This dataset consists of chart images paired with question-answer conversations about the visual information presented.
2.  **Preparing the SmolVLM model:** Loading the base model e.g. `SmolVLM-256M-Instruct` and its processor.
3.  **Applying PEFT:** Configuring and applying LoRA or QLoRA adapters to the model to enable efficient fine-tuning.
4.  **Supervised Fine-tuning (SFT):** Using the `trl` library's `SFTTrainer` to fine-tune the adapted model on the ChartLlama dataset, teaching it to follow instructions and answer chart-related questions accurately and concisely.
5. **Evaluation:** Benchmarking the fine-tuned model against the base model using metrics like Exact Match (EM) and Relaxed Numerical Accuracy on all 2500 samples of the ChartQA dataset.
6.  **Analysis:** Aggregating results, calculating performance improvements, and generating plots for training/validation loss.

## Prerequisites

*   A GPU with CUDA support is highly recommended for feasible training times (NVIDIA L4, T4, A100, etc.).
*   Access to the ChartLlama dataset (JSON files and corresponding images).
*   Access to the ChartQA dataset (via Hugging Face `datasets`).

Our work was mostly done with **Georgia Tech's PACE ICE GPU cluster**, **Amazon SageMaker**, and **Google Colab**. The sample notebook `final_notebook.ipynb` is built to be run in Google colab, but you can adapt it for use in alternative environments.

## Dependencies

You can install the required packages using pip as oper the below command.

```bash
pip install -U datasets bitsandbytes transformers trl peft accelerate torch wandb word2number Pillow pandas matplotlib seaborn scikit-learn Jinja2
# Install flash-attn (optional but recommended for performance on compatible GPUs)
pip install -q flash-attn --no-build-isolation
```