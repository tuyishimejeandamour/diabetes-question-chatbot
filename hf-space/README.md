---
title: Diabetes Medical Chatbot
emoji: 🩺
colorFrom: blue
colorTo: sky
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🩺 Diabetes Medical Chatbot — HF Space

A conversational chatbot fine-tuned on the [`abdelhakimDZ/diabetes_QA_dataset`](https://huggingface.co/datasets/abdelhakimDZ/diabetes_QA_dataset)
using **QLoRA** on top of **Google Gemma 3 1B**.

## Deployment

1. Create a new **Hugging Face Space** (SDK: **Gradio**).
2. Upload the contents of this folder.
3. Set **Hardware** to at least **T4 small** (GPU required for 4-bit inference).
4. If your adapter repo is private, add `HF_TOKEN` as a Space **Secret**.

## Model

| | |
|---|---|
| Base model | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) |
| Adapter | [`tuyishimejeand/gemma3-1b-diabetes-lora`](https://huggingface.co/tuyishimejeand/gemma3-1b-diabetes-lora) |
| Technique | QLoRA (4-bit NF4 + LoRA rank 4) |

## Disclaimer

> This chatbot is **for educational purposes only** and must not be used as a substitute
> for professional medical advice, diagnosis, or treatment.
