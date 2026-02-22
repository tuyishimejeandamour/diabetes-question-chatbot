# Demo Video Script (5–10 Minutes)

## Goal of this script
Use this as a narration guide for your submission video so you cover all required grading points: dataset, fine-tuning, PEFT/LoRA, evaluation, comparison with base model, deployment, and key insights.

---

## 0:00–0:40 — Introduction

**Say:**
"Hello, my name is [Your Name], and this is my individual summative project on domain-specific LLM fine-tuning. I built a healthcare assistant focused on diabetes education by fine-tuning Google Gemma 3 1B using QLoRA. The objective is to improve response quality for diabetes-related questions while maintaining safe, educational behavior."

**Show:**
- GitHub repository root
- Project title in README

---

## 0:40–1:30 — Problem Statement and Domain Choice

**Say:**
"I selected healthcare because reliable medical information is high-impact, and diabetes has many practical question-answer use cases such as symptoms, diet, medication, sick-day management, and risk prevention. I used a domain-specific Q&A dataset to specialize the model."

**Show:**
- README section describing domain and task
- Dataset link

---

## 1:30–2:30 — Dataset and Preprocessing

**Say:**
"The dataset used is `abdelhakimDZ/diabetes_QA_dataset` from Hugging Face. In preprocessing, I removed nulls and duplicates, cleaned text, and formatted each pair into a structured instruction-response prompt using Gemma chat tokens. This helps the model learn conversational response behavior in the medical domain."

**Show:**
- Notebook cells for dataset loading and cleaning
- Prompt template cell
- Example formatted training sample

---

## 2:30–3:40 — Model and Fine-Tuning Approach

**Say:**
"The base model is `google/gemma-3-1b-it`. Because Colab free GPUs are limited, I used parameter-efficient fine-tuning with QLoRA: 4-bit quantization plus LoRA adapters. Instead of updating all model parameters, only small low-rank matrices are trained. This reduces memory usage while keeping performance strong for specialized tasks."

**Show:**
- Hyperparameter configuration cell
- LoRA config cell
- Parameter counts showing trainable vs frozen

---

## 3:40–4:40 — Training Setup and Experiments

**Say:**
"My baseline hyperparameters were learning rate 2e-4, batch size 2, gradient accumulation 4, and 3 epochs. I also tracked experiment variations and documented training time and GPU memory usage for each run. This supports objective comparison of settings and reproducibility."

**Show:**
- Training arguments cell
- Experiment table in README
- Any logs showing runtime and memory

**Important to include verbally:**
- "I tested at least three settings and documented trade-offs."

---

## 4:40–6:00 — Evaluation Results

**Say:**
"For quantitative evaluation, I used ROUGE-1, ROUGE-2, and ROUGE-L on a validation subset. I also performed qualitative testing by comparing generated responses with reference answers. The model gives more relevant and domain-aligned answers after fine-tuning."

**Show:**
- ROUGE evaluation cell output
- ROUGE plot image
- 2–3 qualitative examples from notebook output

---

## 6:00–7:20 — Base vs Fine-Tuned Comparison

**Say:**
"To demonstrate the impact of fine-tuning, I compared the base model and fine-tuned model on the same prompts. The fine-tuned model is generally more specific to diabetes context, more consistent in terminology, and more actionable for patient education. I also tested an out-of-domain prompt to observe how the model handles questions outside scope."

**Show:**
- Comparison table in README
- Side-by-side generated outputs

---

## 7:20–8:30 — Deployment Demo

**Say:**
"I deployed the assistant with Gradio in a Hugging Face Space. Users can type a question and receive a generated response in real time. The interface is lightweight and suitable for practical testing."

**Show:**
- `hf-space/app.py`
- Running UI (local or HF Space)
- 1–2 live queries

---

## 8:30–9:20 — Limitations, Safety, and Ethics

**Say:**
"This system is for educational purposes and not a replacement for professional medical advice. LLMs can produce incorrect or incomplete outputs, so medical decisions should always be validated with qualified professionals. Future improvements include stronger safety guardrails, retrieval-augmented generation, and additional evaluation metrics such as BLEU, perplexity, and BERTScore."

**Show:**
- Disclaimer in README or app
- Future work section

---

## 9:20–10:00 — Closing

**Say:**
"In summary, this project demonstrates end-to-end domain adaptation of an LLM using PEFT on limited hardware: from dataset preparation to QLoRA fine-tuning, evaluation, model comparison, and deployment. Thank you for watching."

**Show:**
- Final project checklist in README
- Repo links (notebook, app, adapter)

---

## Submission Readiness Checklist (for your recording)

- [ ] Notebook runs end-to-end in Colab GPU
- [ ] Experiment table has real numbers (not placeholders)
- [ ] Base vs fine-tuned comparison table completed
- [ ] Colab badge/link is added and tested
- [ ] Demo video is 5–10 minutes and includes live interaction
- [ ] All external sources are cited
- [ ] Disclaimer is clearly stated