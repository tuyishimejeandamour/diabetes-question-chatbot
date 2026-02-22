# Domain-Specific LLM Assistant: Diabetes Medical Q&A (Gemma 3 + QLoRA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14YMiKOwn2ZwOoi2CSJyF0XUP27fzsOKk?usp=sharing)

This project fine-tunes a pre-trained Hugging Face LLM to create a **domain-specific healthcare assistant** focused on diabetes education and support.

## Problem Statement

People living with diabetes often need fast, understandable, and context-relevant answers about symptoms, nutrition, medication routines, sick-day management, and risk prevention. General-purpose chatbots may produce answers that are too generic, inconsistent, or not sufficiently adapted to diabetes-specific needs. This creates a gap between available AI tools and practical patient-education use cases.

## Mission

Build and deploy a domain-specific diabetes assistant by fine-tuning Gemma 3 1B with QLoRA so that it can:

- Provide clearer and more relevant diabetes-focused responses than the base model.
- Support educational use through safe, concise, and conversational answers.
- Run efficiently on limited hardware (for example, free Colab GPU) using PEFT methods.
- Demonstrate measurable improvement through documented experiments and evaluation metrics.

## Project Summary

- **Domain:** Healthcare (Diabetes)
- **Task Type:** Generative Question Answering (instruction-response chatbot)
- **Base Model:** `google/gemma-3-1b-it`
- **Fine-Tuning Method:** QLoRA (4-bit quantization + LoRA via `peft`)
- **Dataset:** `abdelhakimDZ/diabetes_QA_dataset`
- **Deployment:** Gradio app (Hugging Face Space in `hf-space/`)


## Repository Structure

```text
.
├── gemma3_lora_diabetes_finetune.ipynb   # End-to-end Colab/Kaggle-friendly training pipeline
├── main.py                               # Placeholder script (not used for core training)
├── pyproject.toml
├── README.md
└── hf-space/
		├── app.py                            # Gradio inference app
		├── README.md                         # HF Space metadata and deployment notes
		└── requirements.txt
```


## 1) Dataset and Preprocessing

Dataset is loaded from Hugging Face:

- `abdelhakimDZ/diabetes_QA_dataset`

The notebook performs:

1. Missing-value removal
2. Duplicate Q&A removal
3. Text normalization (strip/cleanup)
4. Prompt construction in Gemma chat format
5. Train/validation split (default 90/10)

### Tokenization and normalization details

- **Tokenizer used:** the model-native tokenizer loaded through `AutoTokenizer` for `google/gemma-3-1b-it`.
- **Tokenization type:** SentencePiece/subword tokenizer (Gemma family), not WordPiece (WordPiece is typical for BERT-family encoders).
- **Why this is appropriate:** tokenizer-model alignment avoids vocabulary mismatch and preserves correct special-token behavior for causal generation.
- **Normalization done:** whitespace stripping, null removal, duplicate removal, and template standardization before tokenization.
- **Context-window control:** prompts are truncated to configured max length to fit model context safely.

Prompt template used during training:

```text
<start_of_turn>user
{system_prompt}

{question}
<end_of_turn>
<start_of_turn>model
{answer}
<end_of_turn>
```

## 2) Fine-Tuning Methodology (PEFT / LoRA)

The notebook `gemma3_lora_diabetes_finetune.ipynb` implements QLoRA with:

- 4-bit NF4 quantization (`bitsandbytes`)
- LoRA adapters on attention + MLP projection layers
- `trl.SFTTrainer` for supervised fine-tuning

Default key hyperparameters:

- `learning_rate = 2e-4`
- `batch_size = 2`
- `gradient_accumulation_steps = 4` (effective batch size = 8)
- `num_train_epochs = 3`
- `max_seq_length = 512`
- `lora_r = 4`, `lora_alpha = 16`, `lora_dropout = 0.05`

## 3) Hyperparameter Experiment Log (Required)

Use this table to record your experiments and the effect of changes.

| Exp ID | LR | Batch Size | Grad Accum | Epochs | LoRA r | Max Seq Len | GPU Type | Peak GPU Mem (GB) | Train Time (min) | Val Loss | ROUGE-L | Notes |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| E1 (baseline) | 2e-4 | 2 | 4 | 3 | 4 | 512 | T4 | _fill_ | _fill_ | _fill_ | _fill_ | baseline settings |
| E2 | 1e-4 | 2 | 4 | 3 | 4 | 512 | T4 | _fill_ | _fill_ | _fill_ | _fill_ | lower LR |
| E3 | 2e-4 | 2 | 4 | 2 | 8 | 512 | T4 | _fill_ | _fill_ | _fill_ | _fill_ | higher LoRA rank |

Minimum expectation: document at least **3 experiments** and discuss trade-offs.

### How to report improvement correctly

Use this formula for your primary metric (for example ROUGE-L):

```text
Percent improvement = ((FineTuned - Baseline) / Baseline) * 100
```

Only report “>=10% improvement” if this computed value is at least 10.

## 4) Evaluation

Implemented in notebook section **“Evaluate Model Performance”**:

- Quantitative metric: **ROUGE-1, ROUGE-2, ROUGE-L** (mean F1)
- Qualitative analysis: generated answers vs reference answers
- Visualizations: training curves and ROUGE score plots

Recommended additions (optional but strong academically): BLEU, perplexity, and error analysis by question type.

### Metric coverage checklist for report/video

- ROUGE-1 / ROUGE-2 / ROUGE-L
- Qualitative sample inspection
- Perplexity trend (derived from training/eval loss curves)

## 5) Base Model vs Fine-Tuned Model Comparison

You should show side-by-side outputs for the same prompts:

| Prompt | Base Model Output (Gemma 3 1B IT) | Fine-Tuned Output (Gemma+LoRA) | Observed Difference |
|---|---|---|---|
| Diabetes symptoms question | _fill_ | _fill_ | _fill_ |
| Sick-day management question | _fill_ | _fill_ | _fill_ |
| Out-of-domain question | _fill_ | _fill_ | _fill_ |

Include at least one **out-of-domain** prompt to show safe handling behavior.

## 6) Deployment (User Interaction)

The deployment app is in `hf-space/app.py` using Gradio.

User experience features already included:

- Clean single-panel chat layout with example prompts
- Immediate text input + one-click send
- Focused diabetes-assistant behavior via system prompt
- Built-in instructional text for first-time users

### Run locally (for demo)

```bash
cd hf-space
pip install -r requirements.txt
python app.py
```

### Hugging Face Space deployment

1. Create a new Gradio Space
2. Upload files from `hf-space/`
3. Set GPU hardware (T4 recommended)
4. Add `HF_TOKEN` secret if adapter/model access is gated


## 7) How to Run End-to-End in Google Colab

1. Open notebook: `gemma3_lora_diabetes_finetune.ipynb`
2. Enable GPU runtime (`Runtime` → `Change runtime type` → `T4 GPU`)
3. Run cells sequentially from top to bottom
4. Authenticate Hugging Face (`HF_TOKEN`) for gated model access
5. Train adapter, evaluate metrics, run demo Q&A cells
6. Save/push LoRA adapter to Hugging Face Hub



## References

- Hugging Face dataset: <https://huggingface.co/datasets/abdelhakimDZ/diabetes_QA_dataset>
- Base model: <https://huggingface.co/google/gemma-3-1b-it>
- LoRA paper: <https://arxiv.org/abs/2106.09685>
- QLoRA paper: <https://arxiv.org/abs/2305.14314>
