"""
Gradio Space — Diabetes Medical Chatbot
Loads the fine-tuned Gemma 3 1B LoRA adapter from Hugging Face Hub.

Deploy this folder as a Hugging Face Space (SDK: gradio).
Make sure to:
  1. Set the HF_TOKEN secret in Space settings if your adapter repo is private.
  2. Choose a GPU hardware tier (T4 small or better).
"""

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Model identifiers ────────────────────────────────────────────────────────
BASE_MODEL_ID = "google/gemma-3-1b-it"
ADAPTER_ID    = "tuyishimejeand/gemma3-1b-diabetes-lora"

SYSTEM_PROMPT = (
    "You are a knowledgeable and compassionate diabetes medical assistant. "
    "Answer the user's questions accurately and concisely using evidence-based "
    "medical information. Always recommend consulting a healthcare professional "
    "for personalised medical advice."
)

# ── Load model at startup ────────────────────────────────────────────────────
print(f"Loading tokenizer from {BASE_MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model with 4-bit quantisation …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config if torch.cuda.is_available() else None,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

print(f"Attaching LoRA adapter from {ADAPTER_ID} …")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()
print("✅ Model ready.")


# ── Inference helper ─────────────────────────────────────────────────────────
def build_prompt(history: list[dict], user_message: str) -> str:
    """Convert Gradio message history + new user turn into a Gemma chat prompt."""
    prompt = ""
    for turn in history:
        if turn["role"] == "user":
            prompt += f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{turn['content']}\n<end_of_turn>\n"
        else:
            prompt += f"<start_of_turn>model\n{turn['content']}\n<end_of_turn>\n"
    prompt += f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{user_message}\n<end_of_turn>\n<start_of_turn>model\n"
    return prompt


def respond(
    message: str,
    history: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    prompt = build_prompt(history, message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else None,
            top_p=float(top_p) if temperature > 0 else None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
    title="Diabetes Medical Chatbot",
    css="""
        .disclaimer { background: #fff3cd; border-left: 4px solid #ffc107;
                      padding: 10px 16px; border-radius: 4px; margin-bottom: 12px; }
        footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🩺 Diabetes Medical Chatbot
        Fine-tuned **Gemma 3 1B** (QLoRA) · [`tuyishimejeand/gemma3-1b-diabetes-lora`](https://huggingface.co/tuyishimejeand/gemma3-1b-diabetes-lora)
        """
    )
    gr.HTML(
        '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This chatbot is for '
        "educational purposes only and must <strong>not</strong> replace professional "
        "medical advice, diagnosis, or treatment.</div>"
    )

    chat = gr.ChatInterface(
        fn=respond,
        type="messages",
        additional_inputs=[
            gr.Slider(64, 512, value=256, step=32, label="Max new tokens"),
            gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
        ],
        additional_inputs_accordion=gr.Accordion("⚙️ Generation settings", open=False),
        examples=[
            "What are the early symptoms of type 2 diabetes?",
            "How should I manage my blood sugar when I am sick?",
            "What foods should a person with diabetes avoid?",
            "What is HbA1c and what does my result mean?",
            "How does regular exercise affect blood glucose levels?",
            "What should I include in a diabetes emergency kit?",
        ],
        cache_examples=False,
        submit_btn="Send",
        retry_btn="🔁 Retry",
        undo_btn="↩ Undo",
        clear_btn="🗑️ Clear",
    )

if __name__ == "__main__":
    demo.launch()
