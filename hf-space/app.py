
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
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

print(f"Attaching LoRA adapter from {ADAPTER_ID} …")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()
print("✅ Model ready.")


# ── Inference helper ─────────────────────────────────────────────────────────
def _parse_history_message(message_item):
    """Normalize one history item from Gradio into (role, content)."""
    if isinstance(message_item, dict):
        return message_item.get("role"), message_item.get("content")

    if isinstance(message_item, (list, tuple)) and len(message_item) >= 2:
        return message_item[0], message_item[1]

    role = getattr(message_item, "role", None)
    content = getattr(message_item, "content", None)
    return role, content


def build_prompt(history: list | None, user_message: str) -> str:
    """Build a Gemma prompt from Gradio history and the latest user message."""
    prompt = ""

    for item in history or []:
        role, content = _parse_history_message(item)
        if not content:
            continue

        role = (role or "").lower()
        if role in {"user", "human"}:
            prompt += f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{content}\n<end_of_turn>\n"
        elif role in {"assistant", "model", "bot"}:
            prompt += f"<start_of_turn>model\n{content}\n<end_of_turn>\n"

    prompt += (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{user_message}\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    return prompt


def respond(
    message: str,
    history: list | None,
    # Remove arguments that are no longer passed by ChatInterface
) -> str:
    # Set default generation parameters since we removed the sliders
    max_new_tokens = 256
    temperature = 0.7
    top_p = 0.9

    prompt = build_prompt(history, message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    target_device = getattr(model, "device", None)
    if target_device is None:
        target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

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


# ── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* 1. FORCE LIGHT THEME GLOBALLY */
:root, body, .gradio-container, .dark, .light {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #ffffff !important;
    --body-background-fill: #ffffff !important;
    --body-text-color: #1f2937 !important;
    --color-accent-soft: #f3f4f6 !important;
    --border-color-primary: #e5e7eb !important;
}

/* 2. LAYOUT & WIDTH */
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding-top: 2rem !important;
}

/* 3. HEADER */
.header-title {
    border-bottom: 1px solid #e5e7eb !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
}
.header-title h3 {
    color: #000000 !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* 4. CHATBOT CONTAINER */
#chatbot {
    height: 70vh !important;
    min-height: 500px !important;
    background-color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}
/* Hide the default "Chatbot" label */
#chatbot > .label-wrap { display: none !important; }

/* 5. MESSAGE BUBBLES */
/* The wrapper around messages */
.bubble-wrap {
    background-color: #ffffff !important;
    padding: 0 !important;
}

/* User message bubble */
.message.user {
    background-color: #f3f4f6 !important; /* Light gray */
    border: 1px solid #e5e7eb !important;
    border-radius: 12px 12px 2px 12px !important;
    color: #111827 !important;
    padding: 12px 16px !important;
    margin-left: auto !important; /* Align right */
    max-width: 80% !important;
}

/* Bot message bubble */
.message.bot {
    background-color: #ffffff !important; /* White */
    border: none !important;
    color: #374151 !important;
    padding: 12px 0 !important;
    margin-right: auto !important; /* Align left */
    max-width: 90% !important;
    box-shadow: none !important;
}

/* 6. EXAMPLES (The buttons below the chat) */
/* Target the container and the buttons aggressively */
.examples, .examples-container, div[data-testid="examples"] {
    background-color: #ffffff !important;
    border: none !important;
}
.examples button, div[data-testid="examples"] button, .gallery-item {
    background-color: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 0.9rem !important;
    margin: 4px !important;
}
.examples button:hover, div[data-testid="examples"] button:hover, .gallery-item:hover {
    background-color: #f9fafb !important;
    border-color: #d1d5db !important;
}

/* 7. INPUT AREA */
/* The row containing the textbox and button */
.gradio-row:has(textarea), .form {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    padding: 6px 12px !important;
    margin-top: 10px !important;
    align-items: center !important;
    display: flex !important;
}

/* The actual text input */
textarea {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important; 
    resize: none !important; 
    padding: 8px !important;
    color: #111827 !important;
}
textarea::placeholder {
    color: #9ca3af !important;
}

/* Send Button */
button.primary, button#component-10 { /* Fallback ID targeting */
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important; 
    padding: 0 !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
}
button.primary:hover {
    background-color: #333333 !important;
}

/* 8. CLEANUP */
footer { display: none !important; }
.block { border: none !important; background: transparent !important; box-shadow: none !important; }
"""

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Diabetes Agent", css=CUSTOM_CSS, theme=gr.themes.Default(neutral_hue="slate")) as demo:
    
    # Header
    with gr.Row(elem_classes=["header-title"]):
        gr.Markdown("### ⚫ Diabetes Agent")

    # ── Chat interface ───────────────────────────────────────────────────────
    chat_interface = gr.ChatInterface(
        fn=respond,
        textbox=gr.Textbox(
            placeholder="Type your question here...",
            show_label=False,
            container=False, # Replaces lines=1, max_lines=5 logic often used
            scale=7,
            autofocus=True,
        ),
        # minimal interface
        examples=[
            ["What are the early symptoms of type 2 diabetes?"],
            ["How should I manage my blood sugar when I am sick?"],
            ["What foods should a person with diabetes avoid?"],
        ],
        cache_examples=False,
        submit_btn="➤", 
    )

    # Customize the internal chatbot component
    chat_interface.chatbot.elem_id = "chatbot"
    chat_interface.chatbot.show_label = False
    chat_interface.chatbot.avatar_images = (None, None) # No avatars for cleaner look
    
    # Custom placeholder
    chat_interface.chatbot.placeholder = (
        "<div style='text-align:center; padding-top: 100px; color: #9ca3af;'>"
        "<p>I can help you understand diabetes data and management.</p>"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch()