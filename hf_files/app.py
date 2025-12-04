import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import sys

# Add path for model.py
sys.path.append('.')

from model import SmolLM, SmolConfig


# -----------------------------------------------------------
# Load Model
# -----------------------------------------------------------
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = "checkpoint_model21.pt"
    if not os.path.exists(checkpoint_path):
        print("checkpoint_model21.pt not found, trying model_checkpoint.pt")
        checkpoint_path = "model_checkpoint.pt"

    checkpoint = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        print("No checkpoint found — using random weights.")

    config = SmolConfig(
        hidden_size=576,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        intermediate_size=1536,
        vocab_size=49152,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )

    model = SmolLM(config).to(device)

    if checkpoint is not None:
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully.")
    return model, device


# -----------------------------------------------------------
# Load Tokenizer
# -----------------------------------------------------------
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    return tokenizer


print("Loading model...")
model, device = load_model()
tokenizer = load_tokenizer()
print("Model & tokenizer ready.")


# -----------------------------------------------------------
# Generation Function
# -----------------------------------------------------------
def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1
):

    if not prompt.strip():
        return "Please enter a prompt!"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if input_ids.shape[1] > 1024:
        return "Prompt too long! Maximum is 1024 tokens."

    generated_ids = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Limit window to context size
            input_chunk = torch.tensor(
                [generated_ids[-1024:]],
                dtype=torch.long,
                device=device
            )

            logits = model(input_chunk)
            if isinstance(logits, tuple):
                logits = logits[0]

            logits = logits[0, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok in set(generated_ids):
                    logits[tok] /= repetition_penalty

            # Top-k
            if top_k > 0:
                kth = torch.topk(logits, top_k)[0][..., -1]
                logits[logits < kth] = -float("inf")

            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                logits[sorted_idx[cutoff]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# -----------------------------------------------------------
# Gradio UI (Gradio 3.x Compatible)
# -----------------------------------------------------------
examples = [
    ["First Citizen:", 150, 0.8, 40, 0.9, 1.2],
    ["ROMEO:", 200, 0.9, 50, 0.9, 1.1],
    ["To be or not to be", 150, 0.7, 40, 0.85, 1.2],
]


with gr.Blocks(title="SmolLM2-135M Text Generator") as demo:

    # ---------------------------------------------------------
    # TOP: MODEL DETAILS
    # ---------------------------------------------------------
    gr.Markdown(
        """
        # 🤖 SmolLM2-135M — Custom Trained Model

        This is a **SmolLM2-135M model trained from scratch!**
        Enter a prompt and watch the model generate creative continuations.  
        
        ### **Model Details**
        - **Architecture:** SmolLM2 (Transformer with Grouped Query Attention)
        - **Parameters:** 135M  
        - **Training:** Custom trained from scratch  
        - **Context Length:** 1024 tokens  
        """
    )

    with gr.Row():

        # ---------------------
        # Left panel: inputs
        # ---------------------
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=5)

            max_tokens = gr.Slider(10, 500, 100, step=10, label="Max New Tokens")
            temperature = gr.Slider(0.1, 2.0, 0.8, step=0.1, label="Temperature")
            top_k_slider = gr.Slider(0, 100, 50, step=5, label="Top-K")
            top_p_slider = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-P")
            repetition = gr.Slider(1.0, 2.0, 1.1, step=0.05, label="Repetition Penalty")

            generate = gr.Button("Generate", variant="primary")

        # ---------------------
        # Right panel: output
        # ---------------------
        with gr.Column():
            output = gr.Textbox(label="Generated Text", lines=20)
            copy_btn = gr.Button("Copy Output")

    # Generation button
    generate.click(
        generate_text,
        [prompt, max_tokens, temperature, top_k_slider, top_p_slider, repetition],
        output,
    )

    # Copy helper (works in Gradio 3.x)
    copy_btn.click(lambda x: x, [output], output)

    # Example prompts
    gr.Examples(
        examples=examples,
        fn=generate_text,
        inputs=[prompt, max_tokens, temperature, top_k_slider, top_p_slider, repetition],
        outputs=output,
        cache_examples=False,
    )

    # ---------------------------------------------------------
    # BOTTOM: TRAINING DETAILS
    # ---------------------------------------------------------
    gr.Markdown(
        """
        ---

        ## 🔧 **Training & Parameter Details**

        ### **Training Setup**
        - Trained entirely **from scratch**  
        - Custom PyTorch training loop  
        - Used **Grouped Query Attention (GQA)** for faster inference  
        - Activation: **SiLU**  
        - Normalization: **RMSNorm**  
        - Positional Encoding: **Rotary Embeddings (RoPE)**  
        - Optimizer: **AdamW**  

        ### **Model Hyperparameters**
        - **Hidden Size:** 576  
        - **Layers:** 30  
        - **Attention Heads:** 9  
        - **Key/Value Heads:** 3 (GQA)  
        - **Feedforward Size:** 1536  
        - **Vocabulary Size:** 49,152  
        - **Max Sequence Length:** 1024  
        - **Total Parameters:** ~135 Million  
 

        """
    )


if __name__ == "__main__":
    demo.launch()
