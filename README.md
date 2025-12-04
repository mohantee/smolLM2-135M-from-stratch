# SmolLM2-135M — From-scratch training & Hugging Face Space

This repository contains a compact implementation of a SmolLM2-style language model (~135M parameters) and a small training loop used to train it from scratch, plus a Gradio-based Hugging Face Space for inference and demoing the model.

## Project intention
- Provide a minimal, readable PyTorch implementation of a SmolLM2-like transformer (GQA, RoPE, RMSNorm, gated MLP).
- Demonstrate end-to-end from-scratch training on raw text (single-file dataset `input.txt`) with a tiny, easy-to-inspect training loop (`train.py`).
- Provide a simple Gradio app in `hf_files/` for quick local demo or Hugging Face Space deployment.

## Relation to Hugging Face SmolLM (analysis)

This implementation was built after analyzing publicly available Smol/SmolLM-style models and the tokenizer reference used in the code (e.g. `HuggingFaceTB/SmolLM-135M`). The following notes summarize what was matched from those references, what was simplified, and how to approach converting or loading official Hugging Face weights.

What was matched / intentionally mirrored
- Architecture: transformer with Grouped Query Attention (GQA) + separate KV heads, matching the SmolLM design choice to reduce KV head count while keeping many query heads.
- RoPE: rotary positional embeddings are implemented with precomputed cos/sin buffers for numerical stability and to support an offset (`start_pos`) when using cached KV for autoregressive decoding.
- Normalization & activation: RMSNorm and SiLU (Gated MLP) are used, consistent with SmolLM/SmolLM2 style models.
- Weight-tying: embedding and LM head weights are tied where possible (the code attempts to set `lm_head.weight = embed_tokens.weight`).
- Initialization: linear layers use normal initialization with a smaller std for attention output projections (the code follows a Smol/LLaMA-like initialization heuristic).

## High-level architecture

- Tokenization: uses a pretrained tokenizer referenced in the code (AutoTokenizer from "HuggingFaceTB/SmolLM-135M").
- Model: `SmolLM` (see `model.py`) — transformer-style model using:
	- Grouped Query Attention (GQA) to reduce KV heads while keeping many query heads
	- Rotary positional embeddings (RoPE) with precomputed cos/sin buffers
	- RMSNorm normalization
	- Gated feedforward (SiLU activation)
- Training loop: `train.py` contains a small DataLoader, LR schedule, optimizer (AdamW), checkpoint saving and resume logic.
- Inference / demo: `hf_files/app.py` contains a Gradio interface and a simple autoregressive sampling loop.

## Files and purpose

- `model.py` — Primary model implementation and `SmolConfig`. This is the core transformer code (GQA, RoPE, RMSNorm, Gated MLP).
- `train.py` — Training script with a minimal DataLoader (reads `input.txt`), learning-rate schedule, optimizer (AdamW), checkpointing and CLI. Run with `--steps` to train.
- `input.txt` — (expected) raw text training corpus. The training DataLoader tokenizes this file and creates batches.
- `hf_files/` — Files used for the Hugging Face Space / demo:
	- `hf_files/app.py` — Gradio application for inference (loads checkpoint if present).
	- `hf_files/model.py` — copy/variant of `model.py` used inside the Space (same architecture).
	- `hf_files/README.md` — Space metadata/config.
	- `hf_files/requirements.txt` — dependencies for the Space (torch, transformers, gradio).
- `requirements.txt` — top-level dependency hints (torch, transformers, gradio).
- Checkpoints that may be present in the repo:
	- `model_checkpoint.pt`, or `checkpoint_step_{N}.pt` — saved model states created by `train.py` or exported for the Space.

If you add other scripts, keep their purpose documented here.

## Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r hf_files/requirements.txt
```

Note: If you plan to train and have a CUDA-capable GPU, install a PyTorch version matched to your CUDA: https://pytorch.org/get-started/locally/ (the `torch` in `requirements.txt` is a placeholder).

## Training (local)

The provided `train.py` is intentionally minimal and easy to read. Key points:

- It expects a plain text file `input.txt` in the repo root. The DataLoader tokenizes the entire file and yields contiguous (B, T) batches.
- Default loader in the script uses B=4, T=256 (adjust in the code or change DataLoaderLite usage).
- Example training run (PowerShell):

```powershell
python train.py --steps 20000 --save checkpoint_final.pt --log-interval 100
```

CLI options in `train.py`:
- `--steps` (int, required) — total training steps
- `--ckpt` (str) — optional checkpoint file to resume from
- `--save` (str) — final save path (default `checkpoint_final.pt`)
- `--log-interval` (int) — logging frequency

Training notes and best practices
- Use a machine with a GPU and sufficient VRAM for practical training. The default config (576 hidden, 30 layers) results in ~135M parameters.
- Batch size and sequence length significantly affect memory usage. Lower `B` or `T` to fit smaller GPUs.
- The script saves model.state_dict() by default for checkpoints. Resume support is implemented — the code looks for `model_state` and `optimizer_state` keys if a dict checkpoint was saved.

## Inference / Demo

To run the Gradio demo locally (similar to how the Space is configured):

```powershell
cd hf_files
pip install -r requirements.txt
python app.py
```

Behavior:
- `hf_files/app.py` tries to load `model_checkpoint.pt` by default. If none are present it will run with random weights (not useful for generation).
- The sampling loop supports temperature, top-k, top-p and a repetition penalty.

## Checkpoints & outputs

During training you'll see saved checkpoint files such as `checkpoint_step_{N}.pt` and the final `checkpoint_final.pt` (or whatever you pass to `--save`).

Expected artifacts after a training run
- model state dict files (e.g. `checkpoint_step_1000.pt`) — these contain the learned weights.
- training console logs with step/loss/lr and tokens-per-second metrics.


## Tokenizer

Both `train.py` and `hf_files/app.py` reference a tokenizer from `transformers.AutoTokenizer` (example: `HuggingFaceTB/SmolLM-135M`). Make sure the tokenizer vocab size matches `SmolConfig.vocab_size` or update the config accordingly.


## License & contact

This repository contains research/educational code. No explicit license file is included — add a LICENSE if you plan to redistribute.

If you want help adapting the training loop or deploying to a Space, open an issue or message the repo owner.

----

Files changed: this README updated to provide full project details, usage, and tips.

