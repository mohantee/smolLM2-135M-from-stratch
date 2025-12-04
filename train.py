import os
import time
import argparse
import math
import torch
from transformers import AutoTokenizer
from model import SmolLM, SmolConfig

# ---------------------------------------------------------------------
# Simple DataLoader
# ---------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, input_file="input.txt"):
        self.B = B
        self.T = T

        with open(input_file, "r") as f:
            text = f.read()

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"{len(self.tokens) // (B*T)} batches per full pass")

        self.position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.position: self.position + (B * T + 1)]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.position += B * T
        if self.position + (B * T + 1) > len(self.tokens):
            self.position = 0

        return x, y


# ---------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, filepath, lr_config):
    # ckpt = {
    #     "step": step,
    #     "loss": loss,
    #     "model_state": model.state_dict(),
    #     "optimizer_state": optimizer.state_dict(),
    #     "lr_config": lr_config,
    # }
    torch.save(model.state_dict(), filepath)
    print(f"\nSaved checkpoint: {filepath}")


def load_checkpoint(filepath, device, model, optimizer):
    ckpt = torch.load(filepath, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    step = ckpt["step"]
    loss = ckpt["loss"]
    lr_config = ckpt.get("lr_config", None)

    print(f"Loaded checkpoint '{filepath}' at step={step}, loss={loss}")
    return step, loss, lr_config


# ---------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------
def train(total_steps, ckpt_path, save_path, log_interval=100):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    torch.set_float32_matmul_precision("high")

    # Config matching SmolLM2-135M
    config = SmolConfig(
        hidden_size=576,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        intermediate_size=1536,
        vocab_size=49152,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )

    model = SmolLM(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
    )

    loader = DataLoaderLite(B=4, T=256)

    loss_fn = torch.nn.CrossEntropyLoss()

    # LR schedule settings
    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 100
    original_max_steps = total_steps

    # Resume?
    start_step = 0
    last_loss = None
    lr_config = {
        "original_max_steps": original_max_steps,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
    }

    if ckpt_path and os.path.exists(ckpt_path):
        start_step, last_loss, old_lr_config = load_checkpoint(
            ckpt_path, device, model, optimizer
        )
        if old_lr_config and "original_max_steps" in old_lr_config:
            original_max_steps = old_lr_config["original_max_steps"]
            lr_config = old_lr_config
        print(f"Resuming from step {start_step}")

    print(f"\nTraining from step {start_step} → {total_steps}")
    print(f"LR schedule enabled (max={max_lr}, min={min_lr})")
    print(f"Using max_steps={original_max_steps} for LR\n")

    # -------------------------------
    # Main Training Loop
    # -------------------------------
    loss = torch.tensor(0.0, device=device)
    for step in range(start_step, total_steps):
        t0 = time.time()

        # LR schedule update
        lr = get_lr(step, warmup_steps, original_max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Next batch
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(x)
        loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        tok_per_sec = (loader.B * loader.T) / (t1 - t0)

        if step % log_interval == 0:
            print(f"step {step} | loss {loss.item():.4f} | lr {lr:.6f} | tok/s {tok_per_sec:8.1f}")

        # Save checkpoint periodically
        if step > 0 and step % 1000 == 0:
            ckpt_name = f"checkpoint_step_{step}.pt"
            save_checkpoint(model, optimizer, step, loss.item(), ckpt_name, lr_config)

    # Final save
    save_checkpoint(model, optimizer, total_steps, loss.item(), save_path, lr_config)
    print(f"\nFinal loss: {loss.item():.4f}")
    print(f"Model saved → {save_path}")
    

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmolLM (steps-based)")

    parser.add_argument("--steps", type=int, required=True, help="Total training steps")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save", type=str, default="checkpoint_final.pt", help="Save path")
    parser.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()

    train(
        total_steps=args.steps,
        ckpt_path=args.ckpt,
        save_path=args.save,
        log_interval=args.log_interval,
    )
