import argparse
import os
import time
import numpy as np
import torch
from transformer_lm import TransformerLM
from transformer_utils import get_lr_cosine_schedule, clip_gradients, cross_entropy_loss, load_checkpoint, save_checkpoint, get_batch
from AdamW import AdamW

def main(args):
    device = torch.device(args.device)
    
    # Load training and validation datasets in memory-mapped mode.
    print("Loading training data...")
    train_data = np.load(args.train_data, mmap_mode='r')
    print("Loading validation data...")
    val_data = np.load(args.val_data, mmap_mode='r')
    
    # Initialize model.
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    start_iter = 0
    if args.resume_checkpoint:
        start_iter = load_checkpoint(args.resume_checkpoint, model, optimizer)
        print(f"Resumed checkpoint from iteration {start_iter}")
    
    start_time = time.time()
    for it in range(start_iter, args.iterations):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)  # logits shape: (B, T, vocab_size)
        loss = cross_entropy_loss(logits.view(-1, args.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()
        
        # Update learning rate via cosine schedule.
        lr = get_lr_cosine_schedule(it, args.learning_rate, args.min_lr, args.warmup_iters, args.iterations)
        for group in optimizer.param_groups:
            group['lr'] = lr
        
        # Log training progress.
        if it % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {it:06d}: train loss = {loss.item():.4f}, lr = {lr:.6f}, elapsed = {elapsed:.2f}s")
        
        # Periodically evaluate on the validation set.
        if it % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, args.device)
                val_logits = model(x_val)
                val_loss = cross_entropy_loss(val_logits.view(-1, args.vocab_size), y_val.view(-1))
            print(f"Iteration {it:06d}: validation loss = {val_loss.item():.4f}")
        
        # Periodically save checkpoints.
        if it % args.ckpt_every == 0 and it > 0:
            cp_path = os.path.join(args.ckpt_dir, f"checkpoint_{it:06d}.pt")
            save_checkpoint(model, optimizer, it, cp_path)
            print(f"Saved checkpoint to {cp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer LM")
    
    # Model hyperparameters.
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=128, help="Context length.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimensionality.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward inner dimension.")
    parser.add_argument("--attn_pdrop", type=float, default=0.1, help="Attention dropout probability.")
    parser.add_argument("--residual_pdrop", type=float, default=0.1, help="Residual dropout probability.")
    
    # Optimizer hyperparameters.
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    
    # Training loop hyperparameters.
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--iterations", type=int, default=50000, help="Total iterations.")
    parser.add_argument("--log_every", type=int, default=100, help="Logging frequency (iterations).")
    parser.add_argument("--val_every", type=int, default=500, help="Validation frequency (iterations).")
    parser.add_argument("--ckpt_every", type=int, default=1000, help="Checkpoint frequency (iterations).")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Warm-up iterations for LR schedule.")
    
    # Device.
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., 'cpu' or 'cuda:0').")
    
    # Data and checkpoint paths.
    parser.add_argument("--train_data", type=str,
                        default="/content/ECE491B-assignment1/serialized/openwebtext_train_combined.npy",
                        help="Path to training data (numpy array, memmapped).")
    parser.add_argument("--val_data", type=str,
                        default="/content/ECE491B-assignment1/serialized/openwebtext_dev.npy",
                        help="Path to validation data (numpy array, memmapped).")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/content/ECE491B-assignment1/checkpoints",
                        help="Directory for saving checkpoints.")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Checkpoint to resume from (if any).")
    
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    main(args)