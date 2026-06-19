"""
Training loop for Enhanced SecureDSC
======================================
Implements the modified Algorithm 1 from the base paper with:
  - CSI-based key generation (Enhancement 1)
  - Key consistency loss L_key = MSE(k_A, k_B)
  - Adaptive lambda scheduler per epoch (Enhancement 2)

Run:
    python train.py --epochs 500 --snr 12 --batch_size 64
"""

import argparse
import math
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from model import SecureDSC, CSIKeyGenerator


# ─────────────────────────────────────────────────────────────────
# EUROPARL DATASET
# ─────────────────────────────────────────────────────────────────
class EuroParlDataset(Dataset):
    """
    EuroParl dataset using Hugging Face datasets and transformers tokenizer.
    """
    def __init__(self, tokenizer, seq_len=20, size=50000):
        print(f"[Dataset] Loading Europarl en-fr dataset (subset: {size})...")
        dataset = load_dataset("Helsinki-NLP/europarl", "en-fr", split="train")
        
        if size > 0 and size < len(dataset):
            dataset = dataset.select(range(size))
            
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = dataset["translation"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["en"]
        encoded = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.seq_len, 
            return_tensors="pt"
        )
        src = encoded["input_ids"].squeeze(0)
        tgt = src.clone()
        return src, tgt


# ─────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def cross_entropy_loss(logits, targets):
    """
    L_CE from equation (5) in the paper.
    logits : (B, L, vocab_size)  log-softmax output
    targets: (B, L)              integer word indices
    """
    B, L, V = logits.shape
    return nn.NLLLoss(ignore_index=0)(
        logits.reshape(B * L, V),
        targets.reshape(B * L)
    )


# ─────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # ── Data & Tokenizer ───────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    args.vocab_size = tokenizer.vocab_size
    print(f"[Setup] Tokenizer vocab size: {args.vocab_size}")

    dataset = EuroParlDataset(tokenizer, args.seq_len, args.dataset_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ── Model ──────────────────────────────────────────────────
    model = SecureDSC(
        vocab_size   = args.vocab_size,
        d_model      = 128,
        channel_dim  = 16,
        csi_dim      = 32,
        key_dim      = 64,
        nhead        = 8,
        num_layers   = 4
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Setup] Total parameters: {total_params:,}")

    # ── Optimizers (separate for Alice+Bob vs Eve) ─────────────
    alice_bob_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("eve_")
    ]
    eve_params = [
        p for n, p in model.named_parameters()
        if n.startswith("eve_")
    ]
    opt_ab  = optim.Adam(alice_bob_params, lr=args.lr,
                         betas=(0.9, 0.98), eps=1e-9)
    opt_eve = optim.Adam(eve_params, lr=args.lr,
                         betas=(0.9, 0.98), eps=1e-9)

    # ── Training history ───────────────────────────────────────
    history = {
        "loss_bob": [], "loss_eve": [],
        "loss_key": [], "lambda":   []
    }
    start_epoch = 1

    if os.path.exists("checkpoint_latest.pt"):
        print("[Train] Resuming from checkpoint_latest.pt...")
        checkpoint = torch.load("checkpoint_latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_ab.load_state_dict(checkpoint['opt_ab_state_dict'])
        opt_eve.load_state_dict(checkpoint['opt_eve_state_dict'])
        model.lambda_sched = checkpoint['lambda_sched']
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1

    print("\n[Train] Starting training loop ...\n")
    print(f"{'Epoch':>6}  {'L_Bob':>8}  {'L_Eve':>8}  {'L_key':>8}  {'lam':>6}")
    print("-" * 50)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss_bob = 0.0
        epoch_loss_eve = 0.0
        epoch_loss_key = 0.0
        num_batches    = 0
        num_key_batches = 0

        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(device)
            tgt = tgt.to(device)
            B   = src.size(0)

            # ── ★ Enhancement 1: generate CSI estimates ────────
            h_alice, h_bob, h_eve = CSIKeyGenerator.simulate_csi(
                B, csi_dim=32, snr_db=args.snr, device=device
            )

            # ── 4-phase alternating training (Algorithm 1) ────
            phase = batch_idx % 4

            if phase == 0:
                # Phase 0: train semantic encoder/decoder only
                opt_ab.zero_grad()
                sem_feat = model.sem_encoder(src)
                sem_out  = model.sem_decoder(sem_feat, tgt[:, :-1])
                loss     = cross_entropy_loss(sem_out, tgt[:, 1:])
                loss.backward()
                opt_ab.step()

            elif phase == 1:
                # Phase 1: train encryptor/decryptor only
                opt_ab.zero_grad()
                key_a    = model.csi_key_gen(h_alice)
                key_b    = model.csi_key_gen(h_bob)
                key_emb_a = model._key_embedding(key_a)  # Alice encrypts
                key_emb_b = model._key_embedding(key_b)  # Bob decrypts independently
                feat     = model.sem_encoder(src)
                cipher   = model.encryptor(feat, key_emb_a)
                dec_feat = model.decryptor(cipher, key_emb_b, cipher)
                loss     = cross_entropy_loss(
                    model.sem_decoder(dec_feat, tgt[:, :-1]), tgt[:, 1:]
                )
                # Key consistency loss
                l_key    = CSIKeyGenerator.consistency_loss(key_a, key_b)
                (loss + 0.5 * l_key).backward()
                opt_ab.step()
                epoch_loss_key += l_key.item()
                num_key_batches += 1

            elif phase == 2:
                # Phase 2: train whole network with joint adversarial loss
                opt_ab.zero_grad()
                bob_log, eve_log, key_a, key_b = model(
                    src, tgt[:, :-1], h_alice, h_bob, h_eve, snr_db=args.snr
                )
                l_bob    = cross_entropy_loss(bob_log, tgt[:, 1:])
                l_eve    = cross_entropy_loss(eve_log, tgt[:, 1:])
                l_key    = CSIKeyGenerator.consistency_loss(key_a, key_b)
                # Enhancement 2: use adaptive lambda
                l_joint  = model.lambda_sched.joint_loss(l_bob, l_eve)
                total    = l_joint + 0.3 * l_key
                total.backward()
                opt_ab.step()
                epoch_loss_bob += l_bob.item()
                epoch_loss_eve += l_eve.item()
                epoch_loss_key += l_key.item()
                num_batches    += 1
                num_key_batches += 1

            else:
                # Phase 3: train Eve's network independently
                opt_eve.zero_grad()
                _, eve_log, _, _ = model(
                    src, tgt[:, :-1], h_alice, h_bob, h_eve, snr_db=args.snr
                )
                l_eve_ind = cross_entropy_loss(eve_log, tgt[:, 1:])
                l_eve_ind.backward()
                opt_eve.step()

        # ── End of epoch: update λ (Enhancement 2) ────────────
        if num_batches > 0:
            avg_bob = epoch_loss_bob / num_batches
            avg_eve = epoch_loss_eve / num_batches
            avg_key = epoch_loss_key / max(num_key_batches, 1)
            new_lam = model.lambda_sched.step(avg_bob, avg_eve)

            history["loss_bob"].append(avg_bob)
            history["loss_eve"].append(avg_eve)
            history["loss_key"].append(avg_key)
            history["lambda"].append(new_lam)

            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:>6}  {avg_bob:>8.4f}  {avg_eve:>8.4f}  "
                      f"{avg_key:>8.4f}  {new_lam:>6.2f}")
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_ab_state_dict': opt_ab.state_dict(),
                'opt_eve_state_dict': opt_eve.state_dict(),
                'lambda_sched': model.lambda_sched,
                'history': history
            }, "checkpoint_latest.pt")

    # ── Save model + history ───────────────────────────────────
    torch.save(model.state_dict(), "securedsc_enhanced.pt")
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n[Done] Model saved → securedsc_enhanced.pt")
    print("[Done] History saved → training_history.json")
    return model, history


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Enhanced SecureDSC")
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--snr",          type=float, default=12.0)
    parser.add_argument("--vocab_size",   type=int,   default=1000)
    parser.add_argument("--seq_len",      type=int,   default=20)
    parser.add_argument("--dataset_size", type=int,   default=20000)
    args = parser.parse_args()
    train(args)
