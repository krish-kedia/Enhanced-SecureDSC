"""
Training loop for Enhanced SecureDSC
======================================
Implements the modified Algorithm 1 from the base paper with:
  - CSI-based key generation (Enhancement 1)
  - Key consistency loss L_key = MSE(k_A, k_B)
  - Adaptive lambda scheduler per epoch (Enhancement 2)

Run:
    python train.py --epochs 150 --snr 12 --batch_size 512
"""

import argparse
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model import SecureDSC, CSIKeyGenerator


# ─────────────────────────────────────────────────────────────────
# EUROPARL DATASET
# ─────────────────────────────────────────────────────────────────
class EuroParlDataset(Dataset):
    def __init__(self, seq_len=20, cache_path=None):
        if cache_path is None:
            cache_path = f"europarl_cache_seqlen{seq_len}.pt"
        print(f"[Dataset] Loading pre-tokenized cache from {cache_path}...")
        self.input_ids = torch.load(cache_path)
        print(f"[Dataset] Loaded {len(self.input_ids):,} samples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        src = self.input_ids[idx]
        return src, src


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

    dataset = EuroParlDataset(seq_len=args.seq_len)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[Setup] Split dataset into {train_size} train and {test_size} test samples")

    loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # ── Model ──────────────────────────────────────────────────
    model = SecureDSC(
        vocab_size   = args.vocab_size,
        d_model      = 128,
        channel_dim  = 16,
        csi_dim      = 64,
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
        # Override learning rate with current args (enables LR fine-tuning)
        for pg in opt_ab.param_groups:
            pg['lr'] = args.lr
        for pg in opt_eve.param_groups:
            pg['lr'] = args.lr
        print(f"[Train] Resumed from epoch {checkpoint['epoch']}  →  starting epoch {start_epoch}  (lr={args.lr})")

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
            h_alice, h_bob, _ = CSIKeyGenerator.simulate_csi(
                B, csi_dim=64, snr_db=args.snr, device=device
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
                torch.nn.utils.clip_grad_norm_(alice_bob_params, max_norm=1.0)
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
                (loss + 1.0 * l_key).backward()
                torch.nn.utils.clip_grad_norm_(alice_bob_params, max_norm=1.0)
                opt_ab.step()
                epoch_loss_key += l_key.item()
                num_key_batches += 1

            elif phase == 2:
                # Phase 2: train whole network with joint adversarial loss
                opt_ab.zero_grad()
                bob_log, eve_log, key_a, key_b = model(
                    src, tgt[:, :-1], h_alice, h_bob, snr_db=args.snr
                )
                l_bob    = cross_entropy_loss(bob_log, tgt[:, 1:])
                l_eve    = cross_entropy_loss(eve_log, tgt[:, 1:])
                l_key    = CSIKeyGenerator.consistency_loss(key_a, key_b)
                # Enhancement 2: use adaptive lambda
                l_joint  = model.lambda_sched.joint_loss(l_bob, l_eve)
                total    = l_joint + 1.0 * l_key
                total.backward()
                torch.nn.utils.clip_grad_norm_(alice_bob_params, max_norm=1.0)
                opt_ab.step()
                epoch_loss_bob += l_bob.item()
                epoch_loss_eve += l_eve.item()
                epoch_loss_key += l_key.item()
                num_batches    += 1
                num_key_batches += 1

            else:
                # Phase 3: train Eve's network independently
                opt_eve.zero_grad()
                # Bypass Bob's forward pass and freeze Alice's graph to save GPU cycles
                with torch.no_grad():
                    _, y_bar, _ = model.forward_alice(src, h_alice, snr_db=args.snr)
                rnd = torch.randn(B, model.d_model, device=device).unsqueeze(1)
                eve_log = model.forward_eve(y_bar, tgt[:, :-1], rnd)
                
                l_eve_ind = cross_entropy_loss(eve_log, tgt[:, 1:])
                l_eve_ind.backward()
                torch.nn.utils.clip_grad_norm_(eve_params, max_norm=1.0)
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

            
            print(f"{epoch:>6}  {avg_bob:>8.4f}  {avg_eve:>8.4f}  "f"{avg_key:>8.4f}  {new_lam:>6.2f}")
                
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
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=2.5e-4)
    parser.add_argument("--snr",          type=float, default=12.0)
    parser.add_argument("--seq_len",      type=int,   default=20)
    args = parser.parse_args()
    train(args)
