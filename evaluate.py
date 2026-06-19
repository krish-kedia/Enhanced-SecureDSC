"""
Evaluation module for Enhanced SecureDSC
==========================================
Computes BLEU scores for Bob and Eve across different SNR levels,
and evaluates key agreement rate (Enhancement 1 metric).

Uses real EuroParl sentences and autoregressive decoding to simulate
true inference-time performance over a wireless channel.

Usage:
    python evaluate.py --model_path securedsc_enhanced.pt --snr_range 0 3 6 9 12 15
"""

import argparse
import math
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from model import SecureDSC, CSIKeyGenerator


# -----------------------------------------------------------------
# EUROPARL EVAL DATASET
# -----------------------------------------------------------------
class EuroParlEvalDataset(Dataset):
    """
    Loads a small subset of EuroParl for evaluation.
    Uses an offset to avoid overlap with training data.
    """
    def __init__(self, tokenizer, seq_len=20, size=200, offset=2000000):
        print(f"[Eval] Loading EuroParl eval subset ({size} sentences)...")
        dataset = load_dataset("Helsinki-NLP/europarl", "en-fr", split="train")
        # Use sentences from a different region than training
        start = min(offset, len(dataset) - size)
        dataset = dataset.select(range(start, start + size))
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
        return src, text


# -----------------------------------------------------------------
# BLEU SCORE (n-gram precision, up to 4-gram)
# -----------------------------------------------------------------
def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def bleu_n(hypothesis, reference, n):
    """Single n-gram BLEU precision between two token lists."""
    hyp_counts = ngram_counts(hypothesis, n)
    ref_counts = ngram_counts(reference,  n)
    clipped    = sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
    total      = max(1, len(hypothesis) - n + 1)
    return clipped / total


def bleu_score(hypothesis, reference, max_n=4):
    """
    Sentence-level BLEU with uniform n-gram weights.
    Returns dict: {1: score_1gram, 2: score_2gram, ...}
    """
    bp     = min(1.0, math.exp(1 - len(reference) / max(1, len(hypothesis))))
    scores = {}
    for n in range(1, max_n + 1):
        p_n         = bleu_n(hypothesis, reference, n)
        scores[n]   = bp * math.exp(math.log(max(p_n, 1e-10)))
    return scores


# -----------------------------------------------------------------
# KEY AGREEMENT RATE  (Enhancement 1 specific metric)
# -----------------------------------------------------------------
def key_agreement_rate(model, n_trials=500, csi_dim=32, device="cpu"):
    """
    Measures the fraction of trials where k_A == k_B after quantisation.
    Should be high (>95%) for Enhancement 1 to be effective.
    """
    model.eval()
    agreed = 0
    with torch.no_grad():
        for snr_db in [6, 12, 15]:
            snr_agreed = 0
            for _ in range(n_trials):
                h_a, h_b, _ = CSIKeyGenerator.simulate_csi(
                    1, csi_dim=csi_dim, snr_db=snr_db, device=device
                )
                k_a = model.csi_key_gen(h_a)
                k_b = model.csi_key_gen(h_b)
                # Quantise to bits
                q_a = (k_a > 0).float()
                q_b = (k_b > 0).float()
                bit_agree = (q_a == q_b).float().mean().item()
                snr_agreed += bit_agree
            avg = snr_agreed / n_trials
            print(f"  Key agreement @ {snr_db:>2} dB SNR: {avg*100:.1f}%")
            agreed += avg
    return agreed / 3   # average across SNR levels


# -----------------------------------------------------------------
# MAIN EVALUATION
# -----------------------------------------------------------------
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    args.vocab_size = tokenizer.vocab_size
    sos_token = tokenizer.cls_token_id  # [CLS] token as start-of-sequence

    # Load evaluation dataset with real sentences
    eval_dataset = EuroParlEvalDataset(
        tokenizer, seq_len=args.seq_len, size=args.num_sentences
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Build model
    model = SecureDSC(
        vocab_size  = args.vocab_size,
        d_model     = 128,
        channel_dim = 16,
        csi_dim     = 32,
        key_dim     = 64,
        nhead       = 8,
        num_layers  = 4
    ).to(device)

    if args.model_path:
        state = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f"[Eval] Loaded model from {args.model_path}")
    else:
        print("[Eval] No checkpoint -- using random weights (demo mode)")

    model.eval()

    snr_list    = args.snr_range
    results     = {"snr": snr_list, "bob": {}, "eve": {}}
    for n in range(1, 5):
        results["bob"][n] = []
        results["eve"][n] = []

    print(f"\n{'SNR':>5}  {'Bob-1g':>7}  {'Bob-4g':>7}  {'Eve-1g':>7}  {'Eve-4g':>7}")
    print("-" * 45)

    with torch.no_grad():
        for snr_db in snr_list:
            bob_bleu_all = {n: [] for n in range(1, 5)}
            eve_bleu_all = {n: [] for n in range(1, 5)}

            for src, raw_text in eval_loader:
                src = src.to(device)
                B   = src.size(0)

                # Simulate CSI estimates
                h_a, h_b, h_e = CSIKeyGenerator.simulate_csi(
                    B, csi_dim=32, snr_db=snr_db, device=device
                )

                # Alice transmits
                y_bob, y_eve, _ = model.forward_alice(src, h_a, snr_db=snr_db)

                # Bob decodes autoregressively (no access to ground truth)
                bob_tokens = model.autoregressive_decode(
                    y_bob, h_b, max_len=args.seq_len, sos_token=sos_token
                )

                # Eve decodes autoregressively (random key)
                eve_tokens = model.autoregressive_decode_eve(
                    y_eve, max_len=args.seq_len, sos_token=sos_token
                )

                # Convert to word lists for BLEU
                ref_words = tokenizer.convert_ids_to_tokens(
                    src[0].tolist()
                )
                bob_words = tokenizer.convert_ids_to_tokens(
                    bob_tokens[0].tolist()
                )
                eve_words = tokenizer.convert_ids_to_tokens(
                    eve_tokens[0].tolist()
                )

                # Filter out special tokens ([PAD], [CLS], [SEP])
                special = {'[PAD]', '[CLS]', '[SEP]'}
                ref_words = [w for w in ref_words if w not in special]
                bob_words = [w for w in bob_words if w not in special]
                eve_words = [w for w in eve_words if w not in special]

                if len(ref_words) == 0:
                    continue

                b_scores = bleu_score(bob_words, ref_words)
                e_scores = bleu_score(eve_words, ref_words)

                for n in range(1, 5):
                    bob_bleu_all[n].append(b_scores[n])
                    eve_bleu_all[n].append(e_scores[n])

            bob_avg = {n: sum(v)/max(len(v),1) for n, v in bob_bleu_all.items()}
            eve_avg = {n: sum(v)/max(len(v),1) for n, v in eve_bleu_all.items()}

            for n in range(1, 5):
                results["bob"][n].append(round(bob_avg[n], 4))
                results["eve"][n].append(round(eve_avg[n], 4))

            print(f"{snr_db:>5}  {bob_avg[1]:>7.4f}  {bob_avg[4]:>7.4f}  "
                  f"{eve_avg[1]:>7.4f}  {eve_avg[4]:>7.4f}")

    # Key agreement rate (Enhancement 1)
    print("\n[Enhancement 1] Key Agreement Rate:")
    avg_kar = key_agreement_rate(model, n_trials=200, device=device)
    results["key_agreement_avg"] = round(avg_kar * 100, 1)
    print(f"  Average across SNRs: {avg_kar*100:.1f}%")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[Done] Saved -> eval_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str,   default=None)
    parser.add_argument("--seq_len",      type=int,   default=20)
    parser.add_argument("--num_sentences",type=int,   default=200)
    parser.add_argument("--snr_range",    type=float, nargs="+",
                        default=[0, 3, 6, 9, 12, 15])
    args = parser.parse_args()
    evaluate(args)
