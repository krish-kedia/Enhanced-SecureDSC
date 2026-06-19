"""
Enhanced SecureDSC — Secure Deep Semantic Communication System
B.Tech Project Implementation
=============================================================
Enhancements over base paper (Shi et al., IEEE Comm. Letters 2025):
  1. CSI-based dynamic key generation (replaces random session keys)
  2. Adaptive lambda scheduler    (replaces fixed λ=6 hyperparameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def generate_square_subsequent_mask(sz, device='cpu'):
    """Generate a causal mask preventing attention to future positions.
    Returns a (sz, sz) float mask where -inf blocks future tokens."""
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# ─────────────────────────────────────────────────────────────────
# 1. POSITIONAL ENCODING  (standard Transformer component)
# ─────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                       # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────
# 2. SEMANTIC ENCODER  — 4× Transformer encoder layers
# ─────────────────────────────────────────────────────────────────
class SemanticEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers)
        self.d_model   = d_model

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, mask=src_mask,
                            src_key_padding_mask=src_key_padding_mask)



# ─────────────────────────────────────────────────────────────────
# 4. ENCRYPTOR  — concatenates features + key, Transformer encoder
# ─────────────────────────────────────────────────────────────────
class Encryptor(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj     = nn.Linear(d_model * 2, d_model)   # fuse features + key
        enc_layer     = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers)
        self.mask_gen = nn.Linear(d_model, d_model)        # consecutive mask

    def forward(self, features, key_emb):
        # key_emb: (B, L_k, d_model) — broadcast to match feature length
        key_expanded = key_emb.mean(dim=1, keepdim=True).expand_as(features)
        x = torch.cat([features, key_expanded], dim=-1)
        x = self.proj(x)
        x = self.encoder(x)
        # Consecutive mask: zero out excess dims for dimensional consistency
        mask = torch.sigmoid(self.mask_gen(x))
        return x * mask


# ─────────────────────────────────────────────────────────────────
# 5. CHANNEL ENCODER / DECODER  — dense layers
# ─────────────────────────────────────────────────────────────────
class ChannelEncoder(nn.Module):
    def __init__(self, d_model: int = 128, channel_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, channel_dim)
        )

    def forward(self, x):
        return self.net(x)


class ChannelDecoder(nn.Module):
    def __init__(self, channel_dim: int = 16, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channel_dim, 256), nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# 6. DECRYPTOR  — Transformer decoder
# ─────────────────────────────────────────────────────────────────
class Decryptor(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj     = nn.Linear(d_model * 2, d_model)
        dec_layer     = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers)

    def forward(self, ciphertext, key_emb, tgt):
        key_expanded = key_emb.mean(dim=1, keepdim=True).expand_as(ciphertext)
        memory       = self.proj(torch.cat([ciphertext, key_expanded], dim=-1))
        return self.decoder(tgt, memory)


# ─────────────────────────────────────────────────────────────────
# 7. SEMANTIC DECODER  — Transformer decoder + prediction
# ─────────────────────────────────────────────────────────────────
class SemanticDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        dec_layer      = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.decoder   = nn.TransformerDecoder(dec_layer, num_layers)
        self.predict   = nn.Linear(d_model, vocab_size)
        self.d_model   = d_model

    def forward(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)
        # Auto-generate causal mask if none provided
        if tgt_mask is None:
            tgt_mask = generate_square_subsequent_mask(tgt.size(1), tgt.device)
        out     = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return F.log_softmax(self.predict(out), dim=-1)


# ─────────────────────────────────────────────────────────────────
# 8. ★ ENHANCEMENT 1 — CSI-BASED KEY GENERATOR
#    Replaces random session keys with physically derived keys
#    from the channel state information (h_A ≈ h_B by reciprocity)
# ─────────────────────────────────────────────────────────────────
class CSIKeyGenerator(nn.Module):
    """
    Maps a complex-valued channel estimate h ∈ C^N  →  key vector k ∈ R^key_dim
    via a small MLP. Training uses a consistency loss MSE(k_A, k_B) to force
    Alice and Bob's independent estimates to produce matching keys.

    Physical basis: channel reciprocity theorem guarantees h_Alice→Bob ≈ h_Bob→Alice
    over coherence time. Eve, at a different physical location, sees a different
    channel response and cannot reproduce the key.
    """
    def __init__(self, csi_dim: int = 32, hidden_dim: int = 128,
                 key_dim: int = 64, quantize: bool = True):
        super().__init__()
        # Input: real + imaginary parts concatenated → 2 * csi_dim
        self.net = nn.Sequential(
            nn.Linear(csi_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, key_dim),      nn.Tanh()
        )
        self.quantize = quantize
        self.key_dim  = key_dim

    def forward(self, h_complex):
        """
        Args:
            h_complex: (B, csi_dim) complex tensor — channel estimate
        Returns:
            key: (B, key_dim) float tensor — session key for this batch
        """
        h_real  = torch.real(h_complex).float()
        h_imag  = torch.imag(h_complex).float()
        h_input = torch.cat([h_real, h_imag], dim=-1)
        key     = self.net(h_input)
        if self.quantize and not self.training:
            # Hard quantise to {-1, +1} during inference for bit-level keys
            key = torch.sign(key)
        return key

    @staticmethod
    def consistency_loss(k_alice, k_bob):
        """
        L_key = MSE(k_A, k_B)
        Forces both sides to converge to the same key despite CSI estimation noise.
        """
        return F.mse_loss(k_alice, k_bob)

    @staticmethod
    def simulate_csi(batch_size: int, csi_dim: int = 32,
                     snr_db: float = 12.0, device="cpu"):
        """
        Utility: generates a pair of correlated CSI estimates (Alice, Bob)
        using Rayleigh fading + AWGN. Coherent channel → shared true h,
        independent noise → small estimation difference.
        """
        snr_linear = 10 ** (snr_db / 10)
        sigma      = 1.0 / math.sqrt(2 * snr_linear)
        h_true     = (torch.randn(batch_size, csi_dim)
                      + 1j * torch.randn(batch_size, csi_dim)).to(device) / math.sqrt(2)
        noise_a    = sigma * (torch.randn_like(h_true.real)
                              + 1j * torch.randn_like(h_true.imag))
        noise_b    = sigma * (torch.randn_like(h_true.real)
                              + 1j * torch.randn_like(h_true.imag))
        h_alice    = h_true + noise_a
        h_bob      = h_true + noise_b
        # Eve sees a completely different channel (different location)
        h_eve      = (torch.randn(batch_size, csi_dim)
                      + 1j * torch.randn(batch_size, csi_dim)).to(device) / math.sqrt(2)
        return h_alice, h_bob, h_eve


# ─────────────────────────────────────────────────────────────────
# 9. ★ ENHANCEMENT 2 — ADAPTIVE LAMBDA SCHEDULER
#    Replaces the fixed λ=6 hyperparameter in the adversarial loss
#    with a self-correcting control loop that adjusts λ each epoch
# ─────────────────────────────────────────────────────────────────
class AdaptiveLambdaScheduler:
    """
    Controls the adversarial balance term λ in:
        L_joint = L_Bob + |L_Eve − λ|

    Update rule (sign-based controller, similar to a bang-bang controller):
        λ(t+1) = λ(t) + η · sign(gap − target_gap)

    where gap = L_Eve − L_Bob measures the current separation.

    If gap < target_gap  →  Eve is too good  →  increase λ to push her loss up
    If gap > target_gap  →  gap is healthy   →  decrease λ to let Bob recover

    Clipping ensures λ stays in [lambda_min, lambda_max].
    """
    def __init__(self, lambda_init: float = 6.0,
                 target_gap:   float = 1.5,
                 step_size:    float = 0.1,
                 lambda_min:   float = 1.0,
                 lambda_max:   float = 12.0):
        self.lam        = lambda_init
        self.target_gap = target_gap
        self.eta        = step_size
        self.lam_min    = lambda_min
        self.lam_max    = lambda_max
        self.history    = []           # track λ over training

    def step(self, loss_bob: float, loss_eve: float):
        """Call once per epoch after computing average losses."""
        gap        = loss_eve - loss_bob
        direction  = 1 if gap < self.target_gap else -1
        self.lam   = float(
            max(self.lam_min,
                min(self.lam_max, self.lam + self.eta * direction))
        )
        self.history.append({
            "lambda": self.lam,
            "gap":    gap,
            "L_Bob":  loss_bob,
            "L_Eve":  loss_eve
        })
        return self.lam

    def get_lambda(self) -> float:
        return self.lam

    def joint_loss(self, loss_bob: torch.Tensor,
                   loss_eve: torch.Tensor) -> torch.Tensor:
        """
        L_joint = L_Bob + |L_Eve − λ(t)|
        Use inside the training loop (λ is treated as a constant per epoch).
        """
        return loss_bob + torch.abs(loss_eve - self.lam)


# ─────────────────────────────────────────────────────────────────
# 10. WIRELESS CHANNEL SIMULATION
# ─────────────────────────────────────────────────────────────────
class WirelessChannel(nn.Module):
    """AWGN + quasi-static Rayleigh fading."""
    def __init__(self, channel_type: str = "AWGN"):
        super().__init__()
        self.channel_type = channel_type

    def forward(self, x, snr_db: float = 12.0):
        snr_linear = 10 ** (snr_db / 10)
        signal_pwr = x.pow(2).mean()
        noise_std  = (signal_pwr / snr_linear).sqrt()
        noise      = torch.randn_like(x) * noise_std

        if self.channel_type == "Rayleigh":
            h = (torch.randn(x.size(0), 1, 1, device=x.device)
                 + torch.randn(x.size(0), 1, 1, device=x.device) * 1j).abs() / math.sqrt(2)
            h = h.real
            return h * x + noise
        return x + noise          # AWGN


# ─────────────────────────────────────────────────────────────────
# 11. FULL SECUREDSC SYSTEM
# ─────────────────────────────────────────────────────────────────
class SecureDSC(nn.Module):
    """
    Complete Enhanced SecureDSC pipeline.

    Alice TX path:
        m → SemanticEncoder → Encryptor(key_A) → ChannelEncoder → channel

    Bob RX path:
        channel → ChannelDecoder → Decryptor(key_B) → SemanticDecoder → m̂

    Eve RX path (adversary):
        channel → ChannelDecoder → Decryptor(random_key) → SemanticDecoder → m̄

    Keys:
        key_A = CSIKeyGenerator(h_A)   [Enhancement 1]
        key_B = CSIKeyGenerator(h_B)   [Enhancement 1, same weights]
        λ     = AdaptiveLambdaScheduler [Enhancement 2]
    """
    def __init__(self, vocab_size: int, d_model: int = 128,
                 channel_dim: int = 16, csi_dim: int = 32, key_dim: int = 64,
                 nhead: int = 8, num_layers: int = 4):
        super().__init__()

        # Legitimate pair (Alice + Bob share all weights except are fed different CSI)
        self.sem_encoder    = SemanticEncoder(vocab_size, d_model, nhead, num_layers)
        self.encryptor      = Encryptor(d_model, nhead, num_layers)
        self.ch_encoder     = ChannelEncoder(d_model, channel_dim)
        self.ch_decoder     = ChannelDecoder(channel_dim, d_model)
        self.decryptor      = Decryptor(d_model, nhead, num_layers)
        self.sem_decoder    = SemanticDecoder(vocab_size, d_model, nhead, num_layers)

        # ★ Enhancement 1: shared CSI key generator (Alice & Bob use same weights)
        self.csi_key_gen    = CSIKeyGenerator(csi_dim, d_model, key_dim)
        self.key_proj       = nn.Sequential(
            nn.Linear(key_dim, d_model), nn.Tanh()
        )

        # Eve has separate decoder weights (not jointly trained)
        self.eve_ch_dec     = ChannelDecoder(channel_dim, d_model)
        self.eve_decryptor  = Decryptor(d_model, nhead, num_layers)
        self.eve_sem_dec    = SemanticDecoder(vocab_size, d_model, nhead, num_layers)

        # Channel
        self.channel        = WirelessChannel("AWGN")

        # ★ Enhancement 2: adaptive λ scheduler (stateful, not a nn.Module)
        self.lambda_sched   = AdaptiveLambdaScheduler()

        self.d_model        = d_model

    def _key_embedding(self, key_vec):
        """Project flat key vector → sequence of length 1 for attention."""
        k = self.key_proj(key_vec)                 # (B, d_model)
        return k.unsqueeze(1)                      # (B, 1, d_model)

    def forward_alice(self, src, h_alice, snr_db=12.0):
        """Alice: encode → encrypt → transmit."""
        key_a   = self.csi_key_gen(h_alice)        # (B, key_dim)
        key_emb = self._key_embedding(key_a)       # (B, 1, d_model)
        feat    = self.sem_encoder(src)            # (B, L, d_model)
        cipher  = self.encryptor(feat, key_emb)   # (B, L, d_model)
        sym     = self.ch_encoder(cipher)          # (B, L, channel_dim)
        y_hat   = self.channel(sym, snr_db)        # (B, L, channel_dim)  ← Bob
        y_bar   = self.channel(sym, snr_db)        # (B, L, channel_dim)  ← Eve
        return y_hat, y_bar, key_a

    def forward_bob(self, y_hat, h_bob, tgt):
        """Bob: receive → decrypt → decode."""
        key_b    = self.csi_key_gen(h_bob)
        key_emb  = self._key_embedding(key_b)
        feat_hat = self.ch_decoder(y_hat)
        feat_dec = self.decryptor(feat_hat, key_emb, feat_hat)
        logits   = self.sem_decoder(feat_dec, tgt)
        return logits, key_b

    def forward_eve(self, y_bar, tgt, random_key_emb):
        """Eve: receive → attempt decrypt with wrong key."""
        feat_bar = self.eve_ch_dec(y_bar)
        feat_dec = self.eve_decryptor(feat_bar, random_key_emb, feat_bar)
        logits   = self.eve_sem_dec(feat_dec, tgt)
        return logits

    def forward(self, src, tgt, h_alice, h_bob, h_eve=None, snr_db=12.0):
        """Full forward pass: returns Bob logits, Eve logits, key pair."""
        y_hat, y_bar, key_a = self.forward_alice(src, h_alice, snr_db)

        bob_logits, key_b = self.forward_bob(y_hat, h_bob, tgt)

        # Eve uses random key (she can't access h_alice or h_bob)
        B   = src.size(0)
        rnd = torch.randn(B, self.d_model, device=src.device).unsqueeze(1)
        eve_logits = self.forward_eve(y_bar, tgt, rnd)

        return bob_logits, eve_logits, key_a, key_b

    @torch.no_grad()
    def autoregressive_decode(self, y_received, h_csi, max_len, sos_token):
        """
        Bob's inference-time decoding: generates tokens one at a time
        using only the received channel symbols and his CSI-derived key.
        No access to the ground truth target.
        """
        device  = y_received.device
        key     = self.csi_key_gen(h_csi)
        key_emb = self._key_embedding(key)
        feat    = self.ch_decoder(y_received)
        memory  = self.decryptor(feat, key_emb, feat)

        B = y_received.size(0)
        generated = torch.full((B, 1), sos_token, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits     = self.sem_decoder(memory, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated  = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def autoregressive_decode_eve(self, y_received, max_len, sos_token):
        """
        Eve's inference-time decoding: she intercepts the channel symbols
        but has no access to the CSI-derived key and must use a random one.
        """
        device = y_received.device
        B      = y_received.size(0)
        rnd    = torch.randn(B, self.d_model, device=device).unsqueeze(1)
        feat   = self.eve_ch_dec(y_received)
        memory = self.eve_decryptor(feat, rnd, feat)

        generated = torch.full((B, 1), sos_token, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits     = self.eve_sem_dec(memory, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated  = torch.cat([generated, next_token], dim=1)

        return generated
