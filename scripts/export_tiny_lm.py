"""Run sbintuitions/tiny-lm on one sentence and export what the hero animation draws.

The site does not load the model. This script runs it once, offline, and writes
the real tokens, embedding values, attention weights and next-token
probabilities to src/tinyLmData.ts.

Needs only torch. The SentencePiece model and the Llama forward pass are
re-implemented here so that neither `sentencepiece` nor `transformers` has to
be installed.

  python -I scripts/export_tiny_lm.py --model-dir <dir with pytorch_model.bin and spiece.model>

Model: https://huggingface.co/sbintuitions/tiny-lm (MIT License)
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import unicodedata
from pathlib import Path

import torch

TEXT = "AIと体験を静かにつなぐ"
N_LAYERS = 4
N_HEADS = 4
RMS_EPS = 1e-5
EMBED_DIMS_SHOWN = 8
TOP_K = 5


# ---------------------------------------------------------------- tokenizer


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _fields(buf: bytes):
    """Yield (field number, wire type, value) for one protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            size, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield number, wire, value


class Unigram:
    """Minimal SentencePiece unigram encoder (Viterbi over piece scores)."""

    def __init__(self, path: Path):
        self.pieces: list[str] = []
        self.scores: list[float] = []
        self.kinds: list[int] = []
        self.model_type = 1
        self.add_dummy_prefix = True
        for number, _, value in _fields(path.read_bytes()):
            if number == 1:
                piece, score, kind = "", 0.0, 1
                for n, _, v in _fields(value):
                    if n == 1:
                        piece = v.decode("utf-8")
                    elif n == 2:
                        score = struct.unpack("<f", v)[0]
                    elif n == 3:
                        kind = v
                self.pieces.append(piece)
                self.scores.append(score)
                self.kinds.append(kind)
            elif number == 2:
                for n, _, v in _fields(value):
                    if n == 3:
                        self.model_type = v
            elif number == 3:
                for n, _, v in _fields(value):
                    if n == 3:
                        self.add_dummy_prefix = bool(v)
        if self.model_type != 1:
            raise ValueError("only unigram SentencePiece models are supported")
        self.index = {p: i for i, p in enumerate(self.pieces) if self.kinds[i] in (1, 4)}
        self.unk_id = next(i for i, kind in enumerate(self.kinds) if kind == 2)
        self.max_len = max(len(p) for p in self.index)
        self.unk_score = min(self.scores) - 10.0

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFKC", text)
        text = " ".join(text.split()).replace(" ", "▁")
        if self.add_dummy_prefix:
            text = "▁" + text
        size = len(text)
        best = [-math.inf] * (size + 1)
        back: list[tuple[int, int]] = [(0, 0)] * (size + 1)
        best[0] = 0.0
        for start in range(size):
            if best[start] == -math.inf:
                continue
            matched = False
            for end in range(start + 1, min(size, start + self.max_len) + 1):
                piece_id = self.index.get(text[start:end])
                if piece_id is None:
                    continue
                matched = matched or end == start + 1
                score = best[start] + self.scores[piece_id]
                if score > best[end]:
                    best[end], back[end] = score, (start, piece_id)
            if not matched:
                score = best[start] + self.unk_score
                if score > best[start + 1]:
                    best[start + 1], back[start + 1] = score, (start, self.unk_id)
        ids: list[int] = []
        pos = size
        while pos > 0:
            pos, piece_id = back[pos]
            ids.append(piece_id)
        return ids[::-1]


# -------------------------------------------------------------------- model


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def forward(sd: dict[str, torch.Tensor], ids: list[int]):
    """Llama forward pass. Returns embeddings, attention per layer, last logits."""
    x = sd["model.embed_tokens.weight"][ids]
    embeddings = x.clone()
    seq, dim = x.shape
    head_dim = dim // N_HEADS

    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    angles = torch.cat((freqs, freqs), dim=-1)
    cos, sin = angles.cos(), angles.sin()
    mask = torch.full((seq, seq), -math.inf).triu(1)

    attentions = []
    for layer in range(N_LAYERS):
        prefix = f"model.layers.{layer}."
        h = rms_norm(x, sd[prefix + "input_layernorm.weight"])

        def split(name: str) -> torch.Tensor:
            projected = h @ sd[prefix + f"self_attn.{name}_proj.weight"].T
            return projected.view(seq, N_HEADS, head_dim).transpose(0, 1)

        q, k, v = split("q"), split("k"), split("v")
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        attention = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim) + mask, dim=-1)
        attentions.append(attention)
        mixed = (attention @ v).transpose(0, 1).reshape(seq, dim)
        x = x + mixed @ sd[prefix + "self_attn.o_proj.weight"].T

        h = rms_norm(x, sd[prefix + "post_attention_layernorm.weight"])
        gate = torch.nn.functional.silu(h @ sd[prefix + "mlp.gate_proj.weight"].T)
        up = h @ sd[prefix + "mlp.up_proj.weight"].T
        x = x + (gate * up) @ sd[prefix + "mlp.down_proj.weight"].T

    logits = rms_norm(x, sd["model.norm.weight"]) @ sd["lm_head.weight"].T
    return embeddings, torch.stack(attentions), logits


def load(model_dir: Path):
    sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    return {name: tensor.float() for name, tensor in sd.items()}


def top_tokens(tokenizer: Unigram, logits: torch.Tensor, k: int):
    probs = torch.softmax(logits, dim=-1)
    values, indices = probs.topk(k)
    return [(tokenizer.pieces[i], float(p)) for p, i in zip(values, indices.tolist())]


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--heads", default="0.2,1.3,2.1", help="heads to export as zero-based layer.head")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "src" / "tinyLmData.ts")
    parser.add_argument("--inspect", action="store_true", help="print tokens, sanity checks and head statistics only")
    args = parser.parse_args()

    tokenizer = Unigram(args.model_dir / "spiece.model")
    sd = load(args.model_dir)
    ids = tokenizer.encode(args.text)
    pieces = [tokenizer.pieces[i] for i in ids]

    with torch.no_grad():
        embeddings, attentions, logits = forward(sd, ids)

    if args.inspect:
        print("pieces", pieces, ids)
        for prompt in ["日本の首都は", "The capital of France is", "1 2 3 4 5", args.text]:
            prompt_ids = tokenizer.encode(prompt)
            with torch.no_grad():
                _, _, prompt_logits = forward(sd, prompt_ids)
            shown = ", ".join(f"{piece!r} {p:.3f}" for piece, p in top_tokens(tokenizer, prompt_logits[-1], 6))
            print(f"{prompt!r} [{len(prompt_ids)} tokens] -> {shown}")
        seq = len(ids)
        for layer in range(N_LAYERS):
            for head in range(N_HEADS):
                a = attentions[layer, head]
                rows = range(1, seq)
                first = sum(float(a[i, 0]) for i in rows) / (seq - 1)
                prev = sum(float(a[i, i - 1]) for i in rows) / (seq - 1)
                own = sum(float(a[i, i]) for i in rows) / (seq - 1)
                entropy = sum(float(-(a[i, : i + 1] * (a[i, : i + 1] + 1e-9).log()).sum()) for i in rows) / (seq - 1)
                print(f"L{layer + 1} H{head + 1}  first {first:.2f}  prev {prev:.2f}  self {own:.2f}  entropy {entropy:.2f}")
                for i in range(seq):
                    print("   ", " ".join(f"{float(x):.2f}" for x in a[i, : i + 1]))
        return

    chosen = [tuple(int(part) for part in item.split(".")) for item in args.heads.split(",") if item]
    scale = float(embeddings[:, :EMBED_DIMS_SHOWN].abs().max())
    data = {
        "model": "sbintuitions/tiny-lm",
        "text": args.text,
        "tokens": [piece.replace("▁", "") or "▁" for piece in pieces],
        "ids": ids,
        "embeddingScale": round(scale, 4),
        "embedding": [[round(float(v) / scale, 3) for v in row[:EMBED_DIMS_SHOWN]] for row in embeddings],
        "heads": [
            {
                "layer": layer + 1,
                "head": head + 1,
                "weights": [[round(float(v), 3) for v in row] for row in attentions[layer, head]],
            }
            for layer, head in chosen
        ],
        "next": [
            {"token": piece.replace("▁", " ").strip() or piece, "p": round(p, 4)}
            for piece, p in top_tokens(tokenizer, logits[-1], TOP_K)
        ],
    }
    def dump(value) -> str:
        return json.dumps(value, ensure_ascii=False)

    def rows(matrix) -> str:
        return "[\n" + "".join(f"      {dump(row)},\n" for row in matrix) + "    ]"

    heads = "".join(
        f"    {{\n      layer: {head['layer']},\n      head: {head['head']},\n"
        f"      weights: {rows(head['weights']).replace('      [', '        [').replace('    ]', '      ]')},\n    }},\n"
        for head in data["heads"]
    )
    body = (
        "{\n"
        f"  model: {dump(data['model'])},\n"
        f"  text: {dump(data['text'])},\n"
        f"  tokens: {dump(data['tokens'])},\n"
        f"  ids: {dump(data['ids'])},\n"
        f"  embeddingScale: {dump(data['embeddingScale'])},\n"
        f"  embedding: {rows(data['embedding']).replace('      [', '    [').replace('    ]', '  ]')},\n"
        f"  heads: [\n{heads}  ],\n"
        f"  next: {dump(data['next'])},\n"
        "}"
    )
    args.out.write_text(
        "// Generated by scripts/export_tiny_lm.py: real outputs of sbintuitions/tiny-lm.\n"
        "// Do not edit by hand.\n"
        f"export const tinyLm = {body} as const;\n",
        encoding="utf-8",
        newline="\n",
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
