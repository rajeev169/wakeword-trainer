"""
Shivangi wake-word trainer.

Turns generated voice clips into "shivangi.onnx", a tiny model that runs on top of the
open-source openWakeWord speech features (melspectrogram.onnx + embedding_model.onnx).
The features here are computed exactly like the Android app computes them.

Usage:
  python train.py --pos data/positive --adv data/adversarial --neg data/negative --out output
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly

SR = 16000
CLIP = 32000          # 2.0 s windows
WINDOW = 16           # embeddings the model looks at (1.28 s)
HERE = Path(__file__).parent

rng = np.random.default_rng(7)
random.seed(7)
torch.manual_seed(7)


# ---------------- audio helpers ----------------

def load(path):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a


def trim(a, thr=0.02):
    idx = np.where(np.abs(a) > thr * max(1e-6, np.abs(a).max()))[0]
    return a if len(idx) == 0 else a[max(0, idx[0] - 800): idx[-1] + 800]


def noise(n, kind):
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    f = np.fft.rfft(w)
    k = np.arange(len(f)) + 1.0
    f = f / (np.sqrt(k) if kind == "pink" else k)
    out = np.fft.irfft(f, n).astype(np.float32)
    return out / (np.abs(out).max() + 1e-6)


def mix(sig, bg, snr_db):
    ps = np.mean(sig ** 2) + 1e-9
    pb = np.mean(bg ** 2) + 1e-9
    return sig + bg * np.sqrt(ps / (pb * 10 ** (snr_db / 10)))


def place_at_end(word, babble):
    """Word ends 0–0.25 s before the end of a 2 s window, like when the app hears it."""
    out = np.zeros(CLIP, np.float32)
    word = word[-(CLIP - 1600):]
    end = CLIP - int(rng.uniform(0.0, 0.25) * SR)
    start = max(0, end - len(word))
    out[start:start + len(word)] = word[: end - start]
    return add_background(out, babble)


def add_background(x, babble):
    r = rng.random()
    if r < 0.30:
        x = mix(x, noise(len(x), rng.choice(["white", "pink", "brown"])), rng.uniform(5, 30))
    elif r < 0.55 and babble:
        b = random_crop(random.choice(babble), len(x))
        x = mix(x, b, rng.uniform(8, 20))
    x = x * rng.uniform(0.3, 1.0) / (np.abs(x).max() + 1e-6)
    return x.astype(np.float32)


def random_crop(a, n):
    if len(a) >= n:
        s = rng.integers(0, len(a) - n + 1)
        return a[s:s + n]
    out = np.zeros(n, np.float32)
    s = rng.integers(0, n - len(a) + 1)
    out[s:s + len(a)] = a
    return out


# ---------------- features (same as the app) ----------------

class Features:
    def __init__(self, models_dir):
        o = ort.SessionOptions()
        o.intra_op_num_threads = os.cpu_count() or 2
        self.mel = ort.InferenceSession(str(Path(models_dir) / "melspectrogram.onnx"), o)
        self.emb = ort.InferenceSession(str(Path(models_dir) / "embedding_model.onnx"), o)

    def window(self, clip):
        """2 s float audio (-1..1) -> (16, 96) features ending at the clip end."""
        x = (clip * 32767).astype(np.float32)[None, :]
        m = self.mel.run(None, {self.mel.get_inputs()[0].name: x})[0]
        m = np.squeeze(m) / 10.0 + 2.0                      # (frames, 32)
        starts = list(range(len(m) - 76, -1, -8))[:WINDOW][::-1]
        if len(starts) < WINDOW:
            return None
        batch = np.stack([m[s:s + 76] for s in starts])[:, :, :, None].astype(np.float32)
        e = self.emb.run(None, {self.emb.get_inputs()[0].name: batch})[0]
        return e.reshape(WINDOW, 96)


# ---------------- model ----------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW * 96, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.f(x)


def wavs(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)) if d and os.path.isdir(d) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--adv", default="")
    ap.add_argument("--neg", default="")
    ap.add_argument("--models", default=str(HERE / "base_models"))
    ap.add_argument("--out", default="output")
    ap.add_argument("--aug", type=int, default=3, help="augmented copies per positive clip")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    pos_files, adv_files, neg_files = wavs(a.pos), wavs(a.adv), wavs(a.neg)
    print(f"positive clips: {len(pos_files)}, adversarial: {len(adv_files)}, negative speech: {len(neg_files)}")
    if len(pos_files) < 100:
        raise SystemExit("Not enough positive clips were generated")

    feats = Features(a.models)
    neg_audio = [load(f) for f in neg_files]
    X, y = [], []

    def add(clip, label):
        w = feats.window(clip)
        if w is not None:
            X.append(w); y.append(label)

    for i, f in enumerate(pos_files):
        word = trim(load(f))
        for _ in range(a.aug):
            add(place_at_end(word, neg_audio), 1.0)
        if i % 200 == 0:
            print(f"  positives {i}/{len(pos_files)}")
    for f in adv_files:
        word = trim(load(f))
        for _ in range(2):
            add(place_at_end(word, neg_audio), 0.0)
    for a_ in neg_audio:
        for _ in range(max(1, len(a_) // CLIP)):
            add(add_background(random_crop(a_, CLIP), neg_audio), 0.0)
    n_noise = max(300, len(pos_files) // 2)
    for _ in range(n_noise):
        add(add_background(noise(CLIP, rng.choice(["white", "pink", "brown"])) * rng.uniform(0.0, 0.3), neg_audio), 0.0)
    # Speech with the word earlier in the window (not at the end) also counts as "not now"
    for f in rng.choice(pos_files, size=min(len(pos_files), 400), replace=False):
        word = trim(load(f))
        out = np.zeros(CLIP, np.float32)
        s = int(rng.uniform(0, 0.25) * SR)
        out[s:s + len(word)] = word[: CLIP - s]
        tail = random_crop(random.choice(neg_audio), CLIP // 2) if neg_audio else noise(CLIP // 2, "pink") * 0.05
        out[CLIP // 2:] += tail[: CLIP // 2]
        add(add_background(out, neg_audio), 0.0)

    X = torch.tensor(np.stack(X)); y = torch.tensor(np.array(y, np.float32))[:, None]
    print(f"training windows: {len(y)} (positive {int(y.sum())})")
    idx = torch.randperm(len(y)); n_val = len(y) // 7
    va, tr = idx[:n_val], idx[n_val:]
    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    pos_w = (len(tr) - y[tr].sum()) / max(1.0, float(y[tr].sum()))
    for ep in range(a.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for b in range(0, len(perm), 256):
            j = perm[b:b + 256]
            p = model(X[j])
            w = torch.where(y[j] > 0.5, pos_w.clamp(max=10.0), torch.tensor(1.0))
            loss = nn.functional.binary_cross_entropy(p, y[j], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(X[va])
        yp, yv = (pv > 0.5).float(), y[va]
        recall = float((yp * yv).sum() / max(1.0, float(yv.sum())))
        fp = float((yp * (1 - yv)).sum() / max(1.0, float((1 - yv).sum())))
        print(f"epoch {ep + 1:02d}  loss {loss.item():.4f}  recall {recall:.3f}  false-positive rate {fp:.4f}")

    os.makedirs(a.out, exist_ok=True)
    model.eval()
    torch.onnx.export(model, torch.zeros(1, WINDOW, 96), os.path.join(a.out, "shivangi.onnx"),
                      input_names=["input"], output_names=["score"], dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
                      opset_version=13, dynamo=False)
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump({"windows": len(y), "positives": int(y.sum()), "val_recall": recall, "val_false_positive_rate": fp}, fh, indent=1)
    # Check the exported file works the way the app runs it
    s = ort.InferenceSession(os.path.join(a.out, "shivangi.onnx"))
    out = s.run(None, {"input": X[:1].numpy()})[0]
    print("exported shivangi.onnx, output shape", out.shape)


if __name__ == "__main__":
    main()
