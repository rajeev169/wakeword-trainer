"""Makes speech clips with Piper voices (pip package piper-tts==1.2.0).

  python generate.py --model voice.onnx --texts "Shivangi|Hey Shivangi" --out data/positive/en --count 400
  python generate.py --model voice.onnx --texts-file phrases_en.txt --out data/negative/en --count 800

Each clip uses a random speaker (multi-speaker voices), a random text and a random speed/tone,
so the trainer hears many different voices.
"""
import argparse, json, os, random, sys, wave
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", default="", help="texts separated by |")
    ap.add_argument("--texts-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, required=True)
    a = ap.parse_args()

    texts = [t.strip() for t in a.texts.split("|") if t.strip()]
    if a.texts_file:
        texts += [l.strip() for l in Path(a.texts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not texts:
        sys.exit("no texts given")

    from piper.voice import PiperVoice
    voice = PiperVoice.load(a.model, config_path=a.model + ".json")
    cfg = json.loads(Path(a.model + ".json").read_text(encoding="utf-8"))
    n_spk = int(cfg.get("num_speakers", 1) or 1)

    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(1234 + len(texts) + a.count)
    ok = bad = 0
    for i in range(a.count):
        text = texts[i % len(texts)]
        kw = dict(
            speaker_id=rnd.randrange(n_spk) if n_spk > 1 else None,
            length_scale=rnd.uniform(0.8, 1.3),
            noise_scale=rnd.uniform(0.4, 0.9),
            noise_w=rnd.uniform(0.4, 1.0),
        )
        path = os.path.join(a.out, f"{i:05d}.wav")
        try:
            with wave.open(path, "wb") as w:
                voice.synthesize(text, w, **kw)
            ok += 1
        except Exception as e:
            bad += 1
            if os.path.exists(path):
                os.remove(path)
            if bad <= 3:
                print("clip failed:", repr(e), file=sys.stderr)
    print(f"{a.out}: {ok} clips made, {bad} failed")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
