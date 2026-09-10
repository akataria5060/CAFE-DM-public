"""
Confusion-aware multimodal alignment on Recipe1M.

Stage 1 of the framework: learns a shared 256-d space between food images
(Swin-B) and ingredient text (BERT) under four losses --

    L_total = L_clip + 1.0*L_ing + 0.5*L_bh + 0.5*L_swap

and produces (a) the encoder that later initialises the Food-101 and
UEC-256 classifiers, and (b) the confusion gap, which is the paper's
central measurement.

CHECKPOINTS GO TO /workspace (NFS, persistent). Never /data -- that is
container-local and is how the previous checkpoints were lost.

Usage:
    python align_recipe1m.py --smoke      # val split only, 200 steps
    python align_recipe1m.py              # full run on train
    python align_recipe1m.py --eval-only --ckpt /workspace/align/best.pth
"""

import argparse, json, os, random, re, time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Swin_B_Weights, swin_b
from transformers import AutoModel, AutoTokenizer

# ==================================================================
# CONFIG
# ==================================================================

DATA = Path("/data/recipe1m")
CKPT_DIR = Path("/workspace/align")          # persistent
EMBED_DIM = 256
N_INGREDIENTS = 1047                          # matches RD-FGM's vocabulary size
MAX_TOKENS = 64
BATCH = 32
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_STEPS = 20_000
SAVE_EVERY = 1_000
LOG_EVERY = 50
SWAP_MARGIN = 0.2
TRIPLET_MARGIN = 0.2
LAMBDA_ING, LAMBDA_BH, LAMBDA_SWAP = 1.0, 0.5, 0.5
TEXT_MODEL = "bert-base-uncased"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFUSION_PAIRS = [("potato", "pineapple"), ("onion", "garlic"),
                   ("lemon", "lime"), ("oregano", "basil")]

# Compound forms that name a different ingredient. Excluded from pair
# membership so the swapped text is a genuine hard negative. Stated in
# the paper alongside the confusion gap protocol.
EXCLUDE = {
    "potato":    ["sweet potato", "potato chip", "potato soup"],
    "pineapple": ["pineapple juice"],
    "onion":     ["onion powder", "onion soup"],
    "garlic":    ["garlic powder", "garlic salt"],
    "lemon":     ["lemonade", "lemon pepper", "lemon jello"],
    "lime":      ["limeade", "lime jello"],
    "oregano":   [],
    "basil":     [],
}

GAP_SAMPLES_PER_PAIR = 2_000   # fixed, seeded; reported as n in the paper


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ==================================================================
# ANNOTATIONS
# ==================================================================

def _pat(term):
    return re.compile(rf"\b{re.escape(term)}(e?s)?\b")


def has_ingredient(ings, term):
    """True if any valid ingredient names `term` and is not an excluded form."""
    p = _pat(term)
    for i in ings:
        if p.search(i) and not any(x in i for x in EXCLUDE.get(term, [])):
            return True
    return False


def load_annotations():
    """Returns records keyed by recipe id, plus the ingredient vocabulary."""
    print("loading annotations ...", flush=True)
    det = json.load(open(DATA / "det_ingrs.json"))
    layer1 = json.load(open(DATA / "layer1.json"))
    layer2 = json.load(open(DATA / "layer2.json"))

    ings_by_id = {}
    for r in det:
        valid = [v["text"].lower().strip()
                 for v, ok in zip(r["ingredients"], r["valid"]) if ok]
        if valid:
            ings_by_id[r["id"]] = valid

    imgs_by_id = {r["id"]: [im["id"] for im in r["images"]] for r in layer2}
    part_by_id = {r["id"]: r["partition"] for r in layer1}

    # Vocabulary: the N_INGREDIENTS most frequent valid ingredient strings.
    freq = Counter(i for v in ings_by_id.values() for i in v)
    vocab = [w for w, _ in freq.most_common(N_INGREDIENTS)]
    vocab_idx = {w: k for k, w in enumerate(vocab)}

    records = defaultdict(list)
    for rid, ings in ings_by_id.items():
        if rid in imgs_by_id and rid in part_by_id and imgs_by_id[rid]:
            records[part_by_id[rid]].append((rid, ings, imgs_by_id[rid]))

    for p in ("train", "val", "test"):
        print(f"  {p}: {len(records[p])} recipes")
    print(f"  vocabulary: {len(vocab)} ingredients")
    return records, vocab_idx


def image_path(split, image_id):
    """Recipe1M nests images by the first four characters of the filename."""
    c = image_id[:4]
    return DATA / split / c[0] / c[1] / c[2] / c[3] / image_id


# ==================================================================
# DATASET
# ==================================================================

TRAIN_TF = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class Recipe1M(Dataset):
    """Yields image, positive ingredient text, swap-negative text, and a
    multi-hot ingredient vector.

    The negative is built by substituting one member of a confusion pair
    for the other in the joined ingredient string, but ONLY when the
    recipe contains exactly one member of that pair. If it contains both,
    the swapped text would still be true of the image -- a false negative,
    not a hard one. Recipes with no applicable pair fall back to a random
    ingredient substitution from the vocabulary.
    """

    def __init__(self, records, split, vocab_idx, tf, tokenizer):
        self.records = records
        self.split = split
        self.vocab_idx = vocab_idx
        self.vocab = list(vocab_idx)
        self.tf = tf
        self.tok = tokenizer

    def __len__(self):
        return len(self.records)

    def _swap(self, ings, rng):
        """Return (negative ingredient list, pair_used or None)."""
        applicable = []
        for a, b in CONFUSION_PAIRS:
            ha, hb = has_ingredient(ings, a), has_ingredient(ings, b)
            if ha != hb:                       # exclusivity
                applicable.append((a, b) if ha else (b, a))
        if applicable:
            src, dst = applicable[rng.randrange(len(applicable))]
            p = _pat(src)
            return [p.sub(dst, i) for i in ings], (src, dst)
        # fallback: replace one ingredient with a random vocabulary entry
        neg = list(ings)
        k = rng.randrange(len(neg))
        neg[k] = self.vocab[rng.randrange(len(self.vocab))]
        return neg, None

    def __getitem__(self, i):
        rid, ings, image_ids = self.records[i]
        rng = random.Random(hash((rid, i)) & 0xFFFFFFFF)

        img_id = image_ids[rng.randrange(len(image_ids))]
        path = image_path(self.split, img_id)
        try:
            img = self.tf(Image.open(path).convert("RGB"))
        except Exception:
            img = torch.zeros(3, 224, 224)     # missing file: skip gracefully

        neg_ings, _ = self._swap(ings, rng)
        pos_text = ", ".join(ings)
        neg_text = ", ".join(neg_ings)

        multi_hot = torch.zeros(len(self.vocab_idx))
        for x in ings:
            if x in self.vocab_idx:
                multi_hot[self.vocab_idx[x]] = 1.0

        return img, pos_text, neg_text, multi_hot


def collate(batch, tok):
    imgs, pos, neg, mh = zip(*batch)
    enc = lambda ts: tok(list(ts), padding=True, truncation=True,
                         max_length=MAX_TOKENS, return_tensors="pt")
    return torch.stack(imgs), enc(pos), enc(neg), torch.stack(mh)


# ==================================================================
# MODEL
# ==================================================================

class AlignmentModel(nn.Module):
    def __init__(self, n_ingredients, text_model=TEXT_MODEL, dim=EMBED_DIM):
        super().__init__()
        self.visual = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        vis_dim = self.visual.head.in_features
        self.visual.head = nn.Identity()

        self.text = AutoModel.from_pretrained(text_model)
        txt_dim = self.text.config.hidden_size

        self.proj_img = nn.Linear(vis_dim, dim)
        self.proj_txt = nn.Linear(txt_dim, dim)
        self.ing_head = nn.Linear(dim, n_ingredients)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode_image(self, x):
        return F.normalize(self.proj_img(self.visual(x)), dim=-1)

    def encode_text(self, enc):
        out = self.text(**enc).last_hidden_state           # B, T, H
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean pool
        return F.normalize(self.proj_txt(pooled), dim=-1)


# ==================================================================
# LOSSES
# ==================================================================

def clip_loss(zi, zt, scale):
    logits = scale.exp().clamp(max=100) * zi @ zt.t()
    tgt = torch.arange(len(zi), device=zi.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def swap_loss(zi, zpos, zneg, margin=SWAP_MARGIN):
    """Penalise whenever the confusable text sits closer to the image."""
    return F.relu(margin + (zi * zneg).sum(-1) - (zi * zpos).sum(-1)).mean()


def batch_hard_triplet(zi, zt, margin=TRIPLET_MARGIN):
    """Recipe1M has no category labels, so the positive is each image's own
    paired text and the negative is the most similar non-matching text in
    the batch. This is the batch-hard variant under one-positive-per-anchor.
    """
    sim = zi @ zt.t()
    pos = sim.diag()
    off = sim - torch.eye(len(zi), device=zi.device) * 1e4
    hardest_neg = off.max(dim=1).values
    return F.relu(margin + hardest_neg - pos).mean()


# ==================================================================
# CONFUSION GAP
# ==================================================================

@torch.no_grad()
def confusion_gap(model, records, split, tok, n_per_pair=GAP_SAMPLES_PER_PAIR,
                  return_raw=False):
    """Gap = sim(img, correct text) - sim(img, confusable text), per pair.

    Evaluation set: recipes in `split` containing exactly one member of the
    pair, after excluding the compound forms in EXCLUDE, sampled with a
    fixed seed.
    """
    model.eval()
    rng = random.Random(SEED)
    results = {}
    raw = {}

    for a, b in CONFUSION_PAIRS:
        pool = []
        for rid, ings, imgs in records:
            ha, hb = has_ingredient(ings, a), has_ingredient(ings, b)
            if ha != hb:
                pool.append((rid, ings, imgs, a if ha else b, b if ha else a))
        rng.shuffle(pool)
        pool = pool[:n_per_pair]

        gaps = []
        for k in range(0, len(pool), BATCH):
            chunk = pool[k:k + BATCH]
            imgs, pos, neg = [], [], []
            for rid, ings, image_ids, src, dst in chunk:
                p = image_path(split, image_ids[0])
                try:
                    imgs.append(EVAL_TF(Image.open(p).convert("RGB")))
                except Exception:
                    continue
                pat = _pat(src)
                pos.append(", ".join(ings))
                neg.append(", ".join(pat.sub(dst, i) for i in ings))
            if not imgs:
                continue

            x = torch.stack(imgs).to(DEVICE)
            enc = lambda ts: {k2: v.to(DEVICE) for k2, v in
                              tok(ts, padding=True, truncation=True,
                                  max_length=MAX_TOKENS,
                                  return_tensors="pt").items()}
            zi = model.encode_image(x)
            zp = model.encode_text(enc(pos))
            zn = model.encode_text(enc(neg))
            gaps.extend(((zi * zp).sum(-1) - (zi * zn).sum(-1)).cpu().tolist())

        gaps = np.array(gaps)
        raw[f"{a}/{b}"] = gaps
        results[f"{a}/{b}"] = dict(gap=float(gaps.mean()),
                                   positive_rate=float((gaps > 0).mean()),
                                   n=len(gaps))

    all_gaps = [v["gap"] for v in results.values()]
    results["mean"] = dict(gap=float(np.mean(all_gaps)),
                           positive_rate=float(np.mean(
                               [v["positive_rate"] for v in results.values()])),
                           n=sum(v["n"] for v in results.values()))
    model.train()
    return (results, raw) if return_raw else results


def print_gap(res):
    print("\n  confusion gap")
    for k, v in res.items():
        print(f"    {k:22s} {v['gap']:+.4f}   pos {100*v['positive_rate']:5.1f}%   n={v['n']}")


# ==================================================================
# TRAIN
# ==================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="train on val for 200 steps to verify the pipeline")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    seed_all()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    records, vocab_idx = load_annotations()
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)

    split = "val" if args.smoke else "train"
    steps = args.steps

    model = AlignmentModel(len(vocab_idx)).to(DEVICE)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"loaded {args.ckpt}")

    if args.eval_only:
        print_gap(confusion_gap(model, records["val"], "val", tok))
        return

    ds = Recipe1M(records[split], split, vocab_idx, TRAIN_TF, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=12,
                    pin_memory=True, drop_last=True,
                    collate_fn=lambda b: collate(b, tok))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.cuda.amp.GradScaler()

    step, t0, running = 0, time.time(), Counter()
    print(f"\ntraining on {split}: {len(ds)} recipes, {steps} steps\n")

    while step < steps:
        for img, pos, neg, mh in dl:
            if step >= steps:
                break
            img, mh = img.to(DEVICE, non_blocking=True), mh.to(DEVICE)
            pos = {k: v.to(DEVICE) for k, v in pos.items()}
            neg = {k: v.to(DEVICE) for k, v in neg.items()}

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                zi = model.encode_image(img)
                zp = model.encode_text(pos)
                zn = model.encode_text(neg)

                l_clip = clip_loss(zi, zp, model.logit_scale)
                l_ing = F.binary_cross_entropy_with_logits(model.ing_head(zi), mh)
                l_bh = batch_hard_triplet(zi, zp)
                l_swap = swap_loss(zi, zp, zn)
                loss = l_clip + LAMBDA_ING * l_ing + LAMBDA_BH * l_bh + LAMBDA_SWAP * l_swap

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()

            for k, v in dict(loss=loss, clip=l_clip, ing=l_ing,
                             bh=l_bh, swap=l_swap).items():
                running[k] += v.item()
            step += 1

            if step % LOG_EVERY == 0:
                n = LOG_EVERY
                print(f"step {step:6d}/{steps}  "
                      + "  ".join(f"{k}={running[k]/n:.4f}" for k in
                                  ("loss", "clip", "ing", "bh", "swap"))
                      + f"   {(time.time()-t0)/60:.1f}m", flush=True)
                running.clear()

            if step % SAVE_EVERY == 0 or step == steps:
                p = CKPT_DIR / f"align_step{step}.pth"
                torch.save(model.state_dict(), p)
                torch.save(model.state_dict(), CKPT_DIR / "align_latest.pth")
                print(f"  saved {p}", flush=True)

    print_gap(confusion_gap(model, records["val"], "val", tok))
    torch.save(model.state_dict(), CKPT_DIR / "align_final.pth")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
