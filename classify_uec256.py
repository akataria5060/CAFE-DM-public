# coding: utf-8



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy, json, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, Swin_B_Weights, resnet50, swin_b

DEVICE     = "cuda"
SEED       = 42
UEC_ROOT   = Path("/root/data/UECFood256/UECFOOD256")
ALIGN_CKPT = "/workspace/align/keep/align_train_20k_seed42.pth"
CKPT_DIR   = Path("/workspace/uec_arms"); CKPT_DIR.mkdir(exist_ok=True)

EPOCHS, BATCH, LR = 30, 32, 1e-4
WARMUP, WD, MIXUP, SMOOTH, EMA_DECAY, TTA_VIEWS = 3, 1e-4, 0.2, 0.1, 0.999, 8

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

free, total = torch.cuda.mem_get_info()
print(f"{free/1e9:.1f} GB free of {total/1e9:.1f}")




def build_instances(root):
    root = Path(root)
    cats = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()],
                  key=lambda d: int(d.name))
    c2i = {d.name: i for i, d in enumerate(cats)}
    out = []
    for d in cats:
        bb = d/"bb_info.txt"
        if not bb.exists(): continue
        for line in bb.read_text().splitlines()[1:]:
            p = line.split()
            if len(p) != 5: continue
            img_id, x1, y1, x2, y2 = p
            f = d/f"{img_id}.jpg"
            box = (int(x1), int(y1), int(x2), int(y2))
            if f.exists() and box[2] > box[0] and box[3] > box[1]:
                out.append((f, box, c2i[d.name], img_id))
    return out, len(cats)

def group_split(inst, seed=SEED):
    rng = random.Random(seed)
    by_stem = defaultdict(list)
    for x in inst: by_stem[x[3]].append(x)
    stems_by_cat = defaultdict(list)
    for stem, items in by_stem.items():
        stems_by_cat[min(i[2] for i in items)].append(stem)
    tr, va, te = [], [], []
    for cat in sorted(stems_by_cat):
        s = stems_by_cat[cat][:]; rng.shuffle(s)
        a, b = int(.70*len(s)), int(.85*len(s))
        for x in s[:a]: tr += by_stem[x]
        for x in s[a:b]: va += by_stem[x]
        for x in s[b:]: te += by_stem[x]
    assert not ({i[3] for i in tr} & {i[3] for i in te}), "leakage across split"
    return tr, va, te

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
train_tf = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3,0.3,0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD),
    transforms.RandomErasing(p=0.25)])
eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
tta_tf  = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

class UEC(Dataset):
    def __init__(self, inst, tf): self.inst, self.tf = inst, tf
    def __len__(self): return len(self.inst)
    def __getitem__(self, i):
        f, box, y, _ = self.inst[i]
        return self.tf(Image.open(f).convert("RGB").crop(box)), y

seed_all()
inst, N_CLASSES = build_instances(UEC_ROOT)
a, b, c = group_split(inst)
print(f"{len(inst)} instances -> {len(a)} train / {len(b)} val / {len(c)} test, "
      f"{N_CLASSES} classes")

mk = lambda d, s: DataLoader(d, batch_size=BATCH, shuffle=s,
                             num_workers=12, pin_memory=True)
tr_dl  = mk(UEC(a, train_tf), True)
va_dl  = mk(UEC(b, eval_tf), False)
te_dl  = mk(UEC(c, eval_tf), False)
tta_dl = mk(UEC(c, tta_tf), False)




def build_model(n_classes, kind):
    if kind == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_classes))
        return m
    m = swin_b(weights="IMAGENET1K_V1" if kind == "swin_imagenet" else None)
    dim = m.head.in_features
    m.head = nn.Identity()
    if kind == "swin_align":
        sd = torch.load(ALIGN_CKPT, map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd))
        ext = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        missing, _ = m.load_state_dict(ext, strict=False)
        loaded = len(m.state_dict()) - len(missing)
        print(f"  alignment init: {loaded}/{len(m.state_dict())} tensors")
        assert loaded > 0.8*len(m.state_dict()), "prefix mismatch"
    m.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_classes))
    return m

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m.detach(), alpha=1-self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)

def mixup(x, y, alpha=MIXUP):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); t1 = t5 = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        _, pred = model(x).topk(5, 1, True, True)
        cc = pred.eq(y.view(-1,1))
        t1 += cc[:,0].sum().item(); t5 += cc.any(1).sum().item(); n += y.size(0)
    return 100*t1/n, 100*t5/n

@torch.no_grad()
def evaluate_tta(model, base, tta, n_views=TTA_VIEWS):
    model.eval(); probs = labels = None
    for v in range(n_views):
        loader = base if v == 0 else tta
        ps, ls = [], []
        for x, y in loader:
            ps.append(F.softmax(model(x.to(DEVICE, non_blocking=True)), 1).cpu()); ls.append(y)
        vp = torch.cat(ps)
        probs = vp if probs is None else probs + vp
        if labels is None: labels = torch.cat(ls)
    _, pred = probs.topk(5, 1, True, True)
    cc = pred.eq(labels.view(-1,1))
    return 100*cc[:,0].sum().item()/len(labels), 100*cc.any(1).sum().item()/len(labels)




def train_arm(model, tag):
    model = model.to(DEVICE)
    ema = EMA(model)
    crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.SequentialLR(opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, WARMUP),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS-WARMUP)],
        milestones=[WARMUP])
    scaler = torch.cuda.amp.GradScaler()
    best, hist = -1.0, []
    ck, res = CKPT_DIR/f"best_{tag}.pth", CKPT_DIR/f"resume_{tag}.pt"
    start_ep = 0

    if res.exists():
        st = torch.load(res, map_location=DEVICE)
        model.load_state_dict(st["model"]); ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        best, start_ep, hist = st["best"], st["epoch"], st["hist"]
        print(f"  resumed at epoch {start_ep}", flush=True)

    t0 = time.time()
    for ep in range(start_ep, EPOCHS):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            xm, ya, yb, lam = mixup(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(xm)
                loss = lam*crit(out, ya) + (1-lam)*crit(out, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ema.update(model)
        sched.step()
        r1, r5 = evaluate(model, va_dl)
        e1, e5 = evaluate(ema.shadow, va_dl)
        hist.append({"epoch": ep+1, "raw_top1": r1, "ema_top1": e1})
        if max(r1, e1) > best:
            best = max(r1, e1)
            torch.save(ema.shadow.state_dict() if e1 >= r1 else model.state_dict(), ck)
        torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best": best,
                    "epoch": ep+1, "hist": hist}, res)
        print(f"[{tag}] ep {ep+1:2d}  raw {r1:.2f}/{r5:.2f}  ema {e1:.2f}/{e5:.2f}  "
              f"best {best:.2f}  {(time.time()-t0)/60:.0f}m", flush=True)

    model.load_state_dict(torch.load(ck))
    return model

results = {}
for kind in ("resnet50", "swin_imagenet", "swin_align"):
    print(f"\n=== {kind} ===", flush=True)
    seed_all()
    m = train_arm(build_model(N_CLASSES, kind), f"uec_{kind}")
    t1, t5 = evaluate(m, te_dl)
    a1, a5 = evaluate_tta(m, te_dl, tta_dl)
    results[kind] = dict(top1=t1, top5=t5, top1_tta=a1, top5_tta=a5)
    print(f"{kind}: no-TTA {t1:.2f}/{t5:.2f}   8-view {a1:.2f}/{a5:.2f}", flush=True)
    json.dump(results, open(CKPT_DIR/"uec_results.json", "w"), indent=2)
    del m; torch.cuda.empty_cache()

print(f"\n{'arm':16s} {'Top-1':>7} {'Top-5':>7} {'Top-1 TTA':>10} {'Top-5 TTA':>10}")
for k, v in results.items():
    print(f"{k:16s} {v['top1']:7.2f} {v['top5']:7.2f} {v['top1_tta']:10.2f} {v['top5_tta']:10.2f}")




import json

r = json.load(open("/workspace/RESULTS.json"))

r["uec_food256"] = {
    "protocol": "bounding-box crops from bb_info.txt; 70/15/15 GROUP split by "
                "source photograph (no photo spans train/test); 30 epochs, "
                "batch 32, AdamW 1e-4, 3-epoch warmup + cosine, mixup 0.2, "
                "label smoothing 0.1, EMA 0.999, 8-view TTA, seed 42",
    "n_instances": len(inst),
    "n_train": len(a), "n_val": len(b), "n_test": len(c),
    "n_classes": N_CLASSES,
    "resnet50":      {"top1": 79.41, "top5": 94.81, "top1_tta": 81.14, "top5_tta": 95.36},
    "swin_imagenet": {"top1": 84.13, "top5": 96.90, "top1_tta": 85.56, "top5_tta": 97.39},
    "swin_align":    {"top1": 85.03, "top5": 97.21, "top1_tta": 86.13, "top5_tta": 97.51},
    "alignment_gain_top1": 0.90,
    "alignment_gain_top1_tta": 0.57,
    "note": "Prior work on this dataset (WISeR 83.15, JDNet 84.00, RAFA-Net 91.56) "
            "uses the dataset's five-fold cross-validation protocol, not this split. "
            "Our three arms are an internal comparison only.",
}

json.dump(r, open("/workspace/RESULTS.json", "w"), indent=2)
print(json.dumps(r["uec_food256"], indent=2))






