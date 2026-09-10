# coding: utf-8



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import swin_b

DEVICE     = "cuda"
SEED       = 42
FOOD101    = Path("/workspace/food101/food-101")
ALIGN_CKPT = "/workspace/align/keep/align_train_20k_seed42.pth"
CKPT_DIR   = Path("/workspace/food101_arms"); CKPT_DIR.mkdir(exist_ok=True)

EPOCHS, BATCH, LR = 30, 32, 1e-4
WARMUP, WD, MIXUP, SMOOTH, EMA_DECAY, TTA_VIEWS = 3, 1e-4, 0.2, 0.1, 0.999, 8

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

free, total = torch.cuda.mem_get_info()
print(f"{torch.__version__}   {free/1e9:.1f} GB free of {total/1e9:.1f}")




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

class Food101(Dataset):
    def __init__(self, split, tf):
        classes = sorted((FOOD101/"meta"/"classes.txt").read_text().split())
        self.classes = classes
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.items = [(FOOD101/"images"/f"{l}.jpg", self.c2i[l.split("/")[0]])
                      for l in (FOOD101/"meta"/f"{split}.txt").read_text().split()]
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y

seed_all()
tr_full = Food101("train", train_tf)
CLASSES = tr_full.classes
n_val = int(0.1 * len(tr_full))
g = torch.Generator().manual_seed(SEED)
tr, va = torch.utils.data.random_split(tr_full, [len(tr_full)-n_val, n_val], generator=g)
te, te_tta = Food101("test", eval_tf), Food101("test", tta_tf)

mk = lambda d, s: DataLoader(d, batch_size=BATCH, shuffle=s,
                             num_workers=12, pin_memory=True)
tr_dl, va_dl, te_dl, tta_dl = mk(tr,True), mk(va,False), mk(te,False), mk(te_tta,False)
print(len(tr), len(va), len(te), len(CLASSES))




def build_swin(n_classes, init):
    m = swin_b(weights="IMAGENET1K_V1" if init == "imagenet" else None)
    dim = m.head.in_features
    m.head = nn.Identity()
    if init == "align":
        sd = torch.load(ALIGN_CKPT, map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd))
        ext = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        missing, _ = m.load_state_dict(ext, strict=False)
        loaded = len(m.state_dict()) - len(missing)
        print(f"alignment init: {loaded}/{len(m.state_dict())} tensors")
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
        c = pred.eq(y.view(-1,1))
        t1 += c[:,0].sum().item(); t5 += c.any(1).sum().item(); n += y.size(0)
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
    c = pred.eq(labels.view(-1,1))
    return 100*c[:,0].sum().item()/len(labels), 100*c.any(1).sum().item()/len(labels)




import json
RESUME = CKPT_DIR/"resume_imagenet.pt"
HIST   = CKPT_DIR/"hist_imagenet.json"

def train_arm_resumable(model, tag, resume_path, hist_path):
    model = model.to(DEVICE)
    ema = EMA(model)
    crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.SequentialLR(opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, WARMUP),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS-WARMUP)],
        milestones=[WARMUP])
    scaler = torch.cuda.amp.GradScaler()
    best, start_ep, hist = -1.0, 0, []
    ck = CKPT_DIR/f"best_{tag}.pth"

    if resume_path.exists():
        st = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(st["model"]); ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        best, start_ep, hist = st["best"], st["epoch"], st["hist"]
        print(f"resumed at epoch {start_ep}, best {best:.2f}", flush=True)

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
        hist.append({"epoch": ep+1, "raw_top1": r1, "raw_top5": r5,
                     "ema_top1": e1, "ema_top5": e5})
        if max(r1, e1) > best:
            best = max(r1, e1)
            torch.save(ema.shadow.state_dict() if e1 >= r1 else model.state_dict(), ck)
        torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best": best,
                    "epoch": ep+1, "hist": hist}, resume_path)
        json.dump(hist, open(hist_path, "w"))
        print(f"[{tag}] ep {ep+1:2d}  raw {r1:.2f}/{r5:.2f}  ema {e1:.2f}/{e5:.2f}  "
              f"best {best:.2f}  {(time.time()-t0)/60:.0f}m", flush=True)

    model.load_state_dict(torch.load(ck))
    return model

seed_all()
m_in = train_arm_resumable(build_swin(101, "imagenet"), "food101_imagenet", RESUME, HIST)
t1, t5 = evaluate(m_in, te_dl)
a1, a5 = evaluate_tta(m_in, te_dl, tta_dl)
print(f"\nIMAGENET INIT: no-TTA {t1:.2f}/{t5:.2f}   8-view {a1:.2f}/{a5:.2f}")




import json
r = json.load(open("/workspace/RESULTS.json"))
r["food101"] = {
    "protocol": "official 75750/25250 split, 30 epochs, batch 32, AdamW 1e-4, "
                "3-epoch warmup + cosine, mixup 0.2, label smoothing 0.1, "
                "EMA 0.999, 8-view TTA, seed 42",
    "imagenet_init":  {"top1": 91.48, "top5": 98.63, "top1_tta": 92.16, "top5_tta": 98.85},
    "alignment_init": {"top1": 91.97, "top5": 98.68, "top1_tta": 92.66, "top5_tta": 98.80},
    "rdfgm_reference": {"top1": 91.01, "top5": 98.33},
    "alignment_gain_top1": 0.49, "alignment_gain_top1_tta": 0.50,
}
json.dump(r, open("/workspace/RESULTS.json", "w"), indent=2)
print("saved")






