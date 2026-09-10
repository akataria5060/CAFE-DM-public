# coding: utf-8



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import json, math, random, re, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel

DATA       = Path("/data/recipe1m")
ALIGN_CKPT = "/workspace/align/keep/align_train_20k_seed42.pth"
CKPT_DIR   = Path("/workspace/diffusion"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

RES        = 256
BATCH      = 32
ACCUM      = 15
LR         = 1e-4
MAX_STEPS  = 60_000
SAVE_EVERY = 2_000
LOG_EVERY  = 100
P_UNCOND   = 0.1
MAX_TOKENS = 64
TEXT_MODEL = "bert-base-uncased"
SEED       = 42
DEVICE     = "cuda"

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print(torch.__version__, torch.cuda.is_available())




print("visible GPUs:", torch.cuda.device_count())




def load_annotations():
    det    = json.load(open(DATA / "det_ingrs.json"))
    layer1 = json.load(open(DATA / "layer1.json"))
    layer2 = json.load(open(DATA / "layer2.json"))

    ings_by_id = {}
    for r in det:
        v = [x["text"].lower().strip()
             for x, ok in zip(r["ingredients"], r["valid"]) if ok]
        if v:
            ings_by_id[r["id"]] = v

    imgs_by_id = {r["id"]: [im["id"] for im in r["images"]] for r in layer2}
    part_by_id = {r["id"]: r["partition"] for r in layer1}

    recs = defaultdict(list)
    for rid, ings in ings_by_id.items():
        if rid in imgs_by_id and rid in part_by_id and imgs_by_id[rid]:
            recs[part_by_id[rid]].append((rid, ings, imgs_by_id[rid]))
    for p in ("train", "val", "test"):
        print(f"  {p}: {len(recs[p])}")
    return recs

def image_path(split, image_id):
    c = image_id[:4]
    return DATA / split / c[0] / c[1] / c[2] / c[3] / image_id

records = load_annotations()




class FrozenTextEncoder(nn.Module):
    def __init__(self, ckpt_path=ALIGN_CKPT, model_name=TEXT_MODEL):
        super().__init__()
        self.tok  = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd))
        extracted = {k[len("text."):]: v for k, v in sd.items() if k.startswith("text.")}
        if not extracted:
            raise KeyError("no 'text.' keys in checkpoint -- check ALIGN_CKPT")
        missing, unexpected = self.bert.load_state_dict(extracted, strict=False)
        print(f"text encoder: {len(self.bert.state_dict()) - len(missing)}"
              f"/{len(self.bert.state_dict())} tensors from alignment checkpoint")

        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, texts):
        enc = self.tok(texts, padding="max_length", truncation=True,
                       max_length=MAX_TOKENS, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        return self.bert(**enc).last_hidden_state          # (B, 64, 768)

text_encoder = FrozenTextEncoder().to(DEVICE)
CROSS_DIM = text_encoder.bert.config.hidden_size
print("cross-attention dim:", CROSS_DIM)




vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
vae.eval()
for p in vae.parameters():
    p.requires_grad_(False)
SCALE = vae.config.scaling_factor
print("latent channels:", vae.config.latent_channels, "scale:", SCALE)

img_tf = transforms.Compose([
    transforms.Resize(RES),
    transforms.CenterCrop(RES),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

class Recipe1MGen(Dataset):
    def __init__(self, recs, split, tf):
        self.recs, self.split, self.tf = recs, split, tf
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        rid, ings, image_ids = self.recs[i]
        p = image_path(self.split, image_ids[0])
        try:
            img = self.tf(Image.open(p).convert("RGB"))
        except Exception:
            img = torch.zeros(3, RES, RES)
        return img, ", ".join(ings)

def collate(batch):
    imgs, texts = zip(*batch)
    return torch.stack(imgs), list(texts)

train_ds = Recipe1MGen(records["train"], "train", img_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=12,
                      pin_memory=True, drop_last=True, collate_fn=collate)
print(len(train_ds), "training recipes, batch", BATCH)




unet = UNet2DConditionModel(
    sample_size=RES // 8,
    in_channels=4, out_channels=4,
    layers_per_block=2,
    block_out_channels=(192, 384, 576, 576),
    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                      "CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    cross_attention_dim=CROSS_DIM,
    attention_head_dim=8,
).to(DEVICE)

print(f"{sum(p.numel() for p in unet.parameters())/1e6:.0f}M parameters")

noise_sched = DDPMScheduler(num_train_timesteps=1000,
                            beta_schedule="squaredcos_cap_v2",
                            prediction_type="epsilon")

with torch.no_grad():
    NULL_EMB = text_encoder([""])[0]
print("null embedding:", tuple(NULL_EMB.shape))




torch.cuda.reset_peak_memory_stats()
opt    = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler()
unet.train()

t0 = time.time()
for i, (imgs, texts) in enumerate(train_dl):
    imgs = imgs.to(DEVICE, non_blocking=True)
    with torch.no_grad():
        latents = vae.encode(imgs).latent_dist.sample() * SCALE
        ctx = text_encoder(texts)
        drop = torch.rand(len(texts), device=DEVICE) < P_UNCOND
        ctx[drop] = NULL_EMB

    noise = torch.randn_like(latents)
    t = torch.randint(0, 1000, (latents.size(0),), device=DEVICE).long()
    noisy = noise_sched.add_noise(latents, noise, t)

    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        pred = unet(noisy, t, encoder_hidden_states=ctx).sample
        loss = F.mse_loss(pred.float(), noise.float())
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()

    if i % 5 == 0:
        print(f"step {i}  loss={loss.item():.4f}  "
              f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    if i >= 20:
        break

print(f"{(time.time()-t0)/21:.2f} s/step")




MICRO_STEPS = 90_000          # ~10 epochs at batch 32
SAVE_EVERY  = 5_000
LOG_EVERY   = 200

opt    = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler()
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MICRO_STEPS // ACCUM)

start = 0
latest = CKPT_DIR / "unet_latest.pt"
if latest.exists():
    ck = torch.load(latest, map_location=DEVICE)
    unet.load_state_dict(ck["unet"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); start = ck["micro"]
    print(f"resumed at micro-step {start}")

micro, running, t0 = start, 0.0, time.time()
unet.train()
opt.zero_grad(set_to_none=True)

while micro < MICRO_STEPS:
    for imgs, texts in train_dl:
        if micro >= MICRO_STEPS:
            break
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            latents = vae.encode(imgs).latent_dist.sample() * SCALE
            ctx = text_encoder(texts)
            drop = torch.rand(len(texts), device=DEVICE) < P_UNCOND
            ctx[drop] = NULL_EMB

        noise = torch.randn_like(latents)
        t = torch.randint(0, 1000, (latents.size(0),), device=DEVICE).long()
        noisy = noise_sched.add_noise(latents, noise, t)

        with torch.cuda.amp.autocast():
            pred = unet(noisy, t, encoder_hidden_states=ctx).sample
            loss = F.mse_loss(pred.float(), noise.float()) / ACCUM
        scaler.scale(loss).backward()

        running += loss.item() * ACCUM
        micro += 1

        if micro % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)

        if micro % LOG_EVERY == 0:
            print(f"micro {micro:6d}/{MICRO_STEPS}  opt {micro//ACCUM:5d}  "
                  f"loss={running/LOG_EVERY:.4f}  {(time.time()-t0)/60:.1f}m", flush=True)
            running = 0.0

        if micro % SAVE_EVERY == 0 or micro == MICRO_STEPS:
            torch.save({"unet": unet.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "micro": micro}, latest)
            torch.save(unet.state_dict(), CKPT_DIR / f"unet_micro{micro}.pt")
            print(f"  saved at {micro}", flush=True)

print("done")




@torch.no_grad()
def generate(prompts, guidance=3.5, steps=100, seed=42):
    unet.eval()
    ddim = DDIMScheduler.from_config(noise_sched.config)
    ddim.set_timesteps(steps, device=DEVICE)
    n = len(prompts)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    lat = torch.randn(n, 4, RES//8, RES//8, device=DEVICE, generator=g) * ddim.init_noise_sigma
    ctx = torch.cat([NULL_EMB.unsqueeze(0).expand(n, -1, -1), text_encoder(prompts)])
    for t in ddim.timesteps:
        inp = ddim.scale_model_input(torch.cat([lat]*2), t)
        with torch.cuda.amp.autocast():
            eps = unet(inp, t, encoder_hidden_states=ctx).sample
        e_u, e_c = eps.chunk(2)
        lat = ddim.step((e_u + guidance*(e_c - e_u)).float(), t, lat).prev_sample
    unet.train()
    return (vae.decode(lat / SCALE).sample / 2 + 0.5).clamp(0, 1)




import matplotlib.pyplot as plt

prompts = ["potato, beef, onion, salt, pepper",
           "pineapple, beef, onion, salt, pepper",
           "flour, sugar, egg, butter, chocolate chips",
           "lettuce, tomato, cucumber, olive oil, lemon",
           "spaghetti, tomato, garlic, basil, parmesan",
           "salmon, lemon, dill, butter, black pepper"]

out = generate(prompts, guidance=3.5, steps=100)

fig, ax = plt.subplots(1, len(prompts), figsize=(3.2*len(prompts), 3.6))
for a, im, p in zip(ax, out, prompts):
    a.imshow(im.permute(1, 2, 0).cpu().numpy()); a.axis("off")
    a.set_title("\n".join(p.split(", ")[:3]), fontsize=8)
plt.tight_layout(); plt.show()








from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.functional import multiscale_structural_similarity_index_measure as msssim

N_EVAL    = 2048      # state this in the paper; FID depends on it
DDIM_STEPS = 100
GUIDANCE   = 3.5

test_recs = records["test"][:N_EVAL]
print(len(test_recs), "test recipes")

real_tf = transforms.Compose([
    transforms.Resize(RES), transforms.CenterCrop(RES), transforms.ToTensor(),
])

def load_real(recs):
    out = []
    for rid, ings, image_ids in recs:
        try:
            out.append(real_tf(Image.open(image_path("test", image_ids[0])).convert("RGB")))
        except Exception:
            pass
    return torch.stack(out)

real = load_real(test_recs)
print("real:", real.shape)




gen_batches = []
bs = 16
t0 = time.time()
for i in range(0, len(test_recs), bs):
    chunk = test_recs[i:i+bs]
    prompts = [", ".join(ings) for _, ings, _ in chunk]
    imgs = generate(prompts, guidance=GUIDANCE, steps=DDIM_STEPS, seed=SEED + i)
    gen_batches.append(imgs.cpu())
    if (i // bs) % 10 == 0:
        done = i + len(chunk)
        print(f"{done}/{len(test_recs)}  {(time.time()-t0)/60:.1f}m", flush=True)

gen = torch.cat(gen_batches)[:len(real)]
torch.save(gen, "/workspace/diffusion/gen_test_2048.pt")
print("generated:", gen.shape)




import torchmetrics; print(torchmetrics.__version__)




from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
try:
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim
except ImportError:
    from torchmetrics.functional import multiscale_structural_similarity_index_measure as msssim

def to_uint8(x):
    return (x.clamp(0, 1) * 255).to(torch.uint8)

# FID
fid = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
for i in range(0, len(real), 64):
    fid.update(to_uint8(real[i:i+64]).to(DEVICE), real=True)
for i in range(0, len(gen), 64):
    fid.update(to_uint8(gen[i:i+64]).to(DEVICE), real=False)
fid_val = fid.compute().item()
del fid; torch.cuda.empty_cache()

# Inception Score
iscore = InceptionScore(normalize=False).to(DEVICE)
for i in range(0, len(gen), 64):
    iscore.update(to_uint8(gen[i:i+64]).to(DEVICE))
is_mean, is_std = [v.item() for v in iscore.compute()]
del iscore; torch.cuda.empty_cache()

# MS-SSIM across random pairs of generated images -- lower means more diverse
g = torch.Generator().manual_seed(SEED)
idx = torch.randperm(len(gen), generator=g)
half = len(gen) // 2
a, b = gen[idx[:half]], gen[idx[half:half*2]]
scores = []
for i in range(0, len(a), 32):
    scores.append(msssim(a[i:i+32].to(DEVICE), b[i:i+32].to(DEVICE),
                         data_range=1.0).item())
ms = float(np.mean(scores))

print(f"\nn={len(gen)}   {DDIM_STEPS} DDIM steps   guidance {GUIDANCE}")
print(f"FID      {fid_val:.2f}      (RD-FGM 82.45)")
print(f"IS       {is_mean:.2f} +/- {is_std:.2f}   (RD-FGM 16.88 +/- 0.05)")
print(f"MS-SSIM  {ms:.4f}      (RD-FGM 0.0547, lower = more diverse)")




N_EVAL = 10_000

test_recs = records["test"][:N_EVAL]
print(len(test_recs), "test recipes")

real = load_real(test_recs)
print("real:", real.shape)




gen_batches = []
bs = 16
t0 = time.time()
for i in range(0, len(test_recs), bs):
    chunk = test_recs[i:i+bs]
    prompts = [", ".join(ings) for _, ings, _ in chunk]
    imgs = generate(prompts, guidance=GUIDANCE, steps=DDIM_STEPS, seed=SEED + i)
    gen_batches.append(imgs.cpu())
    if (i // bs) % 25 == 0:
        done = i + len(chunk)
        el = (time.time() - t0) / 60
        print(f"{done}/{len(test_recs)}  {el:.1f}m  eta {el/max(done,1)*(len(test_recs)-done):.0f}m",
              flush=True)

gen = torch.cat(gen_batches)[:len(real)]
torch.save(gen, "/workspace/diffusion/gen_test_10k.pt")
print("generated:", gen.shape)




from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
try:
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim
except ImportError:
    from torchmetrics.functional import multiscale_structural_similarity_index_measure as msssim

def to_uint8(x):
    return (x.clamp(0, 1) * 255).to(torch.uint8)

# FID
fid = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
for i in range(0, len(real), 64):
    fid.update(to_uint8(real[i:i+64]).to(DEVICE), real=True)
for i in range(0, len(gen), 64):
    fid.update(to_uint8(gen[i:i+64]).to(DEVICE), real=False)
fid_val = fid.compute().item()
del fid; torch.cuda.empty_cache()

# Inception Score
iscore = InceptionScore(normalize=False).to(DEVICE)
for i in range(0, len(gen), 64):
    iscore.update(to_uint8(gen[i:i+64]).to(DEVICE))
is_mean, is_std = [v.item() for v in iscore.compute()]
del iscore; torch.cuda.empty_cache()

# MS-SSIM across random pairs of generated images -- lower means more diverse
g = torch.Generator().manual_seed(SEED)
idx = torch.randperm(len(gen), generator=g)
half = len(gen) // 2
a, b = gen[idx[:half]], gen[idx[half:half*2]]
scores = []
for i in range(0, len(a), 32):
    scores.append(msssim(a[i:i+32].to(DEVICE), b[i:i+32].to(DEVICE),
                         data_range=1.0).item())
ms = float(np.mean(scores))

print(f"\nn={len(gen)}   {DDIM_STEPS} DDIM steps   guidance {GUIDANCE}")
print(f"FID      {fid_val:.2f}      (RD-FGM 82.45)")
print(f"IS       {is_mean:.2f} +/- {is_std:.2f}   (RD-FGM 16.88 +/- 0.05)")
print(f"MS-SSIM  {ms:.4f}      (RD-FGM 0.0547, lower = more diverse)")




SWEEP_N = 2048
sweep_recs = records["test"][:SWEEP_N]
sweep_real = load_real(sweep_recs)
sweep_prompts = [", ".join(ings) for _, ings, _ in sweep_recs]
print("sweep real:", sweep_real.shape)

def metrics(real_t, gen_t):
    f = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    for i in range(0, len(real_t), 64):
        f.update(to_uint8(real_t[i:i+64]).to(DEVICE), real=True)
    for i in range(0, len(gen_t), 64):
        f.update(to_uint8(gen_t[i:i+64]).to(DEVICE), real=False)
    fv = f.compute().item(); del f; torch.cuda.empty_cache()

    s = InceptionScore(normalize=False).to(DEVICE)
    for i in range(0, len(gen_t), 64):
        s.update(to_uint8(gen_t[i:i+64]).to(DEVICE))
    im, istd = [v.item() for v in s.compute()]; del s; torch.cuda.empty_cache()

    gg = torch.Generator().manual_seed(SEED)
    ix = torch.randperm(len(gen_t), generator=gg); h = len(gen_t)//2
    A, B = gen_t[ix[:h]], gen_t[ix[h:h*2]]
    sc = [msssim(A[i:i+32].to(DEVICE), B[i:i+32].to(DEVICE), data_range=1.0).item()
          for i in range(0, len(A), 32)]
    return fv, im, istd, float(np.mean(sc))

results = {}
for w in [2.5, 3.5, 5.0, 7.5]:
    print(f"\n--- guidance {w} ---", flush=True)
    batches, t0, bs = [], time.time(), 16
    for i in range(0, len(sweep_recs), bs):
        batches.append(generate(sweep_prompts[i:i+bs], guidance=w,
                                steps=DDIM_STEPS, seed=SEED + i).cpu())
        if (i // bs) % 40 == 0:
            print(f"  {i}/{len(sweep_recs)}  {(time.time()-t0)/60:.1f}m", flush=True)
    gw = torch.cat(batches)[:len(sweep_real)]
    results[w] = metrics(sweep_real, gw)
    f, im, istd, ms = results[w]
    print(f"  w={w}  FID {f:.2f}   IS {im:.2f}+/-{istd:.2f}   MS-SSIM {ms:.4f}", flush=True)

print(f"\n{'w':>5} {'FID':>8} {'IS':>16} {'MS-SSIM':>9}   (n={SWEEP_N})")
for w, (f, im, istd, ms) in results.items():
    print(f"{w:>5} {f:>8.2f}   {im:>6.2f} +/- {istd:.2f}   {ms:>8.4f}")




GUIDANCE = 5.0
N_EVAL = 10_000
test_recs = records["test"][:N_EVAL]
real = load_real(test_recs)

gen_batches, bs, t0 = [], 16, time.time()
for i in range(0, len(test_recs), bs):
    prompts = [", ".join(ings) for _, ings, _ in test_recs[i:i+bs]]
    gen_batches.append(generate(prompts, guidance=GUIDANCE, steps=DDIM_STEPS,
                                seed=SEED + i).cpu())
    if (i // bs) % 50 == 0:
        el = (time.time()-t0)/60
        print(f"{i}/{len(test_recs)}  {el:.1f}m", flush=True)

gen = torch.cat(gen_batches)[:len(real)]
torch.save(gen, "/workspace/diffusion/gen_test_10k_w5.pt")
print(gen.shape)




from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
try:
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim
except ImportError:
    from torchmetrics.functional import multiscale_structural_similarity_index_measure as msssim

def to_uint8(x):
    return (x.clamp(0, 1) * 255).to(torch.uint8)

# FID
fid = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
for i in range(0, len(real), 64):
    fid.update(to_uint8(real[i:i+64]).to(DEVICE), real=True)
for i in range(0, len(gen), 64):
    fid.update(to_uint8(gen[i:i+64]).to(DEVICE), real=False)
fid_val = fid.compute().item()
del fid; torch.cuda.empty_cache()

# Inception Score
iscore = InceptionScore(normalize=False).to(DEVICE)
for i in range(0, len(gen), 64):
    iscore.update(to_uint8(gen[i:i+64]).to(DEVICE))
is_mean, is_std = [v.item() for v in iscore.compute()]
del iscore; torch.cuda.empty_cache()

# MS-SSIM across random pairs of generated images -- lower means more diverse
g = torch.Generator().manual_seed(SEED)
idx = torch.randperm(len(gen), generator=g)
half = len(gen) // 2
a, b = gen[idx[:half]], gen[idx[half:half*2]]
scores = []
for i in range(0, len(a), 32):
    scores.append(msssim(a[i:i+32].to(DEVICE), b[i:i+32].to(DEVICE),
                         data_range=1.0).item())
ms = float(np.mean(scores))

print(f"\nn={len(gen)}   {DDIM_STEPS} DDIM steps   guidance {GUIDANCE}")
print(f"FID      {fid_val:.2f}      (RD-FGM 82.45)")
print(f"IS       {is_mean:.2f} +/- {is_std:.2f}   (RD-FGM 16.88 +/- 0.05)")
print(f"MS-SSIM  {ms:.4f}      (RD-FGM 0.0547, lower = more diverse)")




import matplotlib.pyplot as plt

PAIRS = [("potato", "pineapple"), ("onion", "garlic"),
         ("lemon", "lime"), ("oregano", "basil")]
BASE = {"potato":    "{}, beef, onion, salt, black pepper",
        "pineapple": "{}, beef, onion, salt, black pepper",
        "onion":     "chicken, {}, olive oil, salt, thyme",
        "garlic":    "chicken, {}, olive oil, salt, thyme",
        "lemon":     "salmon fillet, {}, butter, parsley, salt",
        "lime":      "salmon fillet, {}, butter, parsley, salt",
        "oregano":   "tomato, mozzarella, {}, olive oil, bread",
        "basil":     "tomato, mozzarella, {}, olive oil, bread"}
N_SEEDS = 3

fig, axes = plt.subplots(len(PAIRS)*2, N_SEEDS, figsize=(3*N_SEEDS, 3*len(PAIRS)*2))
for pi, (a, b) in enumerate(PAIRS):
    for ri, ing in enumerate((a, b)):
        prompt = BASE[ing].format(ing)
        row = pi*2 + ri
        for s in range(N_SEEDS):
            img = generate([prompt], guidance=5.0, steps=100, seed=1000+s)[0]
            ax = axes[row, s]
            ax.imshow(img.permute(1,2,0).cpu().numpy()); ax.axis("off")
            if s == 0:
                ax.set_ylabel(ing, fontsize=11)
                ax.text(-0.1, 0.5, ing, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=12,
                        color="green" if ri == 0 else "red")
plt.tight_layout()
plt.savefig("/workspace/diffusion/confusion_pairs.png", dpi=150, bbox_inches="tight")
plt.show()




from torchvision.models import swin_b
swin = swin_b(weights=None)
sd = torch.load(ALIGN_CKPT, map_location="cpu")
sd = sd.get("state_dict", sd.get("model", sd))
swin.head = nn.Identity()
swin.load_state_dict({k[len("visual."):]: v for k, v in sd.items()
                      if k.startswith("visual.")}, strict=False)
proj_i = nn.Linear(1024, 256); proj_t = nn.Linear(768, 256)
proj_i.load_state_dict({"weight": sd["proj_img.weight"], "bias": sd["proj_img.bias"]})
proj_t.load_state_dict({"weight": sd["proj_txt.weight"], "bias": sd["proj_txt.bias"]})
swin, proj_i, proj_t = swin.to(DEVICE).eval(), proj_i.to(DEVICE).eval(), proj_t.to(DEVICE).eval()

norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

@torch.no_grad()
def gap_on_generated(a, b, n=64):
    pa, pb = BASE[a].format(a), BASE[b].format(b)
    gaps = []
    for src, dst, prompt in ((a, b, pa), (b, a, pb)):
        imgs = torch.cat([generate([prompt]*8, guidance=5.0, steps=100, seed=2000+k)
                          for k in range(n//8)])
        x = norm(F.interpolate(imgs, 224, mode="bilinear", align_corners=False))
        zi = F.normalize(proj_i(swin(x)), dim=-1)
        correct = BASE[src].format(src); swapped = BASE[src].format(dst)
        zt_c = F.normalize(proj_t(text_encoder([correct]*len(zi)).mean(1)), dim=-1)
        zt_s = F.normalize(proj_t(text_encoder([swapped]*len(zi)).mean(1)), dim=-1)
        gaps += ((zi*zt_c).sum(-1) - (zi*zt_s).sum(-1)).cpu().tolist()
    g = np.array(gaps)
    return g.mean(), (g > 0).mean(), len(g)

print(f"{'pair':22s} {'gap':>9} {'pos':>7} {'n':>5}")
allg = []
for a, b in PAIRS:
    m, p, n = gap_on_generated(a, b)
    allg.append(m)
    print(f"{a}/{b:14s} {m:+9.4f} {100*p:6.1f}% {n:5d}")
print(f"{'mean':22s} {np.mean(allg):+9.4f}")




from torchvision.models import swin_b
import re

def _pat(t):
    return re.compile(rf"\b{re.escape(t)}(e?s)?\b")

EXCLUDE = {"potato": ["sweet potato","potato chip","potato soup"],
           "pineapple": ["pineapple juice"],
           "onion": ["onion powder","onion soup"], "garlic": ["garlic powder","garlic salt"],
           "lemon": ["lemonade","lemon pepper","lemon jello"], "lime": ["limeade","lime jello"],
           "oregano": [], "basil": []}

def has_ing(ings, t):
    p = _pat(t)
    return any(p.search(i) and not any(x in i for x in EXCLUDE.get(t, [])) for i in ings)

def sample_prompts(term, other, n, seed=42):
    """Real test recipes containing `term` but not `other`."""
    pool = [ings for _, ings, _ in records["test"]
            if has_ing(ings, term) and not has_ing(ings, other)]
    random.Random(seed).shuffle(pool)
    return [", ".join(x) for x in pool[:n]]

@torch.no_grad()
def gap_generated(a, b, n_per_dir=256, bs=16, guidance=5.0, steps=100):
    gaps = []
    for src, dst in ((a, b), (b, a)):
        prompts = sample_prompts(src, dst, n_per_dir)
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i+bs]
            imgs = generate(chunk, guidance=guidance, steps=steps, seed=3000+i)
            x = norm(F.interpolate(imgs, 224, mode="bilinear", align_corners=False))
            zi = F.normalize(proj_i(swin(x)), dim=-1)
            swapped = [_pat(src).sub(dst, p) for p in chunk]
            zt_c = F.normalize(proj_t(text_encoder(chunk).mean(1)), dim=-1)
            zt_s = F.normalize(proj_t(text_encoder(swapped).mean(1)), dim=-1)
            gaps += ((zi*zt_c).sum(-1) - (zi*zt_s).sum(-1)).cpu().tolist()
    g = np.array(gaps)
    se = g.std() / np.sqrt(len(g))
    return g.mean(), se, (g > 0).mean(), len(g)

print(f"{'pair':24s} {'gap':>9} {'se':>7} {'pos':>7} {'n':>6}")
allg = []
for a, b in PAIRS:
    m, se, p, n = gap_generated(a, b)
    allg.append(m)
    print(f"{a+'/'+b:24s} {m:+9.4f} {se:7.4f} {100*p:6.1f}% {n:6d}", flush=True)
print(f"{'mean':24s} {np.mean(allg):+9.4f}")




import json, datetime

n_params = sum(p.numel() for p in unet.parameters()) / 1e6

results = {
    "date": str(datetime.date.today()),
    "alignment": {
        "checkpoint": "align_train_20k_seed42.pth",
        "steps": 20000, "batch": 32, "train_recipes": 281297,
        "confusion_gap_real": {
            "potato/pineapple": [0.2864, 0.943],
            "onion/garlic":     [0.3111, 0.920],
            "lemon/lime":       [0.0554, 0.749],
            "oregano/basil":    [0.0120, 0.624],
            "mean":             [0.1662, 0.809],
            "n_per_pair": 2000, "split": "val"},
        "pair_coverage_train": {"overall": 0.440, "onion/garlic": 0.260,
                                "lemon/lime": 0.162, "potato/pineapple": 0.075,
                                "oregano/basil": 0.068},
    },
    "diffusion": {
        "checkpoint": "unet_micro90000.pt",
        "micro_steps": 90000, "optimizer_steps": 6000, "batch": 32, "accum": 15,
        "effective_batch": 480, "hours": 13.7, "resolution": 256,
        "vae": "stabilityai/sd-vae-ft-mse",
        "params_M": round(n_params, 1),
        "headline": {"n": 10000, "ddim_steps": 100, "guidance": 5.0,
                     "FID": 62.14, "IS": [7.93, 0.15], "MS_SSIM": 0.0702},
        "rdfgm_reference": {"FID": 82.45, "IS": [16.88, 0.05], "MS_SSIM": 0.0547},
        "guidance_sweep_n2048": {
            "2.5": {"FID": 90.80, "IS": [6.27, 0.42], "MS_SSIM": 0.0789},
            "3.5": {"FID": 83.28, "IS": [6.82, 0.36], "MS_SSIM": 0.0749},
            "5.0": {"FID": 78.12, "IS": [7.55, 0.40], "MS_SSIM": 0.0722},
            "7.5": {"FID": 81.33, "IS": [8.27, 0.54], "MS_SSIM": 0.0713}},
        "confusion_gap_generated": {
            "potato/pineapple": [0.1789, 0.0062, 0.943],
            "onion/garlic":     [0.2762, 0.0091, 0.922],
            "lemon/lime":       [0.0304, 0.0033, 0.576],
            "oregano/basil":    [0.0112, 0.0012, 0.654],
            "mean": 0.1242, "n_per_pair": 512,
            "prompts": "real test recipes, exclusivity enforced"},
    },
}

with open("/workspace/RESULTS.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"saved, unet = {n_params:.1f}M params")




import matplotlib.pyplot as plt
from pathlib import Path
FIGS = Path("/workspace/figs"); FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "savefig.dpi": 300})




prompts = [
    "spaghetti, tomato, garlic, basil, parmesan cheese",
    "salmon fillet, lemon, butter, dill, black pepper",
    "romaine lettuce, tomato, cucumber, olive oil, feta cheese",
    "chicken breast, rice, curry powder, onion, coconut milk",
    "beef, potato, carrot, onion, beef broth",
    "flour, sugar, eggs, butter, vanilla extract",
    "bread, cheddar cheese, butter, ham",
    "shrimp, garlic, olive oil, parsley, white wine",
]
out = generate(prompts, guidance=5.0, steps=100, seed=7)
fig, ax = plt.subplots(2, 4, figsize=(12, 6.6))
for a, im, p in zip(ax.flat, out, prompts):
    a.imshow(im.permute(1,2,0).cpu().numpy()); a.axis("off")
    a.set_title(", ".join(p.split(", ")[:3]), fontsize=8)
plt.tight_layout(); plt.savefig(FIGS/"fig_samples.png"); plt.show()




PAIR_PROMPTS = {}
for a, b in PAIRS:
    PAIR_PROMPTS[a] = sample_prompts(a, b, 4, seed=11)
    PAIR_PROMPTS[b] = sample_prompts(b, a, 4, seed=11)

fig, axes = plt.subplots(8, 4, figsize=(12, 24.5))
for pi, (a, b) in enumerate(PAIRS):
    for ri, ing in enumerate((a, b)):
        row = pi*2 + ri
        imgs = generate(PAIR_PROMPTS[ing], guidance=5.0, steps=100, seed=500+row)
        for c in range(4):
            ax = axes[row, c]
            ax.imshow(imgs[c].permute(1,2,0).cpu().numpy())
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#1a7f37" if ri == 0 else "#c1121f"); s.set_linewidth(3)
            if c == 0:
                ax.set_ylabel(ing, fontsize=13, fontweight="bold",
                              color="#1a7f37" if ri == 0 else "#c1121f")
plt.tight_layout(); plt.savefig(FIGS/"fig_confusion_pairs.png"); plt.show()




real_gap = [0.2864, 0.3111, 0.0554, 0.0120]
gen_gap  = [0.1789, 0.2762, 0.0304, 0.0112]
gen_se   = [0.0062, 0.0091, 0.0033, 0.0012]
labels   = ["potato /\npineapple", "onion /\ngarlic", "lemon /\nlime", "oregano /\nbasil"]
x = np.arange(4); w = 0.36

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x-w/2, real_gap, w, label="Real images (val, n=2000/pair)", color="#2b6cb0")
ax.bar(x+w/2, gen_gap, w, yerr=gen_se, capsize=3,
       label="Generated images (n=512/pair)", color="#dd8452")
ax.axhline(0.1662, ls="--", c="#2b6cb0", lw=1, label="mean, real (+0.1662)")
ax.axhline(0.1242, ls="--", c="#dd8452", lw=1, label="mean, generated (+0.1242)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("confusion gap"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIGS/"fig_gap_bars.png"); plt.show()




w_ = [2.5, 3.5, 5.0, 7.5]
fid_ = [90.80, 83.28, 78.12, 81.33]
is_  = [6.27, 6.82, 7.55, 8.27]
is_e = [0.42, 0.36, 0.40, 0.54]
ms_  = [0.0789, 0.0749, 0.0722, 0.0713]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(w_, fid_, "o-", color="#2b6cb0"); ax[0].set_ylabel("FID $\\downarrow$")
ax[0].scatter([5.0], [78.12], s=140, facecolors="none", edgecolors="r", zorder=5)
ax[1].errorbar(w_, is_, yerr=is_e, fmt="o-", color="#dd8452"); ax[1].set_ylabel("IS $\\uparrow$")
ax[2].plot(w_, ms_, "o-", color="#55a868"); ax[2].set_ylabel("MS-SSIM $\\downarrow$")
for a in ax:
    a.set_xlabel("guidance scale $w$"); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIGS/"fig_guidance.png"); plt.show()




import re
rows = []
for line in open("/workspace/align_train.log"):
    m = re.match(r"step\s+(\d+)/\d+\s+loss=([\d.]+)\s+clip=([\d.]+)\s+"
                 r"ing=([\d.]+)\s+bh=([\d.]+)\s+swap=([\d.]+)", line)
    if m: rows.append([float(v) for v in m.groups()])
r = np.array(rows)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for i, (n, c) in enumerate(zip(["CLIP","ingredient BCE","batch-hard","swap"],
                               ["#2b6cb0","#dd8452","#55a868","#c44e52"]), start=2):
    ax[0].plot(r[:,0], r[:,i], label=n, lw=1, color=c)
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(r[:,0], r[:,5], color="#c44e52", lw=1.2)
ax[1].set_xlabel("step"); ax[1].set_ylabel("$\\mathcal{L}_{swap}$")
ax[1].set_title("confusion-aware term", fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIGS/"fig_align_loss.png"); plt.show()




n_show = 6
picks = records["test"][100:100+n_show]
prompts_rg = [", ".join(ings) for _, ings, _ in picks]

reals = []
for _, _, image_ids in picks:
    reals.append(real_tf(Image.open(image_path("test", image_ids[0])).convert("RGB")))
reals = torch.stack(reals)
gens = generate(prompts_rg, guidance=5.0, steps=100, seed=21)

fig, ax = plt.subplots(2, n_show, figsize=(2.6*n_show, 5.8))
for c in range(n_show):
    ax[0, c].imshow(reals[c].permute(1,2,0).numpy()); ax[0, c].axis("off")
    ax[1, c].imshow(gens[c].permute(1,2,0).cpu().numpy()); ax[1, c].axis("off")
    ax[0, c].set_title(", ".join(prompts_rg[c].split(", ")[:3]), fontsize=7)
ax[0, 0].text(-0.12, 0.5, "Real", transform=ax[0,0].transAxes, rotation=90,
              va="center", ha="right", fontsize=12, fontweight="bold")
ax[1, 0].text(-0.12, 0.5, "Generated", transform=ax[1,0].transAxes, rotation=90,
              va="center", ha="right", fontsize=12, fontweight="bold")
plt.tight_layout(); save("fig_real_vs_gen"); plt.show()




def save(name):
    plt.savefig(FIGS/f"{name}.png"); plt.savefig(FIGS/f"{name}.pdf")

n_show = 6
picks = records["test"][100:100+n_show]
prompts_rg = [", ".join(ings) for _, ings, _ in picks]

reals = []
for _, _, image_ids in picks:
    reals.append(real_tf(Image.open(image_path("test", image_ids[0])).convert("RGB")))
reals = torch.stack(reals)
gens = generate(prompts_rg, guidance=5.0, steps=100, seed=21)

fig, ax = plt.subplots(2, n_show, figsize=(2.6*n_show, 5.8))
for c in range(n_show):
    ax[0, c].imshow(reals[c].permute(1,2,0).numpy()); ax[0, c].axis("off")
    ax[1, c].imshow(gens[c].permute(1,2,0).cpu().numpy()); ax[1, c].axis("off")
    ax[0, c].set_title(", ".join(prompts_rg[c].split(", ")[:3]), fontsize=7)
ax[0, 0].text(-0.12, 0.5, "Real", transform=ax[0,0].transAxes, rotation=90,
              va="center", ha="right", fontsize=12, fontweight="bold")
ax[1, 0].text(-0.12, 0.5, "Generated", transform=ax[1,0].transAxes, rotation=90,
              va="center", ha="right", fontsize=12, fontweight="bold")
plt.tight_layout(); save("fig_real_vs_gen"); plt.show()








from sklearn.manifold import TSNE

N_PER = 150
emb, lab = [], []
for a, b in PAIRS:
    for ing, other in ((a, b), (b, a)):
        pool = [(ings, ids) for _, ings, ids in records["val"]
                if has_ing(ings, ing) and not has_ing(ings, other)][:N_PER]
        batch = []
        for ings, ids in pool:
            try:
                batch.append(real_tf(Image.open(image_path("val", ids[0])).convert("RGB")))
            except Exception:
                pass
        if not batch:
            continue
        x = torch.stack(batch)
        with torch.no_grad():
            for i in range(0, len(x), 32):
                xb = norm(F.interpolate(x[i:i+32].to(DEVICE), 224,
                                        mode="bilinear", align_corners=False))
                emb.append(F.normalize(proj_i(swin(xb)), dim=-1).cpu())
        lab += [ing] * len(x)
        print(ing, len(x), flush=True)

E = torch.cat(emb).numpy()
Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(E)

colors = {"potato":"#c44e52","pineapple":"#dd8452","onion":"#2b6cb0","garlic":"#64b5f6",
          "lemon":"#55a868","lime":"#a3d977","oregano":"#8172b3","basil":"#c5b0d5"}
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for pi, (a, b) in enumerate(PAIRS):
    for ing in (a, b):
        m = [i for i, l in enumerate(lab) if l == ing]
        ax[pi].scatter(Z[m,0], Z[m,1], s=9, alpha=0.7, c=colors[ing], label=ing)
    ax[pi].legend(fontsize=8, markerscale=1.6); ax[pi].set_xticks([]); ax[pi].set_yticks([])
    ax[pi].set_title(f"{a} vs {b}", fontsize=10)
plt.tight_layout(); save("fig_tsne_pairs.png".replace(".png","")); plt.show()




@torch.no_grad()
def gap_values(a, b, n_per_dir=256, bs=16):
    out = []
    for src, dst in ((a, b), (b, a)):
        pool = [(ings, ids) for _, ings, ids in records["val"]
                if has_ing(ings, src) and not has_ing(ings, dst)][:n_per_dir]
        for i in range(0, len(pool), bs):
            chunk = pool[i:i+bs]
            ims = []
            for ings, ids in chunk:
                try: ims.append(real_tf(Image.open(image_path("val", ids[0])).convert("RGB")))
                except Exception: pass
            if not ims: continue
            x = norm(F.interpolate(torch.stack(ims).to(DEVICE), 224,
                                   mode="bilinear", align_corners=False))
            zi = F.normalize(proj_i(swin(x)), dim=-1)
            corr = [", ".join(c[0]) for c in chunk][:len(zi)]
            swap = [_pat(src).sub(dst, t) for t in corr]
            zc = F.normalize(proj_t(text_encoder(corr).mean(1)), dim=-1)
            zs = F.normalize(proj_t(text_encoder(swap).mean(1)), dim=-1)
            out += ((zi*zc).sum(-1) - (zi*zs).sum(-1)).cpu().tolist()
    return np.array(out)

fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
for i, (a, b) in enumerate(PAIRS):
    g = gap_values(a, b)
    ax[i].hist(g, bins=40, color="#2b6cb0", alpha=0.8)
    ax[i].axvline(0, color="k", lw=1)
    ax[i].axvline(g.mean(), color="#c44e52", lw=2, ls="--")
    ax[i].set_title(f"{a} vs {b}\nmean {g.mean():+.3f}, {100*(g>0).mean():.1f}% > 0", fontsize=9)
    ax[i].set_xlabel("confusion gap")
ax[0].set_ylabel("count")
plt.tight_layout(); save("fig_gap_hist"); plt.show()




@torch.no_grad()
def gap_values(a, b, n_per_dir=1000, bs=32, seed=42):
    out = []
    for src, dst in ((a, b), (b, a)):
        pool = [(ings, ids) for _, ings, ids in records["val"]
                if has_ing(ings, src) and not has_ing(ings, dst)]
        random.Random(seed).shuffle(pool)
        pool = pool[:n_per_dir]
        for i in range(0, len(pool), bs):
            chunk = pool[i:i+bs]
            ims, keep = [], []
            for ings, ids in chunk:
                try:
                    ims.append(real_tf(Image.open(image_path("val", ids[0])).convert("RGB")))
                    keep.append(ings)
                except Exception:
                    pass
            if not ims:
                continue
            x = norm(F.interpolate(torch.stack(ims).to(DEVICE), 224,
                                   mode="bilinear", align_corners=False))
            zi = F.normalize(proj_i(swin(x)), dim=-1)
            corr = [", ".join(k) for k in keep]
            swap = [_pat(src).sub(dst, t) for t in corr]
            zc = F.normalize(proj_t(text_encoder(corr).mean(1)), dim=-1)
            zs = F.normalize(proj_t(text_encoder(swap).mean(1)), dim=-1)
            out += ((zi*zc).sum(-1) - (zi*zs).sum(-1)).cpu().tolist()
    return np.array(out)

fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
for i, (a, b) in enumerate(PAIRS):
    g = gap_values(a, b)
    ax[i].hist(g, bins=45, color="#2b6cb0", alpha=0.85)
    ax[i].axvline(0, color="k", lw=1)
    ax[i].axvline(g.mean(), color="#c44e52", lw=2, ls="--")
    ax[i].set_title(f"{a} vs {b}\nmean {g.mean():+.4f},  "
                    f"{100*(g>0).mean():.1f}% > 0,  n={len(g)}", fontsize=9)
    ax[i].set_xlabel("confusion gap")
    print(f"{a}/{b}: {g.mean():+.4f}  {100*(g>0).mean():.1f}%  n={len(g)}", flush=True)
ax[0].set_ylabel("count")
plt.tight_layout(); save("fig_gap_hist"); plt.show()




@torch.no_grad()
def enc_text_masked(ts):
    e = text_encoder.tok(ts, padding=True, truncation=True,
                         max_length=MAX_TOKENS, return_tensors="pt")
    e = {k: v.to(DEVICE) for k, v in e.items()}
    h = text_encoder.bert(**e).last_hidden_state
    m = e["attention_mask"].unsqueeze(-1).float()
    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
    return F.normalize(proj_t(pooled), dim=-1)

@torch.no_grad()
def gap_values(a, b, n_per_dir=1000, bs=32, seed=42):
    out = []
    for src, dst in ((a, b), (b, a)):
        pool = [(ings, ids) for _, ings, ids in records["val"]
                if has_ing(ings, src) and not has_ing(ings, dst)]
        random.Random(seed).shuffle(pool)
        pool = pool[:n_per_dir]
        for i in range(0, len(pool), bs):
            chunk = pool[i:i+bs]
            ims, keep = [], []
            for ings, ids in chunk:
                try:
                    ims.append(real_tf(Image.open(image_path("val", ids[0])).convert("RGB")))
                    keep.append(ings)
                except Exception:
                    pass
            if not ims:
                continue
            x = norm(F.interpolate(torch.stack(ims).to(DEVICE), 224,
                                   mode="bilinear", align_corners=False))
            zi = F.normalize(proj_i(swin(x)), dim=-1)
            # substitute per ingredient, matching align_recipe1m.py
            corr = [", ".join(k) for k in keep]
            swap = [", ".join(_pat(src).sub(dst, w) for w in k) for k in keep]
            zc = enc_text_masked(corr)
            zs = enc_text_masked(swap)
            out += ((zi*zc).sum(-1) - (zi*zs).sum(-1)).cpu().tolist()
    return np.array(out)

fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
for i, (a, b) in enumerate(PAIRS):
    g = gap_values(a, b)
    ax[i].hist(g, bins=45, color="#2b6cb0", alpha=0.85)
    ax[i].axvline(0, color="k", lw=1)
    ax[i].axvline(g.mean(), color="#c44e52", lw=2, ls="--")
    ax[i].set_title(f"{a} vs {b}\nmean {g.mean():+.4f},  "
                    f"{100*(g>0).mean():.1f}% > 0,  n={len(g)}", fontsize=9)
    ax[i].set_xlabel("confusion gap")
    print(f"{a}/{b}: {g.mean():+.4f}  {100*(g>0).mean():.1f}%  n={len(g)}", flush=True)
ax[0].set_ylabel("count")
plt.tight_layout(); save("fig_gap_hist"); plt.show()




import pickle
raw = pickle.load(open("/workspace/gap_raw.pkl", "rb"))

fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
for i, k in enumerate(["potato/pineapple", "onion/garlic", "lemon/lime", "oregano/basil"]):
    g = raw[k]
    ax[i].hist(g, bins=45, color="#2b6cb0", alpha=0.85)
    ax[i].axvline(0, color="k", lw=1)
    ax[i].axvline(g.mean(), color="#c44e52", lw=2, ls="--")
    ax[i].set_title(f"{k}\nmean {g.mean():+.4f},  {100*(g>0).mean():.1f}% > 0,  n={len(g)}",
                    fontsize=9)
    ax[i].set_xlabel("confusion gap")
ax[0].set_ylabel("count")
plt.tight_layout(); save("fig_gap_hist"); plt.show()






