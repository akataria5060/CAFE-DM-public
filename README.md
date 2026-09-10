# CAFE-DM

Code for *CAFE-DM: A Confusion-Aware Diffusion Framework for Food Image
Generation*.

Food image generation methods confuse visually similar ingredients. RD-FGM
(Wang et al., 2024) reports this failure for pairs such as potato and
pineapple and proposes no mechanism to address it. CAFE-DM introduces a
confusion-aware learning strategy that explicitly penalises representational
overlap between such pairs, together with a **confusion gap** metric that
quantifies ingredient-level discriminability.

Two stages:

1. **Alignment** — a shared image-text embedding space on Recipe1M, trained
   under CLIP + ingredient BCE + batch-hard triplet + swap-based hard
   negative losses.
2. **Generation** — a latent diffusion model conditioned on the frozen text
   encoder from stage 1.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10, PyTorch 2.4 (NVIDIA container build). The `transformers` pin
matters: 4.45+ requires PyTorch >= 2.5. Install torch for your platform
first, then `pip install --no-deps -r requirements.txt`.

## Data

**Recipe1M** — http://im2recipe.csail.mit.edu/ (access request required):

```
<DATA>/det_ingrs.json    parsed ingredients with validity flags
<DATA>/layer1.json       recipe text and official partition labels
<DATA>/layer2.json       recipe id -> image id
<DATA>/{train,val,test}/ images, nested <split>/a/b/c/d/abcd1234.jpg
```

**ETH Food-101** — https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
**UEC Food-256** — http://foodcam.mobi/dataset256.zip

Set the path constants at the top of each script.

---

## Stage 1: confusion-aware alignment

```bash
python align_recipe1m.py --smoke --steps 200      # pipeline check on val
python align_recipe1m.py --steps 20000            # full run
python align_recipe1m.py --eval-only --ckpt <path>
python confusion_gap.py                           # dump per-sample gaps
```

Objective: `L = L_clip + 1.0*L_ing + 0.5*L_bh + 0.5*L_swap`

`L_swap` builds a hard negative by substituting one member of a pre-defined
confusable pair for the other in the ingredient text, then penalises the
model whenever that negative sits closer to the image than the correct text.

### Confusion gap protocol

`Gap = sim(image, correct text) - sim(image, swapped text)`

Fixed before training:

- recipes in the **val** split containing **exactly one** member of the pair
  (a recipe containing both would yield a swapped text still true of the
  image — a false negative, not a hard one)
- ingredient entries filtered on the dataset's own `valid` flag
- word-boundary matching including plural forms
- compound forms naming a different ingredient excluded (see `EXCLUDE`;
  e.g. *sweet potato* does not count as *potato*)
- 2,000 recipes sampled per pair, seed 42
- eval transform Resize(256) + CenterCrop(224)

Pairs: potato/pineapple, onion/garlic, lemon/lime, oregano/basil.
These apply to 44.0% of training recipes; the remainder receive a random
vocabulary substitution.

## Stage 2: latent diffusion

```bash
python diffusion_recipe1m.py
```

- `stabilityai/sd-vae-ft-mse`, frozen. 256x256 image -> 4x32x32 latent.
- `UNet2DConditionModel`, 202.5M parameters, cross-attention dim 768.
- Conditioned on token-level hidden states from the **frozen** stage-1 BERT.
- Conditioning dropout p=0.1, so classifier-free guidance is trained rather
  than approximated at sampling.
- 90,000 micro-steps at batch 32, accumulation 15 (effective batch 480).
- Sampling: DDIM, 100 steps, guidance 5.0.

## Classification transfer

```bash
python classify_food101.py
python classify_uec256.py
```

Each runs the same recipe (30 epochs, batch 32, AdamW 1e-4, 3-epoch warmup
+ cosine, mixup 0.2, label smoothing 0.1, EMA 0.999, 8-view TTA, seed 42)
and varies only the encoder initialisation: ImageNet weights, or the
stage-1 alignment checkpoint.

UEC Food-256 images are cropped to the bounding boxes in `bb_info.txt`, and
the split is **grouped by source photograph** so that no photograph appears
in both train and test. A multi-dish photo appears once per category it
contains, so an instance-level split would leak near-duplicates.

---

## Results

All numbers in `RESULTS.json`.

**Recipe1M generation** (test split, n=10,000, 100 DDIM steps, w=5.0):

| | CAFE-DM | RD-FGM |
|---|---|---|
| FID (lower better) | **62.14** | 82.45 |
| IS (higher better) | 7.93 +/- 0.15 | **16.88 +/- 0.05** |
| MS-SSIM (lower better) | 0.0702 | **0.0547** |

**Confusion gap**, Recipe1M val, n=2,000/pair:

| Pair | Real images | Generated images |
|---|---|---|
| onion / garlic | +0.3111 (92.0%) | +0.2762 (92.2%) |
| potato / pineapple | +0.2864 (94.3%) | +0.1789 (94.3%) |
| lemon / lime | +0.0554 (74.9%) | +0.0304 (57.6%) |
| oregano / basil | +0.0120 (62.4%) | +0.0112 (65.4%) |
| **mean** | **+0.1662 (80.9%)** | **+0.1242** |

Two pairs separate strongly; two remain close to chance. Separability
tracks intrinsic visual distinctness rather than training frequency.

**Classification**, identical recipe, only the initialisation varies:

| | Food-101 Top-1 (TTA) | UEC-256 Top-1 (TTA) |
|---|---|---|
| ResNet-50, ImageNet | — | 81.14 |
| Swin-B, ImageNet | 92.16 | 85.56 |
| Swin-B, alignment | **92.66** | **86.13** |
| RD-FGM | 91.01 | — |

Alignment pretraining contributes +0.50 on Food-101 and +0.57 on UEC-256.
The remaining margin over RD-FGM comes from the backbone and training
recipe.

UEC-256 prior work (WISeR 83.15, JDNet 84.00, RAFA-Net 91.56) uses the
dataset's five-fold cross-validation protocol, not the split above, and is
not directly comparable.

---

## Checkpoints

Not committed here — the alignment checkpoint is 752 MB and the diffusion
U-Net 773 MB, both over GitHub's file limit. See Releases, or retrain with
the commands above.

## Citation

```bibtex
@article{kataria2026cafedm,
  title  = {CAFE-DM: A Confusion-Aware Diffusion Framework for Food Image Generation},
  author = {Kataria, Abhishek and Nijhawan, Rahul and Goyal, Raman Kumar},
  year   = {2026}
}
```

## License

MIT
