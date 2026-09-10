import pickle, torch
from transformers import AutoTokenizer
import align_recipe1m as A

records, vocab_idx = A.load_annotations()
tok = AutoTokenizer.from_pretrained(A.TEXT_MODEL)
model = A.AlignmentModel(len(vocab_idx)).to(A.DEVICE)
model.load_state_dict(torch.load(
    "/workspace/align/keep/align_train_20k_seed42.pth", map_location=A.DEVICE))

res, raw = A.confusion_gap(model, records["val"], "val", tok, return_raw=True)
for k, v in res.items():
    print(f"{k:22s} {v['gap']:+.4f}  {100*v['positive_rate']:5.1f}%  n={v['n']}")
pickle.dump(raw, open("/workspace/gap_raw.pkl", "wb"))
print("saved /workspace/gap_raw.pkl")
