"""
diagnose_signals.py
===================
Quick diagnostic: print statistics for the two new signals to see if they
carry any real information vs being constant/correlated with existing signals.
"""
import torch
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

def stats(tensor, name):
    t = tensor.float().view(-1)
    print(f"  {name:30s}  min={t.min():.4f}  max={t.max():.4f}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}")

def correlation(a, b, name_a, name_b):
    a_flat = a.float().view(-1)
    b_flat = b.float().view(-1)
    a_z = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-9)
    b_z = (b_flat - b_flat.mean()) / (b_flat.std() + 1e-9)
    r = (a_z * b_z).mean()
    print(f"  corr({name_a:20s}, {name_b:20s}) = {r:.4f}")

print("=" * 65)
print("  Signal Diagnostics — Task A Train")
print("=" * 65)

p3  = torch.load(os.path.join(CACHE_DIR, "phase3_task_a_train.pt"),  map_location="cpu")
att = torch.load(os.path.join(CACHE_DIR, "attention_task_a_train.pt"), map_location="cpu")

# Phase 3 signals
iapd = p3["iapd_embeddings"].float()   # (N,3,512)
imgs = p3["image_embeddings"].float()  # (N,C,512)
caps = p3["caption_embeddings"].float()
gap  = p3["fig_lit_gap"].float()

lit = iapd[:,0,:].unsqueeze(1); fig = iapd[:,1,:].unsqueeze(1); ctx = iapd[:,2,:].unsqueeze(1)
s0 = (ctx*imgs).sum(-1); s1 = (fig*imgs).sum(-1); s2 = (lit*imgs).sum(-1)
s3 = (fig*caps).sum(-1); s7 = gap

# New signals
spatial = att["spatial_concentration"]  # (N,C)
cgap    = att["concept_gap"]            # (N,C)

print("\n  Raw signal value ranges:")
stats(s0,      "s0: sim(ctx,img)")
stats(s1,      "s1: sim(fig,img)")
stats(s2,      "s2: sim(lit,img)")
stats(s3,      "s3: sim(fig,cap)")
stats(s7,      "s7: fig_lit_gap")
stats(spatial, "s8: spatial_conc [NEW]")
stats(cgap,    "s9: concept_gap  [NEW]")

print("\n  Correlation of new signals with existing ones:")
correlation(spatial, s1,      "spatial_conc", "sim(fig,img)")
correlation(spatial, s7,      "spatial_conc", "fig_lit_gap")
correlation(cgap,    s1,      "concept_gap",  "sim(fig,img)")
correlation(cgap,    s7,      "concept_gap",  "fig_lit_gap")
correlation(spatial, cgap,    "spatial_conc", "concept_gap")

labels = p3["labels"]
N, C   = spatial.shape

print("\n  Predictive power (does correct image rank highest on each signal?):")
for sig, name in [(s1, "sim(fig,img)"), (s7, "fig_lit_gap"),
                  (spatial, "spatial_conc (LOW=fig)"), (cgap, "concept_gap")]:
    # spatial_concentration: LOW = figurative, so use argmin for it
    if "spatial" in name:
        preds = sig.argmin(dim=-1)
        label_txt = "argmin"
    else:
        preds = sig.argmax(dim=-1)
        label_txt = "argmax"
    acc = (preds == labels).float().mean().item()
    print(f"  {name:35s}  {label_txt} accuracy = {acc:.1%}")

print("\n  Sample spatial_concentration values (first 5 idioms):")
for i in range(min(5, N)):
    vals  = spatial[i].tolist()
    label = labels[i].item()
    vals_str = "  ".join(f"{v:.4f}{'*' if j==label else ' '}" for j,v in enumerate(vals))
    print(f"  idiom {i:2d}: [{vals_str}]  (* = correct)")

print("\n  Sample concept_gap values (first 5 idioms):")
for i in range(min(5, N)):
    vals  = cgap[i].tolist()
    label = labels[i].item()
    vals_str = "  ".join(f"{v:.4f}{'*' if j==label else ' '}" for j,v in enumerate(vals))
    print(f"  idiom {i:2d}: [{vals_str}]  (* = correct)")
