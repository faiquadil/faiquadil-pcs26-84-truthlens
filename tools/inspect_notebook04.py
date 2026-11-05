"""READ-ONLY inspection of notebooks/04_transformer_experiment.ipynb (PCS26/84).
It only READS files. It does not run notebook cells, train anything, or write/delete/rename any file."""
import json, re, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

root = Path.cwd()
for cand in [root, *root.parents]:
    if (cand / "dataset").is_dir():
        root = cand
        break
print("Project folder:", root)

nb_path = root / "notebooks" / "04_transformer_experiment.ipynb"
if not nb_path.exists():
    print("NOT FOUND:", nb_path)
    sys.exit(1)
nb = json.loads(nb_path.read_text(encoding="utf-8"))
print(f"Notebook size: {nb_path.stat().st_size // 1024} KB | cells: {len(nb['cells'])}")

KEYS = ("Pretrained", "Mode:", "status", "NOT EXECUTED", "READY", "Estimated", "Rows trainable",
        "Parameters", "Median time", "Peak GPU", "Skipped", "Training NOT", "epoch ", "Existing saved",
        "Saved model", "No trained", "Eligible rows", "Rows used", "RAM", "CUDA available")
print("\n=== CODE CELLS (number = order among code cells) ===")
n = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    n += 1
    first = "".join(c["source"]).strip().split("\n")[0][:58]
    has_err = any(o.get("output_type") == "error" for o in c.get("outputs", []))
    state = "ERROR" if has_err else ("ran" if c.get("outputs") or c.get("execution_count") else "NOT RUN / no output")
    print(f"\n[code {n:2d}] exec={c.get('execution_count')} | {state} | {first}")
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            text = "".join(o["text"])
        elif o.get("output_type") == "error":
            text = "ERROR: " + o.get("ename", "") + ": " + o.get("evalue", "")[:150]
        elif "text/plain" in o.get("data", {}) or "text/markdown" in o.get("data", {}):
            text = "".join(o["data"].get("text/markdown", o["data"].get("text/plain", "")))
        else:
            continue
        for line in text.splitlines():
            if line.startswith("ERROR") or any(k in line for k in KEYS):
                print("      >", line.strip()[:160])

print("\n=== CONFIGURATION VALUES WRITTEN IN THE NOTEBOOK (not necessarily the values in memory) ===")
for c in nb["cells"]:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        for name in ("RUN_TRAINING", "TRAIN_SUBSET", "VAL_SUBSET", "MAX_LEN", "BATCH_SIZE", "RUN_FINAL_TEST"):
            m = re.search(r"^" + name + r"\s*=\s*([^\n#]+)", src, re.M)
            if m:
                print(f"  {name} = {m.group(1).strip()}")

print("\n=== FILES (checked only) ===")
def show(rel, is_dir=False):
    p = root / rel
    ok = p.is_dir() if is_dir else p.exists()
    extra = ""
    if ok and is_dir:
        extra = " -> " + ", ".join(sorted(x.name for x in p.iterdir())[:10])
    print(f"  {'FOUND  ' if ok else 'missing'} {rel}{extra}")
for rel in ("dataset/combined_fake_news_clean.csv", "results/tables/data_split_manifest.csv",
            "results/tables/p1_eligible_rows.csv", "models/phase1/ml_selected_config.json",
            "results/tables/p1_dl_validation_comparison.csv"):
    show(rel)
show("models/transformer_distilbert", is_dir=True)
show("models/transformer_distilbert_SMOKE", is_dir=True)
for pat in ("p1_transformer*", "p1_smoke_transformer*"):
    found = sorted(x.name for x in (root / "results" / "tables").glob(pat))
    print(f"  results/tables/{pat}: {found if found else 'none'}")
for pat in ("22_*transformer*", "23_*transformer*"):
    found = sorted(x.name for x in (root / "results" / "figures").glob(pat))
    print(f"  results/figures/{pat}: {found if found else 'none'}")

print("\n=== MACHINE ===")
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"  RAM total {vm.total/1e9:.1f} GB | available now {vm.available/1e9:.1f} GB")
except ImportError:
    print("  psutil not installed (optional)")
print("\nDone. Nothing was changed.")
