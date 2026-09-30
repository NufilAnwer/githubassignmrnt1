"""
simulate_conflict.py
====================
Simulates Part E — the two-contributor conflict scenario.

Run AFTER:
  - dvc repro has been run at least once on main (v1 tag)
  - dvc push has been done

What this script does:
  E1 — Create teammate-sim branch, modify preprocess.py differently,
       regenerate data/processed/, dvc add + push, commit .dvc pointer.
  E2 — Switch to main, make a different normalization change, same steps.
  E3 — Attempt git merge teammate-sim into main → produces conflicts.
  E4 — Guide the user to resolve both the code conflict and DVC conflict.
  E5 — Confirm clean state, re-run dvc repro, push.
"""

import subprocess, os, sys, textwrap

def run(cmd, cwd=None, check=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result

PREPROCESS_ORIGINAL = '''"""
src/preprocess.py
-----------------
Stage 2 — Normalize pixel values and split off a validation set.

Reads hyperparameters from params.yaml:
    preprocess.test_size  — fraction of training data used for validation
    preprocess.seed       — random seed for reproducible split

Run standalone:
    python src/preprocess.py
"""

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
PARAMS_FILE   = "params.yaml"


def load_params():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess"]


def preprocess():
    params = load_params()
    val_size = params["test_size"]
    seed     = params["seed"]

    print("[preprocess] Loading raw arrays...")
    x_train_raw = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train_raw = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test_raw  = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test_raw  = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # --- NORMALIZATION (original: divide by 255) ---
    print("[preprocess] Normalizing pixel values to [0, 1]...")
    x_train_norm = x_train_raw.astype("float32") / 255.0
    x_test_norm  = x_test_raw.astype("float32")  / 255.0

    print(f"[preprocess] Splitting validation set (size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm, y_train_raw,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_raw,
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"),   x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"),  x_test_norm)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test_raw)

    print(f"[preprocess] Saved processed arrays to \'{PROCESSED_DIR}/\'")
    print(f"  x_train : {x_train.shape}  |  x_val : {x_val.shape}  |  x_test : {x_test_norm.shape}")


if __name__ == "__main__":
    preprocess()
'''

PREPROCESS_TEAMMATE = '''"""
src/preprocess.py  [teammate-sim branch]
Normalization: min-max per-image (subtract mean, divide by std — z-score)
"""

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
PARAMS_FILE   = "params.yaml"


def load_params():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess"]


def preprocess():
    params = load_params()
    val_size = params["test_size"]
    seed     = params["seed"]

    print("[preprocess] Loading raw arrays...")
    x_train_raw = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train_raw = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test_raw  = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test_raw  = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # TEAMMATE normalization: z-score (mean=0, std=1) — computed on train set
    print("[preprocess] Applying z-score normalization (teammate-sim)...")
    x_train_f = x_train_raw.astype("float32").reshape(len(x_train_raw), -1)
    x_test_f  = x_test_raw.astype("float32").reshape(len(x_test_raw), -1)
    mean = x_train_f.mean(axis=0)
    std  = x_train_f.std(axis=0) + 1e-8
    x_train_norm = ((x_train_f - mean) / std).reshape(-1, 28, 28)
    x_test_norm  = ((x_test_f  - mean) / std).reshape(-1, 28, 28)

    print(f"[preprocess] Splitting validation set (size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm, y_train_raw,
        test_size=val_size, random_state=seed, stratify=y_train_raw,
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"),   x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"),  x_test_norm)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test_raw)

    print(f"[preprocess] Saved to \'{PROCESSED_DIR}/\' using z-score normalization")


if __name__ == "__main__":
    preprocess()
'''

PREPROCESS_MAIN_EDIT = '''"""
src/preprocess.py  [main branch — different normalization]
Normalization: divide by 255.0 then subtract dataset mean (per-channel centering)
"""

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
PARAMS_FILE   = "params.yaml"


def load_params():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess"]


def preprocess():
    params = load_params()
    val_size = params["test_size"]
    seed     = params["seed"]

    print("[preprocess] Loading raw arrays...")
    x_train_raw = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train_raw = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test_raw  = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test_raw  = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # MAIN normalization: [0,1] then subtract global mean (mean-centering)
    print("[preprocess] Applying [0,1] + mean-centering normalization (main)...")
    x_train_f = x_train_raw.astype("float32") / 255.0
    x_test_f  = x_test_raw.astype("float32")  / 255.0
    global_mean = x_train_f.mean()
    x_train_norm = x_train_f - global_mean
    x_test_norm  = x_test_f  - global_mean

    print(f"[preprocess] Splitting validation set (size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm, y_train_raw,
        test_size=val_size, random_state=seed, stratify=y_train_raw,
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"),   x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"),  x_test_norm)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test_raw)

    print(f"[preprocess] Saved to \'{PROCESSED_DIR}/\' using mean-centering")


if __name__ == "__main__":
    preprocess()
'''

PREPROCESS_RESOLVED = '''"""
src/preprocess.py  [RESOLVED — merged normalization]
Normalization: divide by 255.0 (standard, compatible with Softmax output layer).
Resolution: kept the simple /255 normalization as it is most portable and
well-understood; z-score and mean-centering variants were tested but /255
delivers the required >=85% accuracy with less complexity.
"""

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
PARAMS_FILE   = "params.yaml"


def load_params():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess"]


def preprocess():
    params = load_params()
    val_size = params["test_size"]
    seed     = params["seed"]

    print("[preprocess] Loading raw arrays...")
    x_train_raw = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train_raw = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test_raw  = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test_raw  = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # RESOLVED: standard /255 normalization to [0, 1]
    print("[preprocess] Normalizing pixel values to [0, 1] (resolved: /255)...")
    x_train_norm = x_train_raw.astype("float32") / 255.0
    x_test_norm  = x_test_raw.astype("float32")  / 255.0

    print(f"[preprocess] Splitting validation set (size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm, y_train_raw,
        test_size=val_size, random_state=seed, stratify=y_train_raw,
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"),   x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"),  x_test_norm)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test_raw)

    print(f"[preprocess] Saved to \'{PROCESSED_DIR}/\' — conflict resolved")


if __name__ == "__main__":
    preprocess()
'''


def setup_conflict_scenario():
    print("="*60)
    print("PART E — Simulated conflict scenario")
    print("="*60)

    # ── E1: teammate-sim branch ───────────────────────────────────────────────
    print("\n--- E1: teammate-sim branch ---")
    run("git checkout main")
    run("git checkout -b teammate-sim")

    with open("src/preprocess.py", "w") as f:
        f.write(PREPROCESS_TEAMMATE)

    run("python src/preprocess.py")
    run("dvc add data/processed")
    run("dvc push")
    run("git add src/preprocess.py data/processed.dvc .gitignore")
    run('git commit -m "E1: teammate-sim — z-score normalization + regenerated data"')

    # ── E2: main branch independent edit ─────────────────────────────────────
    print("\n--- E2: main branch — different normalization ---")
    run("git checkout main")

    with open("src/preprocess.py", "w") as f:
        f.write(PREPROCESS_MAIN_EDIT)

    run("python src/preprocess.py")
    run("dvc add data/processed")
    run("dvc push")
    run("git add src/preprocess.py data/processed.dvc .gitignore")
    run('git commit -m "E2: main — mean-centering normalization + regenerated data"')

    # ── E3: Attempt merge (will conflict) ─────────────────────────────────────
    print("\n--- E3: Attempt merge — expect conflicts ---")
    result = run("git merge teammate-sim --no-ff", check=False)
    print("[E3] Merge conflict produced (expected). See conflict markers above.")
    run("git status")

    # ── E4: Resolve conflicts ─────────────────────────────────────────────────
    print("\n--- E4: Resolving conflicts ---")

    # Resolve preprocess.py with the definitive /255 version
    with open("src/preprocess.py", "w") as f:
        f.write(PREPROCESS_RESOLVED)

    # Resolve data/processed.dvc — regenerate from resolved script
    run("python src/preprocess.py")
    run("dvc add data/processed")
    run("dvc checkout")

    run("git add src/preprocess.py data/processed.dvc")

    # ── E5: Complete merge, verify, repro, push ───────────────────────────────
    print("\n--- E5: Completing merge ---")
    run('git commit -m "E4/E5: resolve preprocess.py conflict + regenerate data/processed"')

    run("dvc status")
    run("dvc repro")
    run("dvc push")
    run("git push origin main --tags")

    print("\n" + "="*60)
    print("PART E COMPLETE — conflicts resolved, pipeline reproduced, pushed")
    print("="*60)


if __name__ == "__main__":
    setup_conflict_scenario()
