# Assignment 3 — Complete Command Reference
# ==========================================
# Copy-paste these commands in order. Each section corresponds to
# a Part in the assignment spec. Run inside fashion-ann-pipeline/.

# ──────────────────────────────────────────
# PREREQUISITES — Install dependencies
# ──────────────────────────────────────────
python -m venv venv
venv\Scripts\activate                         # Windows
# source venv/bin/activate                   # macOS/Linux

pip install -r requirements.txt

# ──────────────────────────────────────────
# PART A — Git fundamentals
# ──────────────────────────────────────────

# A1 — Init + first commit
git init -b main
git config user.name "Hassan"
git config user.email "hassan@example.com"
git add README.md .gitignore requirements.txt
git commit -m "A1: initial project scaffold (README, .gitignore, requirements)"

# Add GitHub remote (replace URL with yours)
git remote add origin https://github.com/YOUR_USERNAME/fashion-ann-pipeline.git
git push -u origin main

# A2 — dev branch with 6 incremental commits
git checkout -b dev
git add params.yaml
git commit -m "A2[1/6]: add params.yaml — central hyperparameters"
git add src/prepare.py
git commit -m "A2[2/6]: add src/prepare.py — download Fashion-MNIST raw data"
git add src/preprocess.py
git commit -m "A2[3/6]: add src/preprocess.py — normalize and split data"
git add src/train.py
git commit -m "A2[4/6]: add src/train.py — build and train ANN"
git add src/evaluate.py
git commit -m "A2[5/6]: add src/evaluate.py — compute metrics & confusion matrix"
git add dvc.yaml
git commit -m "A2[6/6]: add dvc.yaml — four-stage DVC pipeline definition"
git push -u origin dev

# A3 — Log variants (screenshot outputs for report)
git log --oneline --graph --all
git log --stat -3
git log -p -1
git log main..dev

# A4 — Diff variants
# (a) make a small edit, then:
git diff                      # unstaged working changes
git add <file>
git diff --staged             # staged changes
git restore --staged <file>   # un-stage
# Branch diffs:
git diff main..dev            # all commits reachable from dev but not main
git diff main...dev           # changes since their common ancestor (merge base)

# A5 — Stash scenario
# (while mid-edit on preprocess.py, uncommitted)
git stash push -m "WIP: preprocess mid-edit"
git stash list
git checkout main             # switch away
git checkout dev              # come back
git stash pop                 # resume

# A6 — Hotfix branch + rebase
git checkout main
git checkout -b hotfix
# (edit README.md — fix typo)
git add README.md
git commit -m "A6: hotfix — fix README typo"
git checkout main
git merge hotfix --no-ff -m "A6: merge hotfix into main"
git branch -d hotfix
# Before rebase:
git log --oneline --graph --all
# Rebase dev onto updated main:
git checkout dev
git rebase main
# If conflict: resolve → git add . → git rebase --continue
git log --oneline --graph --all

# A7 — Reset demo (on a scratch branch)
git checkout -b scratch-reset
echo "commit A" > scratch_A.txt && git add . && git commit -m "A7: commit A"
echo "commit B" > scratch_B.txt && git add . && git commit -m "A7: commit B"
git reset --soft HEAD~1       # commit B removed; changes remain STAGED
git status                    # see staged changes
git reset --hard HEAD~1       # commit A removed; changes fully DISCARDED
git status                    # clean working tree
git checkout dev && git branch -D scratch-reset

# A8 — git mv + git rm
echo "obsolete" > old_scratch.py && git add . && git commit -m "A8 setup: add file"
git rm old_scratch.py
git commit -m "A8: remove obsolete file via git rm"
git mv src/prepare.py src/prepare.py   # (example: move a file into a subfolder)
# Real example: git mv prepare.py src/prepare.py  (if it were at root)
git commit -m "A8: reorganize script location via git mv"

# ──────────────────────────────────────────
# PART B — Run scripts individually (test)
# ──────────────────────────────────────────
python src/prepare.py
python src/preprocess.py
python src/train.py
python src/evaluate.py

# ──────────────────────────────────────────
# PART C — DVC + Google Drive setup
# ──────────────────────────────────────────

# C1 — Install DVC with Google Drive extra
pip install "dvc[gdrive]"

# C2 — Init DVC (on dev branch, after A1 commit)
git checkout dev
dvc init
git add .dvc .dvcignore
git commit -m "C2: dvc init — initialize DVC in repository"

# C3 — Add Google Drive remote
# 1. Create a folder in Google Drive
# 2. Get the folder ID from the URL:
#    https://drive.google.com/drive/folders/FOLDER_ID_HERE
dvc remote add -d gdrive_storage gdrive://FOLDER_ID_HERE
git add .dvc/config
git commit -m "C3: configure Google Drive DVC remote"

# C4 — Authenticate (triggers OAuth browser flow)
dvc push   # first push — opens browser for Google OAuth

# C5 — Track artifacts with DVC
dvc add data/raw data/processed models
git add data/raw.dvc data/processed.dvc models.dvc .gitignore
git commit -m "C5: track data/raw, data/processed, models with DVC"
dvc push

# ──────────────────────────────────────────
# PART D — DVC Pipeline
# ──────────────────────────────────────────

# D3 — Run full pipeline (all 4 stages)
dvc repro

# Commit pipeline artifacts
git add dvc.lock metrics.json
git commit -m "D3: run dvc repro — v1 pipeline (all 4 stages executed)"
git tag v1
git push origin main --tags
dvc push

# D4 — Change a hyperparameter and re-run
# Edit params.yaml: change dense_units from 256 to 512
dvc repro
# Observe: preprocess skipped, train+evaluate re-run

# D5 — Commit v2
git add params.yaml dvc.lock metrics.json
git commit -m "D5: increase dense_units to 512 — v2 model"
git tag v2
git push origin main --tags
dvc push

# ──────────────────────────────────────────
# PART E — Conflict simulation
# ──────────────────────────────────────────
python simulate_conflict.py   # runs E1–E5 automatically
# Or follow the manual steps inside the script for screenshots

# ──────────────────────────────────────────
# USEFUL INSPECTION COMMANDS
# ──────────────────────────────────────────
dvc status                    # check if pipeline is up-to-date
dvc dag                       # visualize pipeline DAG
dvc metrics show              # show current metrics
dvc metrics diff v1 v2        # compare metrics between tags
dvc params diff v1 v2         # compare params between tags
git log --oneline --graph --all
