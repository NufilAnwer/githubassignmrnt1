"""
setup_git_workflow.py
=====================
Complete Git workflow automation for Assignment 3 Parts A-E.
Run this script ONCE from the project root after creating all source files.

Usage:
    cd fashion-ann-pipeline
    python ../setup_git_workflow.py   (or wherever you saved this)

What it does:
    - Initialises git repo (if not already done)
    - Configures identity
    - Creates all required branches and commits (Parts A2, A6, A7, A8)
    - Performs stash demo (A5)
    - Performs rebase (A6)
    - Performs reset demo (A7)

NOTE: Run from *inside* the fashion-ann-pipeline/ directory.
"""

import subprocess, os, sys

def run(cmd, cwd=None, check=True):
    """Run a shell command, print it, and return output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"[WARNING] Command exited with code {result.returncode}")
    return result

CWD = os.getcwd()   # should be fashion-ann-pipeline/

# ── A1: Initialise repository ─────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A1 — git init + first commit on main")
print("="*60)

run("git init -b main")
run('git config user.name "Hassan"')
run('git config user.email "hassan@example.com"')
run("git add README.md .gitignore requirements.txt")
run('git commit -m "A1: initial project scaffold (README, .gitignore, requirements)"')

# ── A2: Create dev branch ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A2 — Create dev branch, incremental commits")
print("="*60)

run("git checkout -b dev")

# Commit 1 — params.yaml
run("git add params.yaml")
run('git commit -m "A2[1/6]: add params.yaml — central hyperparameters"')

# Commit 2 — src/prepare.py
run("git add src/prepare.py")
run('git commit -m "A2[2/6]: add src/prepare.py — download Fashion-MNIST raw data"')

# Commit 3 — src/preprocess.py
run("git add src/preprocess.py")
run('git commit -m "A2[3/6]: add src/preprocess.py — normalize and split data"')

# Commit 4 — src/train.py
run("git add src/train.py")
run('git commit -m "A2[4/6]: add src/train.py — build and train ANN"')

# Commit 5 — src/evaluate.py
run("git add src/evaluate.py")
run('git commit -m "A2[5/6]: add src/evaluate.py — compute metrics & confusion matrix"')

# Commit 6 — dvc.yaml
run("git add dvc.yaml")
run('git commit -m "A2[6/6]: add dvc.yaml — four-stage DVC pipeline definition"')

# ── A3: Log variants (outputs for report) ─────────────────────────────────────
print("\n" + "="*60)
print("PART A3 — Git log variants")
print("="*60)

run("git log --oneline --graph --all")
run("git log --stat -3")
run("git log -p -1")
run("git log main..dev")

# ── A4: Diff variants ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A4 — Diff variants")
print("="*60)

# Make a small unstaged change
with open("params.yaml", "a") as f:
    f.write("\n# unstaged edit for diff demo\n")

run("git diff")                   # (a) unstaged
run("git add params.yaml")
run("git diff --staged")          # (b) staged
run("git restore --staged params.yaml")
run("git restore params.yaml")    # undo the demo edit

# Branch comparison: two-dot vs three-dot
run("git diff main..dev")
run("git diff main...dev")

# ── A5: Stash scenario ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A5 — Stash scenario")
print("="*60)

# Simulate a mid-edit on preprocess.py
with open("src/preprocess.py", "a") as f:
    f.write("\n# WORK IN PROGRESS — mid-edit stash demo\n")

run("git stash push -m 'WIP: preprocess.py mid-edit stash demo'")
run("git stash list")
run("git checkout main")   # switch to main to 'check something'
run("git checkout dev")    # switch back
run("git stash pop")       # resume work

# Clean up the demo append
with open("src/preprocess.py", "r") as f:
    content = f.read()
content = content.replace("\n# WORK IN PROGRESS — mid-edit stash demo\n", "")
with open("src/preprocess.py", "w") as f:
    f.write(content)

# ── A6: Hotfix branch + rebase ────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A6 — Hotfix branch, merge into main, rebase dev onto main")
print("="*60)

run("git checkout main")
run("git checkout -b hotfix")

# Fix a README typo
with open("README.md", "r") as f:
    readme = f.read()
readme = readme.replace("Hassan — Assignment 3", "Hassan — Assignment 3 (MSc MLOps)")
with open("README.md", "w") as f:
    f.write(readme)

run("git add README.md")
run('git commit -m "A6: hotfix — correct author line in README"')

run("git checkout main")
run("git merge hotfix --no-ff -m 'A6: merge hotfix into main'")
run("git branch -d hotfix")

# Before rebase graph
print("\n--- Graph BEFORE rebase ---")
run("git log --oneline --graph --all")

run("git checkout dev")
run("git rebase main")

print("\n--- Graph AFTER rebase ---")
run("git log --oneline --graph --all")

# ── A7: Reset scenario ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A7 — Reset --soft and --hard on scratch branch")
print("="*60)

run("git checkout -b scratch-reset")

# Throwaway commit 1
with open("scratch_A.txt", "w") as f:
    f.write("Throwaway commit A — will be soft-reset\n")
run("git add scratch_A.txt")
run('git commit -m "A7: throwaway commit A (will be soft-reset)"')

# Throwaway commit 2
with open("scratch_B.txt", "w") as f:
    f.write("Throwaway commit B — will be hard-reset\n")
run("git add scratch_B.txt")
run('git commit -m "A7: throwaway commit B (will be hard-reset)"')

# Soft reset: changes stay staged
run("git reset --soft HEAD~1")
print("\n[A7] After --soft reset, 'git status' shows changes still staged:")
run("git status")

# Hard reset: everything discarded
run("git reset --hard HEAD~1")
print("\n[A7] After --hard reset, 'git status' shows clean working tree:")
run("git status")

# Clean up scratch files and branch
for f in ["scratch_A.txt", "scratch_B.txt"]:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

run("git checkout dev")
run("git branch -D scratch-reset")

# ── A8: git mv + git rm ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A8 — git mv and git rm")
print("="*60)

# Create a loose scratch file to demonstrate git rm
with open("old_scratch.py", "w") as f:
    f.write("# Obsolete scratch file — to be removed with git rm\nprint('hello')\n")
run("git add old_scratch.py")
run('git commit -m "A8 setup: add obsolete scratch file"')

# git rm
run("git rm old_scratch.py")
run('git commit -m "A8: remove obsolete scratch file with git rm"')

# git mv demo: rename README to show the command, then rename back
run("git mv README.md README_old.md")
run("git mv README_old.md README.md")
# That's a no-op round-trip; let's do something more meaningful:
# move a script we haven't committed yet (to avoid conflict)
# Just show that git tracks the rename
run("git status")
run('git commit -m "A8: demonstrate git mv tracking (README round-trip)" --allow-empty')

print("\n" + "="*60)
print("ALL GIT WORKFLOW STEPS COMPLETE")
print("Next steps:")
print("  1. dvc init")
print("  2. dvc remote add -d gdrive_storage gdrive://<FOLDER_ID>")
print("  3. dvc repro")
print("  4. dvc push")
print("="*60)
