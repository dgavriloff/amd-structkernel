import os
import re
import json
import shutil

src_base = r"C:\Users\ooousay\Documents\repos\amd-structkernel\structkernel-transfer\kernel-state"
dst_base = r"C:\Users\ooousay\Documents\repos\amd-structkernel\kernels"

kernels = ["mxfp4-mm", "mixed-mla", "moe-mxfp4"]

best_data = {
    "mxfp4-mm": {"version": "v188", "score": "9.01µs"},
    "mixed-mla": {"version": "v057", "score": "28.8µs (LB) / 26.5µs (BM)"},
    "moe-mxfp4": {"version": "v106", "score": "~139µs (LB)"}
}

for kernel in kernels:
    src_dir = os.path.join(src_base, kernel)
    dst_dir = os.path.join(dst_base, kernel)
    os.makedirs(dst_dir, exist_ok=True)
    state_dir = os.path.join(dst_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    
    # 1. problem.md
    claude_md_path = os.path.join(src_dir, "CLAUDE.md")
    if os.path.exists(claude_md_path):
        with open(claude_md_path, "r", encoding="utf-8") as f:
            claude_content = f.read()
        
        end_pattern = re.search(r"## (Files|Status & Results)", claude_content)
        if end_pattern:
            problem_content = claude_content[:end_pattern.start()].strip()
        else:
            problem_content = claude_content
            
        with open(os.path.join(dst_dir, "problem.md"), "w", encoding="utf-8") as f:
            f.write(problem_content)
        
    # 2. state/tried.jsonl & state/dead.jsonl
    results_md_path = os.path.join(src_dir, "results.md")
    if os.path.exists(results_md_path):
        with open(results_md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        tried_entries = []
        dead_entries = []
        
        current_branch = None
        current_branch_fails = []
        is_exhausted = False
        
        for line in lines:
            line = line.strip()
            branch_match = re.match(r"^## Branch: (.*)", line)
            if branch_match:
                current_branch = branch_match.group(1).strip()
                current_branch_fails = []
                is_exhausted = False
                continue
                
            if line.startswith("→") or line.lower().startswith("branch exhausted"):
                if "branch exhausted" in line.lower() or "exhausted" in line.lower():
                    is_exhausted = True
                
                if current_branch:
                    dead_entries.append({
                        "branch": current_branch,
                        "insight": line.lstrip("→ ").strip(),
                        "failed_techniques": current_branch_fails if is_exhausted else []
                    })
                    if is_exhausted:
                        current_branch_fails = []
                continue

            if not line.startswith("|") or line.startswith("| #") or line.startswith("|---"):
                continue
                
            # Table row
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 6:
                version = parts[1]
                if not version.startswith("v"):
                    continue
                if len(parts) >= 8:
                    target = parts[2]
                    hypothesis = parts[3]
                    test = parts[4]
                    score = parts[5]
                    vs_best = parts[6]
                    keep = parts[7]
                elif len(parts) == 7:
                    target = ""
                    hypothesis = parts[2]
                    test = parts[3]
                    score = parts[4]
                    vs_best = parts[5]
                    keep = parts[6]
                else:
                    continue

                tried_entries.append({
                    "version": version,
                    "target": target,
                    "hypothesis": hypothesis,
                    "test": test,
                    "score": score,
                    "vs_best": vs_best,
                    "keep": keep,
                    "branch": current_branch
                })
                
                if "no" in keep.lower() or "fail" in test.lower():
                    current_branch_fails.append({
                        "version": version,
                        "hypothesis": hypothesis,
                        "reason": keep,
                    })

        with open(os.path.join(state_dir, "tried.jsonl"), "w", encoding="utf-8") as f:
            for entry in tried_entries:
                f.write(json.dumps(entry) + "\n")
                
        with open(os.path.join(state_dir, "dead.jsonl"), "w", encoding="utf-8") as f:
            for entry in dead_entries:
                f.write(json.dumps(entry) + "\n")
                
    # 4. state/best.json
    with open(os.path.join(state_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best_data.get(kernel, {}), f, indent=2)
        
    # 5. Copy files
    for fname in ["submission.py", "best_submission.py"]:
        src_file = os.path.join(src_dir, fname)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dst_dir, fname))
            
print("Done extracting files.")
