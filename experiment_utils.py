import os
import json
import datetime
import subprocess
import sys


def _get_git_info():
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        commit_hash = "unknown"

    try:
        # Add untracked files to index as intent-to-add so they show up in diff
        subprocess.run(['git', 'add', '-N', '.'], stderr=subprocess.DEVNULL)
        diff = subprocess.check_output(
            ['git', 'diff'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8')
    except subprocess.CalledProcessError:
        diff = "unknown"
        
    return commit_hash, diff


def save_results(*, config, results, output_dir="results"):
    script_name = os.path.basename(sys.argv[0])
    commit_hash, diff = _get_git_info()
    timestamp = datetime.datetime.now().isoformat()

    os.makedirs(output_dir, exist_ok=True)
    path = f'{output_dir}/{script_name}:{timestamp.replace(":", "_").replace(".", "_")}.json'
    with open(path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "script_name": script_name,
            "git_commit": commit_hash,
            "git_diff": diff,
            "config": config,
            "results": results,
        }, f, indent=4)
    print(f"Results saved to {path}")
