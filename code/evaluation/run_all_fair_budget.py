import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "run_flca_20seed_fair_budget.py",
    "run_flca_no_hetero_20seed_fair_budget.py",
    "run_flca_no_layered_20seed_fair_budget.py",
    "run_flca_no_orthogonal_20seed_fair_budget.py",
]


def main():
    for script in SCRIPTS:
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")
        print(f"\n===== Running {script} =====")
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(SCRIPT_DIR))
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
