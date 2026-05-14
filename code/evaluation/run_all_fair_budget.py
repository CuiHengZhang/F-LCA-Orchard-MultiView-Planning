import subprocess
import sys

SCRIPTS = [
    'run_flca_20seed_fair_budget.py',
    'run_flca_no_hetero_20seed_fair_budget.py',
    'run_flca_no_layered_20seed_fair_budget.py',
    'run_flca_no_orthogonal_20seed_fair_budget.py',
]


def main():
    for script in SCRIPTS:
        print(f"\n===== Running {script} =====")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
