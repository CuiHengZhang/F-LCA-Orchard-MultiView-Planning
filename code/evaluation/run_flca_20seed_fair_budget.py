from exp_suite_common_fair_budget import (
    SEEDS,
    run_flca,
    save_results_bundle,
    print_brief,
    print_saved_paths,
)

OUT_DIR = "flca_20seed_fair_budget"

def main():
    rows = []
    print("\n===== F-LCA (fair budget) =====")
    total_runs = len(SEEDS)
    for idx, seed in enumerate(SEEDS, start=1):
        row, _ = run_flca(seed)
        rows.append(row)
        print(
            f"run={idx}/{total_runs} | seed={seed} | "
            f"primary={row['primary_cost']:.2f} | "
            f"distance={row['total_distance']:.2f} | "
            f"energy={row['energy_range_percent']:.2f}% | "
            f"time={row['runtime_sec']:.2f}s"
        )

    save_results_bundle(OUT_DIR, rows)
    print("\n===== Summary =====")
    print_brief(rows)
    print_saved_paths(OUT_DIR)

if __name__ == "__main__":
    main()
