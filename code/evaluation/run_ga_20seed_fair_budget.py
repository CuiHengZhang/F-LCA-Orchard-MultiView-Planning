from exp_suite_common_fair_budget import SEEDS, run_ga, write_csv, aggregate_rows, print_brief
rows = []
print("\n===== GA (fair budget) =====")
for idx, seed in enumerate(SEEDS, start=1):
    row, _ = run_ga(seed)
    rows.append(row)
    print(f"run={idx}/{len(SEEDS)} | seed={seed} | primary={row['primary_cost']:.2f} | distance={row['total_distance']:.2f} | energy={row['energy_range_percent']:.2f}% | time={row['runtime_sec']:.2f}s")
write_csv("ga_20seed_fair_budget/seed_results.csv", rows)
write_csv("ga_20seed_fair_budget/summary.csv", aggregate_rows(rows))
print_brief(rows)
