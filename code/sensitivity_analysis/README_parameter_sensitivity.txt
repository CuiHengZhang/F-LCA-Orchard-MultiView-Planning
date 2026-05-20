# F-LCA parameter sensitivity analysis

This folder contains the release script for rerunning the one-at-a-time F-LCA parameter sensitivity analysis.

## Files in this folder

- `run_flca_parameter_sensitivity_release.py`: script for rerunning the parameter sensitivity analysis.
- `README_parameter_sensitivity.txt`: this description file.

## Curated manuscript-ready results

The cleaned results used in the manuscript and supplementary material are stored in:

- `results/parameter_sensitivity/flca_parameter_sensitivity_seed_results_clean.csv`
- `results/parameter_sensitivity/flca_parameter_sensitivity_summary_clean.csv`
- `results/parameter_sensitivity/table_S8_parameter_sensitivity.csv`
- `results/parameter_sensitivity/table_S8_parameter_sensitivity_short.csv`
- `results/parameter_sensitivity/figS6_parameter_sensitivity_source_data.csv`

Supplementary Fig. S6 figure files and source data are also available under:

- `figures_source_data/supplementary_figures/figS6_parameter_sensitivity/`

## Rerun output location

To avoid overwriting the curated manuscript-ready files, rerunning

```bash
python code/sensitivity_analysis/run_flca_parameter_sensitivity_release.py
```

writes new outputs to:

- `outputs/parameter_sensitivity_release/flca_parameter_sensitivity_seed_results_clean.csv`
- `outputs/parameter_sensitivity_release/flca_parameter_sensitivity_summary_clean.csv`

## Parameters

The tested parameters are:

- `lambda_o`: obstacle-penalty weight.
- `partition_battery_power`: battery-aware task-partition exponent.
- `lambda_b`: path-balance weight.
- `lambda_c`: observation-completeness penalty multiplier, applied jointly to the missing observation pose penalty and the incomplete-tree penalty.

Each parameter is perturbed one at a time using multipliers of 0.8x, 0.9x, 1.0x, 1.1x, and 1.2x, while all other parameters are kept at the baseline values used in the main experiments. Each setting is evaluated on 20 matched random instances with seeds 42-61.

## Cleaning note

The cleaned result files retain only the metrics used in the manuscript, Supplementary Fig. S6, and Supplementary Table S8. Auxiliary iterative logging fields not used for the analysis were removed for clarity.
