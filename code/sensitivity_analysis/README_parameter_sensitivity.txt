# F-LCA parameter sensitivity analysis

This folder contains the cleaned data and release script for the F-LCA parameter sensitivity analysis.

## Files

- `run_flca_parameter_sensitivity_release.py`: release script for rerunning the one-at-a-time parameter sensitivity analysis.
- `flca_parameter_sensitivity_seed_results_clean.csv`: cleaned seed-level results for all four tested parameters.
- `flca_parameter_sensitivity_summary_clean.csv`: cleaned summary statistics for all tested parameter levels.
- `Table_S8_ready.csv`: formatted table values for Supplementary Table S8.
- `Table_S8_ready_short.csv`: shortened version of Supplementary Table S8 without the parameter-meaning column.
- `Fig_S6_source_data.csv`: source data used for Supplementary Fig. S6.

## Parameters

The tested parameters were:
- `lambda_o`: obstacle-penalty weight.
- `partition_battery_power`: battery-aware task-partition exponent.
- `lambda_b`: path-balance weight.
- `lambda_c`: observation-completeness penalty multiplier, applied jointly to the missing observation pose penalty and the incomplete-tree penalty.

Each parameter was perturbed one at a time using multipliers of 0.8x, 0.9x, 1.0x, 1.1x, and 1.2x, while all other parameters were kept at the baseline values used in the main experiments. Each setting was evaluated on 20 matched random instances with seeds 42–61.

## Cleaning note

The cleaned result files retain only the metrics used in the manuscript, Supplementary Fig. S6, and Supplementary Table S8. Auxiliary iterative logging fields not used for the analysis were removed for clarity.
