# F-LCA Orchard Multi-View Path Planning

This repository provides the code, processed input data, result files, and supplementary materials supporting the manuscript:

**Observation-complete cooperative multi-view path planning for multi-UGV citrus orchard phenotyping**

## Overview

This study reformulates citrus orchard path planning as a multi-view constrained cooperative routing problem with observation completeness requirements. Each citrus tree is modeled as a multi-view observation unit rather than a single visiting target.

The repository contains seed-level algorithm results, summary tables, ablation data, convergence data, statistical test outputs, execution proxy metrics, parameter sensitivity analysis outputs, processed real-tree-location case data, representative example inputs, and supplementary materials used in the manuscript.

## Repository structure

- `code/`: scripts for algorithm execution, evaluation, plotting, statistical analysis, scenario generation, and parameter sensitivity analysis.
- `data/`: input data, including processed real-tree-location data, example inputs, and notes on seed-based random instance generation.
- `results/`: seed-level results, summary tables, ablation results, convergence data, execution proxy metrics, statistical tests, and parameter sensitivity analysis outputs.
- `supplementary/`: supplementary files and materials related to the manuscript.
- `docs/`: additional documentation.
- `requirements.txt`: Python dependency list for reproducing the analysis scripts.

The folder `data/random_instances/` provides notes on seed-based random orchard instance generation. The matched random seeds used in the manuscript are 42-61.

## Main experimental settings

- Orchard dimensions: 120 m x 120 m
- Number of citrus trees: 80
- Number of obstacles: 11
- Number of UGVs: 4
- Random seeds: 42-61
- Number of repeated runs: 20
- Theoretical observation poses per tree: 4

## Requirements

The scripts were developed for Python 3.10 or later. Python 3.11 is recommended.

The required Python packages are listed in `requirements.txt`. Install them from the repository root with:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `openpyxl`
- `Pillow`

## Data-to-manuscript mapping

| Manuscript item | Repository file |
|---|---|
| Main algorithm comparison results | `results/summary_tables/main_comparison_summary_numeric.csv` |
| Headline improvements in Abstract | `results/summary_tables/headline_improvements.csv` |
| Table 1 | `results/summary_tables/main_comparison_summary_numeric.csv` |
| Table 2 | `results/ablation_results/table1_ablation_summary.csv` |
| Table S4 | `results/execution_proxy_results/execution_proxy_metrics_summary.csv` |
| Table S5a | `results/statistical_tests/friedman_results.csv` |
| Table S5b | `results/statistical_tests/wilcoxon_holm_results.csv` |
| Table S6 | `data/real_tree_locations/real_orchard_case_metrics.csv` |
| Table S7 | `results/summary_tables/table_S7_full_algorithm_comparison.csv` |
| Table S8 | `results/parameter_sensitivity/table_S8_parameter_sensitivity.csv` |
| Main convergence results | `results/convergence_results/main_convergence_summary_1500iters.csv` |
| Parameter sensitivity analysis | `results/parameter_sensitivity/` and `code/sensitivity_analysis/` |
| Real-tree-location case data | `data/real_tree_locations/` |
| Example inputs | `data/example_inputs/` |

## Code organization

- `code/evaluation/`: main experimental scripts, summary-generation scripts, and execution-proxy post-processing scripts.
- `code/plotting/`: plotting scripts for convergence curves, representative path visualization, and real-orchard case visualization.
- `code/statistics/`: scripts for Friedman and Wilcoxon-Holm statistical tests.
- `code/scenario_generation/`: optional utilities for generating or documenting scenario inputs.
- `code/sensitivity_analysis/`: release scripts for reproducing the F-LCA parameter sensitivity analysis.

## Results organization

- `results/seed_results/`: seed-level results for F-LCA, SISR-VRP, ACO, GA, and PSO.
- `results/summary_tables/`: manuscript-ready summary tables and headline improvement calculations.
- `results/ablation_results/`: seed-level and summary results for F-LCA ablation variants.
- `results/convergence_results/`: convergence summary data for iterative algorithms.
- `results/execution_proxy_results/`: execution-oriented proxy evaluation results using a unified post-processing protocol.
- `results/statistical_tests/`: Friedman and Wilcoxon-Holm statistical test outputs.
- `results/parameter_sensitivity/`: cleaned results and source tables for the F-LCA parameter sensitivity analysis.

## Reproducing summary tables and selected analyses

From the repository root, the following lightweight scripts can be used to regenerate the manuscript-ready summary tables, statistical-test outputs, and selected convergence or real-tree-location analyses:

```bash
python code/evaluation/generate_main_summary.py
python code/evaluation/generate_ablation_summary.py
python code/statistics/generate_statistical_tests.py
python code/plotting/plot_convergence_all_algorithms.py
python code/plotting/draw_real_orchard_fig5_local.py
```

Generated files are saved under `results/` or to the output paths defined in the corresponding scripts.

The full 20-seed algorithm runs can be time-consuming. Seed-level results used in the manuscript are already provided in `results/seed_results/`.

## Re-running full experiments

The main fair-budget experimental scripts are provided under `code/evaluation/`. They can be used to rerun the algorithm comparisons if full recomputation is required.

Representative scripts include:

```bash
python code/evaluation/run_flca_20seed_fair_budget.py
python code/evaluation/run_aco_20seed_fair_budget.py
python code/evaluation/run_ga_20seed_fair_budget.py
python code/evaluation/run_pso_20seed_fair_budget.py
python code/evaluation/run_sisr_vrp_20seed_fair_budget.py
```

These scripts may require substantial runtime because they rerun 20 matched random seeds for each algorithm.

## Notes on statistical tests

The file `wilcoxon_holm_results.csv` provides the full pairwise Wilcoxon-Holm results, including Total Cost. Supplementary Table S5b reports the main pairwise metrics discussed in the manuscript.

## Notes on parameter sensitivity analysis

The parameter sensitivity analysis evaluates F-LCA under single-factor parameter perturbations. For each tested parameter, only one parameter is changed at a time while the remaining parameters are kept at their baseline settings. The same 20 matched random instances used in the main experiment are used for evaluation.

The cleaned result files provided in this repository contain all metrics required to reproduce the manuscript tables and supplementary tables. Auxiliary logging columns not used for analysis were removed for clarity.

## Notes on real-tree-location data

The real-tree-location case uses processed local planar coordinates derived from KML-exported citrus tree markers. These processed coordinates are provided to support reproducibility of the real-tree-location planning case.

## Notes on random instances

The random orchard instances used in the manuscript are generated by the scenario generation functions in the code directory. The matched random seeds are 42-61. Each seed defines one orchard instance with 80 citrus trees, 11 static circular obstacles, and 4 UGVs. Seed-level algorithm outputs are provided in `results/seed_results/`.

## Data and code availability

All code, processed input data, seed-level results, summary tables, statistical-test outputs, parameter sensitivity results, execution proxy metrics, and supplementary materials required to reproduce the manuscript tables and computational results are provided in this repository.

A permanent archival record is provided through Zenodo.

Zenodo DOI: `10.5281/zenodo.20372354`

Repository record: https://doi.org/10.5281/zenodo.20372354

## Citation

If you use this repository, please cite the associated manuscript and the archived Zenodo record.

```bibtex
@misc{flca_orchard_multiview_planning_2026,
  title  = {F-LCA Orchard Multi-View Path Planning},
  author = {Cui, Hengzhang and Jiang, Rui},
  year   = {2026},
  doi    = {10.5281/zenodo.20372354},
  url    = {https://doi.org/10.5281/zenodo.20372354}
}
```

## License

Unless otherwise stated, the code in this repository is released under the MIT License, and the data, supplementary materials, and documentation are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.

If a different license is required by the authors, institution, or target journal, this section should be updated before public release.

