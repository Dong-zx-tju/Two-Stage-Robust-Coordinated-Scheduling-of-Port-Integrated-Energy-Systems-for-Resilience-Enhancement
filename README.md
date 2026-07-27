# C&CG convergence audit package

This package exposes the parameter-independent implementation of the
column-and-constraint generation (C&CG) procedure used in the manuscript and
the complete saved iteration record underlying the reported six-iteration
convergence result.

## What is included

- `src/ccg_core.m`: generic C&CG master/subproblem loop, bound updates, scenario
  generation and stopping rule.
- `src/budget_extreme_point.m`: exact extreme-point representation of a
  continuous budgeted uncertainty set for an integer uncertainty budget.
- `src/signed_bigm_product.m`: signed big-M formulation for the product of a
  binary extreme-point indicator and a free continuous dual multiplier.
- `src/verify_convergence_log.m`: independently recomputes the gaps and checks
  the stopping rule from the released log.
- `src/plot_convergence_log.m`: regenerates the convergence plot from the log.
- `logs/ccg_convergence_complete.xlsx`: original saved MATLAB workbook.
- `logs/ccg_convergence_complete.csv`: machine-readable copy of the iteration
  sheet.
- `logs/ccg_convergence_complete.txt`: human-readable iteration-by-iteration
  record.
- `figures/ccg_convergence_reported.pdf` and `.png`: reported convergence
  figure.
- `reviewer_response_code_availability.tex`: response-letter wording that
  accurately describes the release scope.

## Convergence definition

At iteration `k`, the implementation updates the incumbent upper bound and
computes

```text
UB(k)  = min(UB(k-1), first_stage_cost(k) + worst_recourse_cost(k))
gap(k) = max(UB(k) - LB(k), 0)
```

The stopping criterion is `gap(k) <= 1e-5`. At iteration 6, the saved values
are

```text
LB = 89876.74771736123
UB = 89876.74771736082
```

The raw floating-point difference is approximately `-4.1e-10`; the reported
nonnegative gap is therefore zero. This is a numerical zero under the stated
stopping rule, not a claim of symbolic equality.

## Reproduce the audit

In MATLAB, change to the `src` directory and run:

```matlab
verify_convergence_log
plot_convergence_log
```

The first command checks every logged gap and the final convergence decision.
The second command writes a regenerated PDF and PNG to `figures/reproduced`.
Neither command requires the undisclosed port case data or a mathematical
programming solver.

The files `budget_extreme_point.m` and `signed_bigm_product.m` use YALMIP
objects and are included to disclose the exact subproblem reformulation. The
generic `ccg_core.m` accepts master- and subproblem-solver callbacks and does
not contain case-specific model data.

## Release boundary

This is a minimal convergence-audit and algorithm-disclosure package, not the
complete engineering project. It intentionally excludes the port-specific
input time series, equipment capacities, economic coefficients, complete
master/subproblem matrix assembly, and post-processing code unrelated to the
C&CG convergence claim. These exclusions are listed explicitly in
`DISCLOSURE_SCOPE.md`.

The package should therefore be described as a release of the core C&CG
implementation and complete convergence logs. It should not be described as a
release of the full case-study source code.


