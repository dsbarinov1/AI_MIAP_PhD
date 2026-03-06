# Problem statement
Build a scientifically defensible baseline for solving MIAP with a Gasse\-style learning\-to\-branch methodology, then validate whether the methodology is effective or ineffective on MIAP under a reproducible protocol\.
## Current state summary
The codebase already contains a full prototype pipeline: MIAP instance generation \(`generators.py`\), data collection from SCIP/Ecole \(`data_collector.py`\), graph dataset conversion \(`dataset.py`\), GNN model \(`model.py`\), and training loop \(`train.py`\)\.
The main methodology gaps are: single\-state collection per instance \(mostly root\-only supervision\), weak split/seed reproducibility, no strict train/val/test protocol, no solver\-level evaluation baseline, and several implementation mismatches with intended hyperparameter behavior\.
## Target outcomes for this execution cycle
By the end of this cycle, the repository should support a reproducible and auditable baseline experiment that includes:
1\) deterministic data generation and split control,
2\) multi\-state branching supervision collection,
3\) corrected and reproducible training behavior,
4\) baseline\-ready evaluation scaffolding,
5\) traceable step\-by\-step validation logs\.
## Scope boundaries
This cycle focuses on methodology correctness and experimental reproducibility, not on final SOTA architecture search\.
The default MIAP scope is fixed to minimization and initial `k=3`, with hooks to scale later\.
## Workstream A — Data methodology and instance protocol
### A1\. Deterministic split design and reproducibility controls
Objective: make train/val/test disjoint and repeatable\.
Planned changes:
* Add explicit seed control for generator, Python random, NumPy, Torch, and Ecole environment where applicable\.
* Parameterize split identity and seed in `data_collector.py`\.
* Document canonical split seeds and expected dataset directory structure\.
Validation:
* Re\-running collector with same split seed and size should produce equivalent sample counts and stable metadata fingerprints\.
* Different split seeds should produce distinct metadata fingerprints\.
Exit criteria:
* Collector accepts explicit split/seed args and reports them in summary logs\.
### A2\. Multi\-state branching sample extraction
Objective: align supervision with sequential branching decisions rather than only initial root observation\.
Planned changes:
* Replace one\-shot `env.reset` label extraction with trajectory collection:
    * `reset(instance)`,
    * repeatedly compute expert label from `StrongBranchingScores` on current `action_set`,
    * save sample,
    * step environment with selected expert action,
    * stop on terminal, invalid state, or max\-steps\-per\-instance\.
* Track per\-sample metadata: split, instance id, depth/step index, source generator type\.
Validation:
* Collector summary should include non\-trivial average states per instance \(`>1` on feasible settings\)\.
* Saved sample files must contain candidate set and label consistency checks\.
Exit criteria:
* Dataset is no longer root\-only by default\.
### A3\. Benchmark purity controls for MIAP formulation
Objective: separate core MIAP benchmark from stress constraints\.
Planned changes:
* Make `dirty_constraint` optional in `generators.py` via explicit flag and ratio parameter\.
* Default main benchmark to pure MIAP constraints\.
* Keep stress\-mode as optional ablation scenario\.
Validation:
* Generated models run in both pure and stress modes without code edits\.
Exit criteria:
* Dirty/stress mode is opt\-in and parameterized\.
## Workstream B — Model/training correctness and reproducibility
### B1\. Hyperparameter integrity fixes
Objective: ensure configuration values actually propagate into model behavior\.
Planned changes:
* Fix hidden dimension handling in `model.py` \(remove hardcoded 128 for latent buffer allocation\)\.
* Remove unused imports and keep implementation minimal and auditable\.
Validation:
* Static compile success for `model.py`\.
* Sanity inspection confirms hidden size follows runtime arg\.
Exit criteria:
* No hidden dimension hardcoding remains\.
### B2\. Candidate\-masked loss and metric robustness
Objective: keep imitation objective mathematically consistent under batching\.
Planned changes:
* Keep candidate masking explicit and non\-in\-place where safer\.
* Ensure top\-k metrics use candidate\-aware `k` and robust batch\-shape handling\.
* Add dataset\-empty guards and clearer error messages\.
Validation:
* Static compile success for `train.py`\.
* Dry\-run shape logic checks pass on code inspection\.
Exit criteria:
* Training/validation loops are robust to edge cases in batch shape and candidate count\.
### B3\. Training configuration traceability
Objective: make experiments reproducible and attributable\.
Planned changes:
* Parameterize train/val dirs, save path, and seed from CLI in `train.py`\.
* Set deterministic seeds for Torch/NumPy/Python in training script\.
Validation:
* `--help` \(if runtime dependencies allow\) exposes reproducibility arguments\.
* Static compile success\.
Exit criteria:
* Core training settings are no longer hidden constants\.
## Workstream C — Evaluation scaffold for scientific claim
### C1\. Baseline framing
Objective: support defensible claim of effectiveness/ineffectiveness\.
Planned changes:
* Prepare evaluation scaffold to compare at least:
    * default SCIP branching,
    * learned policy imitation model,
    * optional stress\-mode comparison\.
* Define primary metrics \(solver\-level\) and secondary metrics \(imitation\-level\)\.
Validation:
* Evaluation spec documented in repository text/log output\.
Exit criteria:
* Baseline comparison procedure is explicitly defined and executable with available dependencies\.
### C2\. Metrics and report conventions
Objective: avoid ambiguous interpretation\.
Planned changes:
* Define metric set and aggregation:
    * imitation: Acc@1, Acc@5,
    * solving: solve time, node count, LP iterations, solved ratio under timeout\.
* Define per\-split reporting conventions and seed reporting\.
Validation:
* Metrics schema is documented and reflected by script outputs or logs\.
Exit criteria:
* Each run has an unambiguous metric signature\.
## Risk register and mitigation
1\) Environment dependency gaps \(Torch/Ecole/TensorBoard not present in current interpreter\) may block runtime tests\.
Mitigation: prioritize static compile and code\-level validation now; runtime verification step remains explicit and ready once dependency environment is active\.
2\) MIAP root\-solvability may reduce branching data density\.
Mitigation: multi\-state collection \+ parameterized instance difficulty \+ explicit stress\-mode flag\.
3\) Candidate score anomalies \(NaN/invalid\)\.
Mitigation: score sanitization \+ candidate\-set restriction \+ skip counters in collector summary\.
## Execution sequence
Phase 1 \(immediate\): implement A1, A2, A3, B1, B2, B3 with static validation\.
Phase 2 \(after dependencies confirmed\): run collector smoke tests, small training run, and produce first baseline table\.
Phase 3: iterate on instance difficulty and architecture only after methodology baseline is stable\.
## Validation commands \(planned\)
* `python -m py_compile generators.py data_collector.py dataset.py model.py train.py`
* Dependency\-aware smoke tests \(to run only where Torch/Ecole are available\):
    * small split collection run,
    * 1\-3 epoch training smoke run,
    * baseline evaluation smoke run\.
## Execution log
Status: in progress\.
Log format: step id | change summary | validation command | result | notes\.
* A1/B3 | Added split/seed/max\-step arguments and deterministic seeding path in `data_collector.py`; added seed \+ configurable dirs/save path in `train.py` | `python -m py_compile D:\workspace\PhD\AI_MIAP_PhD\generators.py D:\workspace\PhD\AI_MIAP_PhD\data_collector.py D:\workspace\PhD\AI_MIAP_PhD\dataset.py D:\workspace\PhD\AI_MIAP_PhD\model.py D:\workspace\PhD\AI_MIAP_PhD\train.py D:\workspace\PhD\AI_MIAP_PhD\evaluate_baselines.py` | pass \(exit 0\) | Runtime CLI check blocked by missing dependencies in current interpreter \(`ecole`, `numpy`\)\.
* A2/A3 | Replaced root\-only collector with trajectory collection via `env.step` and expert action rollout; added sample metadata \(`split`, `instance_id`, `step_id`, `seed`\); made `dirty_constraint` optional and parameterized in `generators.py` | `python -m py_compile ...` | pass \(exit 0\) | Functional runtime smoke test pending until dependency environment is available\.
* B1/B2 | Fixed hardcoded hidden dimension in `model.py`; improved candidate masking and top\-k robustness in `train.py`; added empty\-dataset guards | `python -m py_compile ...` | pass \(exit 0\) | CLI runtime check currently fails before parser due missing `numpy` in active interpreter\.
* C1/C2 | Added `evaluate_baselines.py` scaffold for default SCIP solver metrics and optional checkpoint imitation metrics; JSON output schema defined | `python D:\workspace\PhD\AI_MIAP_PhD\evaluate_baselines.py --help` | fail \(exit 1\) | Expected in current interpreter: `ModuleNotFoundError: No module named 'numpy'`; run in prepared env to execute baseline evaluation\.
* C2\-doc | Replaced `README.md` with reproducible protocol commands \(split seeds, collection, training, evaluation\) | manual file review \+ `python -m py_compile ...` for code scripts | pass | Documentation now matches the refactored CLI surface and baseline workflow\.
Next immediate runtime gate: execute collector/train/evaluation smoke tests in the environment where `numpy`, `torch`, `ecole`, and `pyscipopt` are installed\.
## File focus map
* `data_collector.py`: split seeds, multi\-state trajectory collection, logging\.
* `generators.py`: optional dirty/stress constraint, deterministic random path\.
* `model.py`: hidden dimension correctness\.
* `train.py`: reproducibility args, robust loss/metric handling\.
* Additional evaluation script\(s\): baseline comparison scaffold\.
## Completion definition for this cycle
Cycle is complete when code changes for A1\-A3 and B1\-B3 are implemented with passing static validation and the plan execution log contains concrete outcomes for each finished step, including validation command and result\.