# sample_vla

A task template, not a migrated model. It holds the structure source of truth a single-backbone,
single-action-head VLA produces, so that F3, F4 and F5 can be exercised and reviewed before an
actual repository is in hand. Copy the directory, rename the task, and replace the `src` paths
and the source repository in `anatomy/STEP_ANATOMY.yaml` with the real ones.

## Goal

Migrate the original VLA model into `loongforge/embodied/` so that it has the same structure as
the models already there — a dataset, a model and a trainer in the framework's own layout — with
identical logic and aligned loss.

## Acceptance

All four gates, in order; a failure returns to the function that owns it:

- coverage: every module the original has lands on a slot (frozen structure judge)
- structure: slot and target file correspond exactly, no missing slot and no surplus file
- numeric: per-step loss deviation max, p95 and mean each at most one percent, not amplifying
- self-contained: the target project starts training with no dependency on the original repository

## Reuse before rewrite

The slots that land on the target framework's shared files — `model/registry.py`,
`data/datasets/compute_stats.py`, `optimizer/`, `distributed/` — use `wrap` or `adapt`, never
`copy` or `reimplement`. Those files carry every other embodied model; the migration adds to them.
A backbone, optimizer, schedule, parallel strategy or checkpoint layer the framework already
implements is reused through a wrapper that keeps the original interface, so there is one
implementation rather than two.

## Run

```bash
cd loongforge/embodied/rsi_workflow/model_migration
TASK=work/sample_vla
ANATOMY=examples/sample_vla/anatomy/STEP_ANATOMY.yaml

python -m validate freeze                                    # lock the judges first
python core/stages/f3_scaffold/scaffold.py --anatomy $ANATOMY --target $TASK/target
python core/stages/f3_scaffold/check_tree.py --anatomy $ANATOMY --target $TASK/target
python core/stages/f4_migrate/map/build_mapping.py build --anatomy $ANATOMY \
    --mapping $TASK/mapping/mapping.yaml
python core/stages/f4_migrate/migrate.py --anatomy $ANATOMY --mapping $TASK/mapping/mapping.yaml \
    --source-repo <original repo> --target $TASK/target --work-log $TASK/logs/WORK_LOG.md
python core/stages/f4_migrate/weights/convert_weights.py --src-ckpt <ckpt> \
    --dst-ckpt $TASK/target/weights.pt --out $TASK/weights_report.json
python -m validate judge --anatomy $ANATOMY --target $TASK/target \
    --weights $TASK/weights_report.json
python core/stages/f5_align/align_driver.py --adapters examples/sample_vla/adapters.json \
    --workdir $TASK
```

`align_driver.py` stages a skill request and exits 4 the first time: load the
`precision-alignment` skill, answer the request file, and rerun.
