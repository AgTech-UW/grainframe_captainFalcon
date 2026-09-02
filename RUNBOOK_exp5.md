# Exp5 publication runbook

Everything between the current draft and a submittable section, in order.
All commands from the repository root. Every long run supports `--resume`,
so use it on anything unattended. Time estimates assume roughly 5 to 8
seconds per run at small squadrons and up to 20 at twelve wingmen.

## Before anything

The code now differs from every archived result (new SEP_PROFILE field,
arrival logging), so the config hash has moved and **every number destined
for the paper regenerates below**. Archive the old CSVs first so nothing
mixes:

    mkdir -p results/archive_preRevision
    mv results/exp5_grid.csv results/exp5_capacity.csv \
       results/exp5_study.csv results/exp5_tune.csv \
       results/archive_preRevision/

Two smoke-test files may exist from validation, `results/exp5_ablation.csv`
and `results/exp5_capacity2.csv` with 2-seed rows. Leave them: the real
runs below use `--resume` semantics on the same seeds, and a fresh run
without `--resume` overwrites them anyway.

## 1. The grid, flat profile, including the sub-unity cells (~3 h)

Decides whether the threshold at delta = P exists. This is contribution 2.

    python experiments/exp5_finish.py grid2 \
        --prs 8,10,13,16,20 --ratios 0.5,0.75,1.0,1.25,1.5,2.0 \
        --sizes 3,4 --seeds 30 --tag flat --resume

Feeds: Fig. 4 (point exp5_grid_figure.py at results/exp5_grid2_flat.csv),
the marginals quoted in Sec. 5.4, and the threshold claim in the opener.

## 2. The grid under the normalized profile (~3 h)

Decides whether the threshold is a property of the vehicles or of the flat
profile's cutoff. This is the caveat now recorded in Limitations, and it is
the difference between a finding and an artifact.

    python experiments/exp5_finish.py grid2 \
        --prs 8,10,13,16,20 --ratios 0.5,0.75,1.0,1.25,1.5,2.0 \
        --sizes 3,4 --seeds 30 --set SEP_PROFILE=norm --tag norm --resume

Read the two grids together. Threshold in both: it belongs to the slots
and the vehicles. Threshold only under flat: it belongs to the cutoff, and
the spacing claim must be restated. Note the norm profile weakens
long-range repulsion by design, so absolute collision levels may differ
between grids; the question is where the cliff sits, not the levels.

## 3. The ablation (~1.5 h)

Contribution 1 currently has no ablation and a reviewer will ask. Collision
AND arrival per variant, because noswirl and noorder are expected to fail
by deadlock, which collision counts cannot see. The collFree&noArrive
column of the report is that outcome directly.

    python experiments/exp5_finish.py ablation \
        --sizes 4,8 --seeds 50 --resume
    python experiments/exp5_finish.py report --csv results/exp5_ablation.csv

Feeds: a new small table in Sec. 5.2 or 5.5, full scheme against noswirl,
noorder, nocorridor, nobrake.

## 4. Table 1, both columns, one hash each (~2 h)

    python experiments/exp5_finish.py capacity2 \
        --sizes 3,4,6,8,12 --seeds 50 --resume
    python experiments/exp5_finish.py report --csv results/exp5_capacity2.csv

The violated column is the satisfied configuration with SLOT_SPACING
dropped from 25 to 15 (delta/P = 0.75) and nothing else changed, so the
columns differ in exactly the quantity the condition is about. That is a
cleaner contrast than the old violated config, which differed in many
things at once. Adjust with --violated-spacing if you want it harsher.

Feeds: Table 1 wholesale, plus arrival columns if you take my suggestion
to report them there. Replaces the XXX about provenance.

## 5. The half-radius grid (~2 h)

Decides whether the gradient in P is really about the turning radius, the
sizing claim in the mechanism paragraph. Sub-unity cells are not needed
here, only the knee.

    python experiments/exp5_finish.py grid2 \
        --prs 4,6,8,10,13,16,20 --ratios 1.0,1.25,1.5,2.0 \
        --sizes 3,4 --seeds 30 --set RboidFactor=0.5 --tag halfR --resume

Baseline knee sits between P = 13 and 16 at R = 5. If the account is
right, this grid's knee should sit near half that; the extra low P values
are there so it has somewhere to appear. If it does not move, the P
gradient belongs to some other fixed length in the scene and the mechanism
paragraph needs revisiting.

## 6. Guidance study rerun (~1 h)

Sec. 5.1's numbers were verified against the archived study, but that file
is now under a stale hash, so regenerate for a consistent set:

    python experiments/exp5_study.py sweep

## 7. Optional, overnight

Sensitivity under the satisfied regime, closing the loop on the rescoped
sentence in Sec. 5.4. The old study ran in the broken regime at ceiling.
The current experiment file IS the satisfied configuration, so plain:

    python experiments/exp5_tune.py sensitivity --nFan 10 \
        --tune-seeds 30 --holdout-seeds 30 --resume

Reeds-Shepp reverse motion for wingmen remains unimplemented, and per the
revised Sec. 5.4 it now tests only the margin, not the threshold, so it is
defensible to leave it as future work if the deadline is close.

## What changed in the code

    grainframe/config.py          SEP_PROFILE field on SimOptions
    grainframe/flocking.py        'norm' separation profile, weight applied
                                  to the swirl as well as the radial term
    experiments/exp5_priority.py  run() returns per-leader arrival ('done')
    experiments/exp5_study.py     analyse() records redArrived/goldArrived,
                                  FIELDS extended
    experiments/exp5_tune.py      FIELDS extended to match
    experiments/exp5_finish.py    NEW: ablation, capacity2, grid2 with
                                  --set overrides, report with arrival and
                                  the collision-free-but-never-arrived count

Old CSVs lack the arrival columns; the report prints "not recorded" for
them rather than guessing.
