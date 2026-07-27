# Sealed evaluation sets

## `sealed_final_v1.json` — DO NOT GENERATE, TRAIN, TUNE OR EVALUATE ON THIS

20 HM3D **val-split** scenes never touched by any dataset, model or diagnostic.
Reserved for the single final evaluation of Phase 8 **v1** and beyond.

It exists because the Phase 8 v0 test split is now **consumed**: its results
have been read, and any v1 design informed by them makes that split biased for
v1. A held-out number is only unbiased the first time it is looked at.

Rules:
- No dataset generation from these scenes until v1 is frozen.
- No checkpoint selection, threshold tuning or architecture choice may reference
  them.
- Evaluate exactly once, then mark consumed like v0.

## `dev_pool_v1.json` — development pool

103 HM3D train-split annotated scenes, untouched so far. Use these to build v1
train/validation splits. Diagnosis, redesign and tuning all happen here.

## Consumed

| set | scenes | status |
| --- | --- | --- |
| `full.json` train/val/test (realised) | 42 | **consumed by Phase 8 v0** |
| `pilot_debug.json` (realised) | 16 | consumed, pipeline development only |
| `sealed_final_v1.json` | 20 | **SEALED** |
| `dev_pool_v1.json` | 103 | available for v1 development |

`archive/phase8_v0/consumed_scenes.json` lists every scene touched to date.
