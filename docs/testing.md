# Testing

Every pytest test is weightless: tests never use the Official Music 3 checkpoint,
network access, downloaded tokenizer assets, or other local model files. The suite
is split by component boundary and execution cost. A test belongs to the lowest
tier that can verify the behavior without weakening its contract.

| Tier | Location | Contract | Default cadence |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | One module or behavior in isolation, using small arrays, generated inputs, or narrow stubs | During implementation, then on every push, pull request, and release verification |
| Weightless integration | `tests/integration/` | Multiple real components using runtime-generated synthetic checkpoints and persisted metadata contracts | After affected unit tests, then on every push, pull request, and release verification |

Run the default gates in order so a cheap, focused failure stops before the
cross-component checks:

```sh
uv run pytest -q tests/unit
uv run pytest -q tests/integration
```

Plain `uv run pytest -q` discovers the same two default tiers. GitHub CI keeps
them as separate steps for clear failure attribution without running any test
twice. The workflow runs automatically for pull requests and pushes to `main`,
and it can also be started manually.

`make check` is the local release gate. It runs Ruff, both pytest tiers, the
public-tree validator, and a source-only package build in that order.

Generation-checkpoint tests use temporary directories and synthetic MLX arrays.
They cover atomic round trips, corruption fallback, acoustic-prefix resume,
restored progress, stage timing, and model-load omission without model weights.

The golden integration fixture is versioned in
`tests/fixtures/music3_golden_v1.json`. It persists the miniature model contract,
inference inputs, expected numerical outputs, topology digests, and tolerances.
Its deterministic dense and selective-q8 SafeTensors files are materialized only
inside pytest's temporary directory.

`tests/fixtures/music3_official_schema_v1.json` separately persists the Official
Music 3 configs, all 982 mapped-parameter topology digests, and representative
source-to-MLX mapping cases. It contains metadata only, so the default integration
tier continues to verify the real model contract after the local source weights
are removed.

Official checkpoints may be consulted outside pytest when initially calibrating
or intentionally revising one of these persisted contracts. Only the minimal,
reviewable metadata or numerical reference vectors belong in the test suite. A
fixture update must explain the intended contract change and must never be an
automatic response to a failing golden assertion.

Reference conditioning is covered by both tiers with miniature models, the real
sampler, and the real loop. Miniature logits are far more peaked than the released
checkpoint's, so the guidance tests pin both ends of the calibrated bias instead of
asserting the released value's behavior. A recording sampler pins the width of the
semantic top-k window on free and guided frames. A null-reference identity test
cannot do that, because both of its runs share the same window. A miniature model
whose semantic head rows repeat in groups of four puts a tie on the window
boundary of every frame, which pins the tie rule of the window and the draw.

`dev/verify_reference_conditioning.py` repeats the comparisons against a local
checkpoint when a change touches semantic sampling. It prints a JSON report and is
run by hand, never from pytest. The released bias is calibrated with its sweep:
`--interval 1 --penalties 8,12,13,14,15,16,20,24,32` guides every frame towards a
plausible reference, the semantic codes of a baseline rendered from
`--plausible-caption` and `--plausible-lyrics`, and towards an implausible one,
the baseline shifted by 4,096 codes. The script defaults to `--interval 4`.
`--reference-npz` adds an encoder stream saved as int32 `codes` `[frames, 8]` and
`semantic_candidates` `[frames, k]` arrays; a guided frame follows it when the
emitted code is one of that frame's candidates.

Listening validation with complete dense or quantized weights is a separate
release-quality activity. It must not be represented as a pytest pass/fail check
or added to the default CI suite.
