# Changelog

All notable project changes are documented here.

## 0.0.1a0 - Unreleased

- Establish the pure-MLX project contract.
- Add architecture, porting, weight, and reference documentation.
- Implement the checkpoint-native tokenizer and pure-MLX model pipeline.
- Add strict dense conversion and selective affine-q8 conversion with manifests.
- Add phase-scoped model residency with MLX, process-footprint, and swap checks.
- Add native 44.1 kHz stereo PCM16 WAV output and the public Python API.
- Add guarded instrumental prompt construction to prevent tag-only conditioning.
- Align autoregressive seed derivation, c0 column order, and Gumbel-max sampling
  with the SGLang reference.
- Correct the acoustic carry window from 86 to 172 latent frames.
- Keep selective q8 experimental while multi-seed listening validation is in
  progress.
- Add a pure-MLX source-to-dense tensor and digest verifier.
- Enforce the MLX-only runtime import and dependency boundary in tests.
- Include third-party notices and the Apache License 2.0 in wheel and source
  distributions.
- Verify the release tag with the official MLX CPU backend and smoke-test the
  built wheel and source distribution before Trusted Publishing can begin.
- Add package metadata, unit tests, and public-tree validation.
- Separate unit and weightless golden integration test tiers with explicit CI
  cadence and a metadata-only Official topology contract.
- Add a PyPI trusted-publishing workflow for GitHub pre-releases.
- Add optional reference-code conditioning for the semantic codebook. Guidance
  biases the draw with a finite logit penalty calibrated against the selective-q8
  checkpoint, cover restricts each covered frame to its reference code, and
  continue injects whole reference frames as context that stays out of the result.
  A live reference window masks the audio-end token. A text-only request keeps its
  previous output for the same seed.
- Enforce the 500-line limit on runtime modules in the unit tests.
- Evaluate each `CONTINUE` prefix frame as it is prefilled. A long prefix no
  longer builds one lazy graph that is first evaluated at the first generated
  frame and exhausts unified memory.
- Fix the semantic top-k window, which kept only the conditional argmax and the
  codes tied with it, because `mx.topk` does not return its values in sorted
  order. `top_k` and the CFG scale now shape the c0 draw, so a request draws
  different c0 codes than before for the same seed. `restrict_top_k` uses the same
  order-independent threshold. Generation checkpoints move to behavior version
  `music3-resume-v2`, so a checkpoint written by the old window is recomputed.
- Keep every column tied with the `top_k`-th largest value as a candidate of the
  seeded draw, as SGLang's sampler does. Output changes only where guided or
  residual logits tie at that boundary. Generation checkpoints stay at behavior
  version `music3-resume-v2`.
- Size a guided c0 draw to every reachable column. A tie can widen the model's
  window beyond `top_k`, and a draw of `top_k` plus the candidate count then
  dropped window columns or reference candidates before the sampler saw them.
- Add `GUIDANCE_LOGIT_PENALTY` to the generation fingerprint, so a checkpoint
  written under another penalty is recomputed without a behavior-version bump.
- `GUIDANCE_LOGIT_PENALTY` stays at 16 after a sweep against the full window. The
  sweep guided every frame of 8 s intros on the selective-q8 checkpoint, over two
  prompts and two seeds each. Of 800 guided frames each, 16 follows 778
  plausible, 192 implausible, and 747 encoded-stream frames, and 15 follows 776,
  134, and 683. Frames within a run are correlated, and the runs do not separate
  the two values. No value from 8 to 32 follows at least 80 % of plausible and at
  most 10 % of implausible frames. Under the old window, 8, 16, 24, and 32 units
  followed 8, 22, 24, and 25 of 25 plausible and 0, 1, 13, and 17 of 25
  implausible guided frames.
- Guidance at a `reference_interval` above 1 now steers weakly. At interval 4 and
  penalty 16, guided frames follow 24 % of plausible, 13 % of implausible, and
  49.5 % of encoded-stream references. At penalty 20, plausible and implausible
  both reach 52 %.
- Report progress during a `CONTINUE` prefill. A 1,500-frame prefix took about
  47 s on the selective-q8 checkpoint without a report. `GenerationProgress` gains
  `prefilled_frames` and `prefix_frames`, both zero by default. Each evaluated
  prefix frame reports `completed_frames=0` and the prefix frames evaluated so
  far. Generated-frame reports have `prefilled_frames` equal to `prefix_frames`. A
  caller that reads only `completed_frames` receives one report at zero per prefix
  frame.
