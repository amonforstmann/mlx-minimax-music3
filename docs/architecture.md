# Architecture

This document defines the intended pure-MLX runtime. Values are taken from the
released checkpoint configuration and current reference implementations; every
value that affects tensor layout must be verified against the pinned upstream
revision before implementation.

## Model stages

MiniMax Music 3 is a lyrics- and caption-conditioned music generator with two
major generation phases.

### Autoregressive representation

The first phase produces music frames at 25 frames per second:

1. A Qwen3-based global language model predicts the semantic codebook token for
   each frame and carries long-range musical structure.
2. A local depth decoder predicts seven residual acoustic codebooks, conditioned
   sequentially on the global token and preceding local tokens.
3. The runtime retains the global and local hidden states needed by acoustic
   synthesis. Discrete tokens alone are not sufficient for waveform generation.

The released model uses eight Residual Vector Quantization (RVQ) codebooks per
frame. The first codebook has 16,384 entries; each of the remaining seven has
1,024 entries.

#### Reference conditioning

A request may carry a reference code stream, one frame of RVQ codes per generated
frame, with optional semantic alternates per frame. An encoder emits every
codebook of every frame. A stream captured from an earlier generation's semantic
head carries the first codebook alone.

The stream steers only the first codebook. `GUIDANCE` subtracts a fixed penalty
from every non-candidate column on each interval-th covered frame, so the draw is
biased but a non-candidate code stays reachable. `COVER` leaves only the reference
column reachable on every covered frame. `CONTINUE` injects whole reference frames
as context before the first emitted frame, then frees the loop.

Under `GUIDANCE` and `COVER`, residual codebooks and hidden states are
model-generated, so the acoustic stage receives the conditioning it receives for a
text-only request. A guided draw covers the model's own top-k window plus every
reference candidate, so a candidate outside that window is still reachable. A
covered draw reaches the reference code alone.

A `CONTINUE` prefix never enters the result. It extends the key-value cache beyond
the requested duration and needs every codebook: a semantic-only stream makes the
depth decoder resynthesize the residual codes, which is reported as a quality
warning. Because the prefix is context, the requested duration always counts new
frames. Each prefix frame costs one evaluated language-model step, about 32 ms on
the selective-q8 checkpoint, and progress reports start only after the prefix.

A live reference window masks the stop token, so the model cannot end the song
before or inside the reference. Two limits remain. A reference longer than the
requested duration keeps its tail unused, reported as a quality warning. The
reference never covers the loop's first frame, which is feedback for
`<|audio_start|>` and is not part of the result.

Without a stream the autoregressive loop is byte-identical to a text-only run.

### Acoustic synthesis

The second phase converts fused hidden states into waveform audio:

1. A condition encoder projects autoregressive hidden states into acoustic
   conditioning.
2. A flow-matching Diffusion Transformer (DiT) predicts Flow Variational
   Autoencoder (Flow-VAE) latents.
3. An Euler ordinary differential equation solver advances the latent state.
4. A Descript Audio Codec (DAC)-style decoder converts latents into native
   44.1 kHz stereo waveform audio.
5. Long sequences are synthesized in overlapping acoustic windows and combined
   without discontinuities.
6. The official serving path resamples the native waveform to 32 kHz, 16-bit
   stereo WAV. Native 44.1 kHz output and reference-compatible 32 kHz output
   must be treated as explicit output profiles, not silently conflated.

Current references use 200-frame acoustic windows with a 100-frame hop and 30
Euler steps. These are model behavior defaults, not permanent public API choices.

## Runtime ownership

The MLX runtime should be organized by model responsibility rather than by an
upstream framework's package layout:

```text
src/mlx_minimax_music3/
├── config.py
├── loading.py
├── tokenizer.py
├── prompting.py
├── reference.py
├── frames.py
├── sampling.py
├── chunking.py
├── audio.py
├── pipeline.py
└── models/
    ├── qwen3.py
    ├── rvq_depth.py
    ├── condition_encoder.py
    ├── flow_transformer.py
    └── vocoder.py
```

Files should be added only when the corresponding stage has a real implementation
and focused tests. Empty framework classes are intentionally avoided.

## Stage ownership contract

Modularity is defined by weight ownership and lifetime, not only by source-file
boundaries. The top-level pipeline coordinates stages but must not retain every
model component:

| Stage | Resident weights | Durable output |
|---|---|---|
| Prompt | None | Conditional and unconditional token IDs |
| Autoregressive | Global language model and RVQ depth decoder | Evaluated frame codes and fused frame hidden states |
| Acoustic | Condition encoder and flow transformer | Evaluated latent chunks and overlap state |
| Decode | Vocoder | Cropped waveform chunks |
| Output | None | Final WAV and generation metadata |

Callers may provide a generation-checkpoint directory. The pipeline fingerprints
the full request, runtime flow dtype, and model manifest identity. It stores the
evaluated autoregressive handoff and each acoustic window with hashes and tensor
metadata. Restore validates the autoregressive artifact as one unit and the
acoustic windows as a prefix. A bad autoregressive artifact invalidates the
generated cache. A missing or bad acoustic artifact truncates only that prefix.
Decode always reruns from the assembled latent chunks.

The generated cache is separate from model weights. Cache corruption causes
recomputation. Model-manifest or model-weight corruption remains a hard error.

Every weight-owning stage runs inside an explicit session with this lifecycle:

```text
validate checkpoint and memory budget
        |
instantiate model topology for the checkpoint profile
        |
load and validate component weights
        |
run stage and force evaluation of cross-stage outputs
        |
record peak and active memory
        |
drop model, temporary arrays, caches, and session ownership
```

MLX execution is lazy, so releasing Python variables before evaluating the
handoff arrays is incorrect: pending work may still capture the previous stage's
weights. A stage session must materialize every durable output before teardown.
Teardown is verified through memory telemetry and tests; garbage collection alone
is not accepted as the lifecycle design.

`stage_runners.py` runs each weight-owning stage inside its own `StageSession`
and returns the evaluated handoff with the stage's memory report. `pipeline.py`
calls the runners in order and saves and restores the generation checkpoint. It
does not hold model weights.

## Model residency

Apple silicon uses unified memory, but keeping every component resident is still
unnecessarily expensive. The initial runtime should use phase-scoped residency:

```text
tokenize and prepare prompt
        |
        v
load global LM + RVQ depth decoder
        |
generate tokens and hidden states
        |
release autoregressive weights
        |
load condition encoder + flow transformer
        |
synthesize acoustic latents in overlapping windows
        |
release flow-transformer weights
        |
load waveform decoder, decode, and write WAV
```

This design is intentionally different from a multi-GPU serving pipeline. It
optimizes for a single Apple silicon device and bounded peak memory rather than
request concurrency.

At the maximum 9,000 audio frames, fused frame hidden states have shape
`[1, frames, 8 * 4096]` and are substantial even in BF16. They are durable stage
state, not a reason to keep autoregressive weights resident. The first runtime
may retain the evaluated BF16 handoff in unified memory; later optimization may
page completed windows to a local artifact without changing model semantics.

## Precision and checkpoint profiles

The pipeline supports explicit checkpoint profiles. It never infers a global
precision from a directory name and never quantizes weights during inference.

| Profile | Purpose | Initial precision policy |
|---|---|---|
| `dense` | Correctness and parity baseline | Language model and RVQ at published BF16; acoustic components and solver at published FP32 |
| `q8` | Experimental memory profile | Allowlisted MLX affine 8-bit linear layers; not accepted for release quality until long-sequence sampling passes |

Sampling logits, classifier-free guidance, and top-k filtering use FP32. Both
top-k cuts keep ties, as SGLang's `apply_cfg` and `sample_topk_seeded` do. The
semantic window keeps every column whose conditional logit reaches the `top_k`-th
largest conditional logit. The draw keeps every window column whose guided logit
reaches the `top_k`-th largest guided logit in the window. A residual codebook
keeps every column whose guided logit reaches its `top_k`-th largest. A tie at any
of these boundaries widens the cut beyond `top_k` columns. A guided frame draws
from its whole window plus every reference candidate.
Reference-compatible categorical draws use the SGLang request-seed derivation,
MurmurHash32 column noise, and FP64 Gumbel-max scoring. Flow integration and
waveform clamping use FP32 even when their inputs come from a lower-precision
component.

Each converted checkpoint carries a manifest with the official source revision,
mapping version, component files, tensor digests, and a per-module precision or
quantization declaration. The loader instantiates dense or quantized module
topology from that manifest before loading arrays. Unknown mapping versions,
unvalidated quantization policies, and implicit fallbacks are hard errors.

## Public API boundary

The alpha API exposes a small immutable request, an explicit local checkpoint
path, deterministic seed handling, optional reference conditioning, native
waveform output, optional atomic WAV writing, and stage memory reports. Internal
codebook tensors, framework-specific schedulers, model owners, and checkpoint
mappings remain private. A `Music3Pipeline` retains only its validated manifest
and tokenizer. Generation is serialized and no model weights survive a completed
stage. Restored stages skip model loading and report zero elapsed time. Memory
reports list only stages whose models loaded during the current call.
