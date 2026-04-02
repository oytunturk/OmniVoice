# OmniVoice paper ↔ code mapping

Reference: [OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models](https://arxiv.org/pdf/2604.00688) (arXiv:2604.00688).

This note maps sections of the paper to the implementation under `OmniVoice/` and summarizes the paper’s main ideas.

---

## Paper outline and main ideas (summary)

The paper presents **OmniVoice**, a **zero-shot multilingual TTS** system aimed at **600+ languages** and **~581k hours** of (open-source) training data. The core is a **single-stage**, **non-autoregressive (NAR)** model in a **discrete diffusion / masked-diffusion language model** style: a **bidirectional Transformer** predicts **multi-codebook acoustic tokens** in one shot—**no separate semantic stage**, avoiding error propagation and bitrate bottlenecks of two-stage (text→semantic→acoustic) pipelines.

Two design choices are highlighted: **(1) full-codebook random masking** during training (dense supervision across all codebooks and time steps, vs. “one codebook layer per step” schedules), and **(2) initializing the backbone from a pretrained AR LLM** (Qwen3) so linguistic structure transfers and **intelligibility** improves vs. training a single-stage discrete model from scratch.

The paper also covers **multilingual scaling** (data curation, cleaning/restoration, **language-level resampling** with a β hyperparameter), **inference** (fixed-step iterative unmasking with a time-shifted schedule, confidence-based position choice, layer bias, classifier-free guidance), and **controllability** (prompt denoising, attribute-based voice design, paralinguistics, hybrid text with pinyin/English phonetics). **Section 3–4** give training/inference details and benchmark numbers (LibriSpeech-PC, Seed-TTS, MiniMax-24, FLEURS-102, etc.).

---

## §2 Proposed method → implementation

### §2.1 Architecture (single-stage NAR, text + multi-codebook audio, summed embeddings, per-codebook heads)

| Paper | Code |
|--------|------|
| Text sequence + acoustic matrix; embeddings summed over codebooks; **C independent heads** | `OmniVoice`: `audio_embeddings` with per-layer offsets, sum over codebooks in `_prepare_embed_inputs`; `audio_heads` projects to `num_audio_codebook * audio_vocab_size` and is reshaped to per-codebook logits in `forward`. |
| Loss only on **masked** target positions | `forward`: `cross_entropy` with `ignore_index=-100`; labels set in `OmniVoiceSampleProcessor` so unmasked positions are `-100`. |
| Weighted loss across codebooks | `normalized_audio_codebook_weights` (default `[8,8,6,6,4,4,2,2]`) applied to per-layer mean loss in `forward`. |

Key locations:

- `omnivoice/models/omnivoice.py` — `_prepare_embed_inputs`, `forward` (logits reshape, loss).

### §2.1.1 Full-codebook random masking

| Paper | Code |
|--------|------|
| Independent Bernoulli mask per (t,c) in the target region; masking ratio p_t ~ U(0,1) per example | `OmniVoiceSampleProcessor`: `mask_ratio = random.uniform(*self.mask_ratio_range)` with default `(0.0, 1.0)` in `TrainingConfig`; `token_mask = torch.rand(maskable_region.shape) < mask_ratio`; masked positions get `audio_mask_id`, labels `-100` elsewhere. |

- `omnivoice/data/processor.py` — masking block under “Apply masking”.
- `omnivoice/training/config.py` — `mask_ratio_range`.

### §2.1.2 LLM initialization (Qwen3 backbone)

| Paper | Code |
|--------|------|
| Backbone = pretrained LLM weights | `build_model_and_tokenizer`: either `OmniVoice.from_pretrained(...)` (full checkpoint) or `AutoModel.from_pretrained(config.llm_name_or_path)` with default `Qwen/Qwen3-0.6B` in `TrainingConfig`, wrapped as `OmniVoice(config=ov_config, llm=llm)`. |

- `omnivoice/training/builder.py` — `build_model_and_tokenizer`.
- `omnivoice/training/config.py` — `llm_name_or_path`.

### §2.2 Multilingual scaling (tokenizer, data balancing)

| Paper | Code |
|--------|------|
| Subword tokenizer from the LLM | Same `AutoTokenizer` from checkpoint / Qwen; special tokens added in `builder.py`. |
| Language-level resampling r_i with β | Repo exposes **per-manifest `repeat`** in the data JSON (`prepare_data_manifests_from_json`); you implement the paper’s r_i by choosing repeats (the closed-form β schedule is not hard-coded). |

- `omnivoice/data/dataset.py` — `prepare_data_manifests_from_json`, `repeat` on train/dev entries.
- `omnivoice/data/processor.py` — `<|lang_start|>…<|lang_end|>`, `<|instruct_start|>…<|instruct_end|>` style prefix.

### §2.3 Multi-dimensional controllability

#### §2.3.1 Acoustic control — prompt denoising

| Paper | Code |
|--------|------|
| `<|denoise|>` + noisy prompt training | Processor prepends `<|denoise|>` when `clean_start_token_idx` is in the label; `omnivoice/scripts/extract_audio_tokens_add_noise.py` builds noisy prompt data. |

#### §2.3.2 Identity — voice design (attributes)

| Paper | Code |
|--------|------|
| Speaker attributes in instructions | `omnivoice/utils/voice_design.py` defines allowed English/Chinese attribute tags; `generate(..., instruct=...)` uses them. |

#### §2.3.3 Linguistic — pinyin / phonetics

| Paper | Code |
|--------|------|
| Stochastic pinyin (Chinese) / phoneme (English) in training | `OmniVoiceSampleProcessor`: if `text_pinyin` in label and `use_pinyin_ratio`, use `text_pinyin` instead of `text`. |

- `omnivoice/data/processor.py` — `text_pinyin` branch.
- `omnivoice/training/config.py` — `use_pinyin_ratio`.

---

## §3 Experimental setup → implementation

### §3.2 Model details (Higgs-Audio tokenizer, 8 codebooks)

| Paper | Code |
|--------|------|
| Higgs-audio tokenizer, 8 codebooks | `from_pretrained` loads `HiggsAudioV2TokenizerModel`; `OmniVoiceConfig` sets `num_audio_codebook = 8`. |

- `omnivoice/models/omnivoice.py` — `from_pretrained` (audio tokenizer path).

### §3.3 Training (AdamW, lr, cosine + warmup, BF16, sequence packing)

| Paper | Code |
|--------|------|
| Hyperparameters in config | `TrainingConfig`: `learning_rate=1e-4`, `warmup_ratio=0.03`, `batch_tokens=8192`, `mixed_precision="bf16"`, etc. |
| Packed sequences | `PackingIterableDataset` + `PackingDataCollator` in `builder.py`; `forward` can use `document_ids` + flex attention block mask. |

- `omnivoice/training/config.py`
- `omnivoice/training/builder.py`
- `omnivoice/training/trainer.py`

### §3.4 Inference (32 steps, Eq. 3 schedule, CFG, layer penalty, position temperature)

| Paper | Code |
|--------|------|
| N=32, τ=0.1 (defaults) | `OmniVoiceGenerationConfig`: `num_step=32`, `t_shift=0.1`. |
| Time-shifted cumulative schedule | `_get_time_steps`: `t_shift * timesteps / (1 + (t_shift - 1) * timesteps)` on a linear grid. |
| CFG in log-prob space, guidance scale 2 | `_predict_tokens_with_scoring`: default `guidance_scale=2.0`. |
| Layer penalty (unmask lower codebooks first) | `scores - layer_ids * layer_penalty_factor` (default 5). |
| Position sampling with temperature T=5 | `position_temperature=5.0` + `_gumbel_sample` on scores. |
| Token choice = argmax | `class_temperature == 0` → `log_probs.argmax`. |

- `omnivoice/models/omnivoice.py` — `OmniVoiceGenerationConfig`, `_generate_iterative`, `_get_time_steps`, `_predict_tokens_with_scoring`, `_gumbel_sample`.

### §3.4.1 / §3.5 Evaluation

| Paper | Code |
|--------|------|
| WER/CER, SIM-o, UTMOS | `omnivoice/eval/` (e.g. `eval/wer/`, `eval/speaker_similarity/sim.py`, `eval/mos/utmos.py`). |

---

## §4–5 Results and conclusions

Sections **4** (benchmark tables) and **5** (conclusions) are **experimental reporting**, not duplicated as core training logic; use **training/eval scripts** (`examples/run_eval.sh`, `omnivoice/cli/train.py`, `omnivoice/eval/`) to reproduce metrics.

---

## Gaps / notes

- **Language resampling β=0.8**: described in the paper as a formula; the open codebase uses **explicit `repeat` counts** per manifest—same role, different interface.
- **HTML version** of the paper can garble some figure captions; use the **PDF** for authoritative notation.

---

## Why `.md` for this file?

**Markdown** is recommended over `.doc` for project docs: plain text (git-friendly), renders in GitHub and IDEs, no Word required, and easy to edit. Use Word only if you need tracked changes or corporate templates.
