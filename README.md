<<<<<<< HEAD
# System-1 Decision Engine

Query -> decoder Transformer -> pooled representation -> custom Decision Head -> K logits -> temperature -> softmax
-> calibration head (confidence) -> policy (ROUTE / CLARIFY / ABSTAIN / ESCALATE). No LM head, no text generation.

## Setup (MacBook Air M2)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_data.py          # synthetic data -> data/processed (pipeline check only, replace with real data)
pytest -q                            # 19 tests
```
The Apple GPU (MPS) is picked automatically; fallback is CUDA, then CPU.

## Two backbones (`model.backbone` in the config)
| config | backbone | notes |
|---|---|---|
| `configs/tiny.yaml` | `scratch`: own RMSNorm + RoPE + causal attention + SwiGLU decoder (~0.65M params) | no download, trains in about a minute |
| `configs/pretrained.yaml` | `pretrained`: `HuggingFaceTB/SmolLM2-135M` via `AutoModel` (LM head never loaded) | first run downloads about 270 MB from Hugging Face |

Any Llama/Qwen-style decoder works: set `pretrained_name` (e.g. `Qwen/Qwen2.5-0.5B`, about 1 GB, still fine on 8 GB RAM for inference and top-layer tuning).

## Train / evaluate / predict
```bash
python scripts/train.py --config configs/tiny.yaml --train-data data/processed/train.jsonl --val-data data/processed/val.jsonl
python scripts/evaluate.py  --checkpoint experiments/tiny/best.pt --test-data data/processed/test.jsonl
python scripts/calibrate.py --checkpoint experiments/tiny/best.pt --val-data data/processed/val.jsonl   # temperature + calibration head on validation only
python scripts/predict.py   --checkpoint experiments/tiny/best_calibrated.pt --text "My login page is not opening in WhatsApp."
python scripts/benchmark.py --checkpoint experiments/tiny/best.pt
python scripts/train.py --config configs/tiny.yaml --train-data ... --val-data ... --resume   # resume from last.pt
```
Pretrained: same commands with `configs/pretrained.yaml`. Stage 1 trains only the heads (`head_warmup_epochs`), stage 2 unfreezes the top
`unfreeze_fraction` of layers at `backbone_lr`. `lora_rank > 0` adds LoRA on q_proj/v_proj.

## Dynamic options (candidates supplied with the query)
```bash
python scripts/train.py --config configs/tiny_dynamic.yaml --train-data data/processed/train_dyn.jsonl --val-data data/processed/val_dyn.jsonl
python scripts/predict.py --checkpoint experiments/tiny_dynamic/best.pt --text "I was charged twice" --options "billing,tech support,sales"
```
Layout is `Candidate: <text> [OPT_k]` so each marker (causal decoder) can see its own candidate text; one shared `OptionScorer` scores all options.
For the pretrained backbone set `decision.mode: dynamic` and `max_seq_len: 256`.

## Data formats (JSONL)
- fixed: `{"query": "...", "label": 0, "difficulty": "hard"}`
- dynamic: `{"context": "...", "question": "...", "options": ["..."], "target": 0}`
- labels: `data/labels.json` (`{"0": "technical_team", ...}`); set `model.num_classes` to match.

## Outputs
`experiments/<name>/`: `config.json, env.json, history.json, metrics.json, best.pt, last.pt, tokenizer/`, and `eval_*/` with
`metrics.json, confusion_matrix.png, reliability_diagram.png, confidence_histogram.png, coverage_vs_accuracy.png`.

## Notes
- `entropy` is normalised by log(K) (range 0 to 1). Calibration head input is detached, so confidence training never changes the decision logits.
- Synthetic data is trivially separable (hence ~100% accuracy); real evaluation needs a real dataset with hard negatives and leakage-checked splits.
- Thresholds in `policy:` are tunable on validation data only, never test.
- Pretrained `decision_token` pooling is equivalent to `last`: the marker is plain text for HF tokenizers.

## Layout
`src/system_one/`: `config, utils, tokenizer, data, losses, metrics, policy, calibration, trainer, predictor`, and `model/` (`layers, backbone, heads, lora, system_one`).
=======
# DeciFormer-S1
DeciFormer-S1: a small, fast decoder-Transformer that answers any question about a text by scoring the options you supply, with calibrated confidence and a route / clarify / abstain policy. Runs locally, no text generation.
>>>>>>> 3634b9057c64e98cf99355eb8f761a1c068fb39b
