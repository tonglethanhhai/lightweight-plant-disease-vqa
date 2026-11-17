# Lightweight VQA for Plant Leaf Disease Recognition (TAIMMA 2027)

This repository provides the standalone implementation and inference checkpoint of the proposed model **MNet-LSTM + paraphrases**, together with the training logs (`.log`) and training/summary metrics (`.csv`) from the controlled study.

## Proposed Model

- **Image encoder:** ImageNet-pretrained `mobilenetv3_large_100` (last two blocks fine-tuned, `960`-d pooled feature)
- **Question encoder:** Word embedding (`300`-d) + 2-layer Bi-LSTM (`512`-d, trained from scratch)
- **Fusion & classifier:** Raw concatenation `[v ; q]` (`1472`-d) -> MLP (`1472 -> 512 -> 15`)
- **Parameters:** `6.47 M` total (`5.68 M` trainable)
- **Reported test scores (3-seed mean ± std):** `95.56 ± 0.17` template macro-F1, `95.54 ± 0.18` paraphrase macro-F1, `98.91 ± 0.19` consistency

## Repository Contents

```
lightweight-plant-disease-vqa/
├── proposed_mnet_lstm_para.ipynb                                   # Standalone notebook (model, inference, eval, CSV summary)
├── examples/
│   └── sample_rice_brown_spot.jpg                                  # Sample leaf image used in notebook inference cell
├── checkpoints/
│   └── mnet_lstm_finetuned_concat_raw_paraphrases/seed42/model_best.pth       # Inference weights + vocab/config (~26.1 MB)
├── results/
│   ├── tables/summary.csv                                          # Seed-averaged summary grouped by study design stage
│   ├── baselines/                                                  # Frozen MobileNetV3/DeiT + Bi-LSTM/BERT baselines
│   ├── encoder_finetuning/                                         # Partial encoder fine-tuning runs
│   ├── fusion/                                                     # Fusion operator comparisons (frozen encoders)
│   ├── self_attention/                                             # Post-fusion self-attention variants
│   ├── combined/                                                   # Combined configurations & proposed model
│   └── lvlm_reference/                                             # Qwen3-VL-4B LoRA reference training CSV
├── logs/
│   ├── baselines/, encoder_finetuning/, fusion/, self_attention/, combined/, lvlm_reference/
│   └── *.log                                                       # Session, validation, and latency benchmark logs
├── requirements.txt
└── .gitignore
```

## Dataset

The RiMa rice and maize leaf disease images are publicly available at https://doi.org/10.57967/hf/11351.

## Quickstart

```bash
pip install -r requirements.txt
```

Open and run `proposed_mnet_lstm_para.ipynb` from the repository root. Optional environment variables for local evaluation on downloaded RiMa files:

- `RIMA_SAMPLE_IMAGE`: path to a leaf image for single-image prediction
- `RIMA_DATA_DIR` and `RIMA_IMAGE_ROOT`: paths to local RiMa JSON annotations (`test.json`) and images
