# Emotion Damage (GoEmotions) Classifier

Fine-tuning [ERNIE 3.0 Medium](https://github.com/PaddlePaddle/PaddleNLP) on the 28-label GoEmotions dataset for multi-label emotion detection.

## Dataset

`train.csv` and `test.csv` follow the Kaggle GoEmotions split. Each line is a TAB-separated pair:

```
<text>\t<label_ids>
```

`<label_ids>` is a comma-separated list covering the 28 emotions below (IDs match the notebook/original paper):

| id | label         | id | label      |
|----|---------------|----|------------|
| 0  | admiration    | 14 | fear       |
| 1  | amusement     | 15 | gratitude  |
| 2  | anger         | 16 | grief      |
| 3  | annoyance     | 17 | joy        |
| 4  | approval      | 18 | love       |
| 5  | caring        | 19 | nervousness|
| 6  | confusion     | 20 | optimism   |
| 7  | curiosity     | 21 | pride      |
| 8  | desire        | 22 | realization|
| 9  | disappointment| 23 | relief     |
| 10 | disapproval   | 24 | remorse    |
| 11 | disgust       | 25 | sadness    |
| 12 | embarrassment | 26 | surprise   |
| 13 | excitement    | 27 | neutral    |

The original exploratory workflow remains in `CIS6930_NLP_Team28_EmotionalDamage (Epoch-3)-Copy1.ipynb` for reference.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> **Note:** PaddlePaddle wheels are platform specific. Refer to [Paddle's installation guide](https://www.paddlepaddle.org.cn/install/quick) if you target GPU/CPU environments other than macOS.

## Training

The new `train.py` script mirrors the notebook pipeline with CLI ergonomics.

```bash
python3 train.py \
  --train-data train.csv \
  --eval-data test.csv \
  --epochs 3 \
  --learning-rate 5e-3 \
  --checkpoint-dir ernie_ckpt \
  --log-steps 10 \
  --eval-steps 40
```

Key flags:

- `--model-name`: switch to other ERNIE variants if desired.
- `--device`: force `cpu`/`gpu:0` when Paddle's auto-detection is not desired.
- `--resume`: resume from an existing checkpoint.

The script logs training loss/AUC/F1, periodically evaluates against the validation loader, and stores the best model/tokenizer inside `--checkpoint-dir`.

## Inference

`predict.py` loads a saved checkpoint and scores raw text strings:

```bash
python3 predict.py \
  --checkpoint-dir ernie_ckpt \
  --texts "You do a great job!" "I am so disappointed" \
  --threshold 0.5
```

To score a file (one example per line):

```bash
python3 predict.py --checkpoint-dir ernie_ckpt --input-file examples.txt --output predictions.json
```

By default, checkpoints are ignored via `.gitignore` to avoid pushing large binaries. Remove `ernie_ckpt/` from `.gitignore` if you explicitly want to commit trained weights.

## Project Structure

```
emotion_damage/      # Reusable package (data loaders, metrics, model helpers)
train.py             # CLI training script
predict.py           # CLI inference script
CIS6930_...ipynb     # Original notebook
CIS_6930_Project__Team__28_Emotional_Damage___Copy_.pdf  # Final project paper
train.csv / test.csv # Kaggle GoEmotions split
```

## Paper

The written report detailing methodology, experiments, and conclusions lives at `CIS_6930_Project__Team__28_Emotional_Damage___Copy_.pdf` in the repo root. Include it when sharing the project so readers can reference the full study alongside the code.

## Next Steps

- Capture experiment metadata (hyper-parameters, metrics) per run for easier comparison.
- Add lightweight unit tests around data loading / label encoding once Paddle is available in CI.
- Consider exporting the fine-tuned checkpoint to [PaddleHub](https://www.paddlepaddle.org.cn/hub) or ONNX if sharing with others.
