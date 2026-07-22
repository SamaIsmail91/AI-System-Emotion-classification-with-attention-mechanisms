# Emotion Classification with Attention

A complete, end-to-end emotion classification project on Google's **GoEmotions** dataset:
data pipeline → LSTM / GRU / BiLSTM+Attention / fine-tuned DistilBERT → evaluation →
attention visualization → a deployable **Streamlit** app.

## What's in this folder

```
├── Emotion_Classification_with_Attention.ipynb   # the full project notebook (run this first)
├── requirements-notebook.txt                     # deps for the notebook / training
├── app/
│   ├── app.py                                    # the Streamlit deployment app
│   ├── requirements.txt                          # lightweight deps for the app only
│   └── models/                                   # <- copy this here after running the notebook
└── README.md
```

## 1. Run the notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook Emotion_Classification_with_Attention.ipynb
```

Run all cells top to bottom. The notebook will:

1. Download the **real GoEmotions** `train.tsv` / `dev.tsv` / `test.tsv` + label mappings straight
   from the `google-research/google-research` GitHub repo (no manual download needed).
2. Map the 27 fine-grained emotions to the **6 core classes** (joy, sadness, anger, fear, surprise, disgust)
   using Google's own Ekman mapping.
3. Clean, tokenize, and pad the text; load pretrained **GloVe** embeddings.
4. Train and evaluate **LSTM**, **GRU**, and **BiLSTM + Attention** models.
5. Fine-tune **DistilBERT** with the HuggingFace `Trainer` API (needs internet access to
   `huggingface.co`; the cell degrades gracefully with a clear message if that's unavailable in
   your environment, e.g. a locked-down sandbox — it runs fine on Colab/Kaggle/a normal machine).
6. Plot attention heatmaps to interpret predictions.
7. Save the trained BiLSTM + Attention model + tokenizer to `models/`.

> Increase `RNN_UNITS`/`EPOCHS` near the top of the training section, or switch to
> `glove-wiki-gigaword-300`, for a stronger (slower) model.

## 2. Run the Streamlit app

Copy the `models/` folder produced by the notebook into `app/models/`, then:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints. Type any sentence and get:
- the dominant emotion with an emoji + color,
- a confidence bar chart across all 6 classes,
- a token-level **attention heatmap** explaining the prediction.

## Design notes

The app uses a light, low-glare theme (soft lavender-white background, white cards, indigo
accents) with a distinct color per emotion, so it stays comfortable to read for long sessions
and works well for screenshots/demos.

## Extending the project

- Swap the deployed model for the fine-tuned DistilBERT (`models/distilbert_emotion/`) for higher
  accuracy at the cost of a heavier runtime dependency (`transformers` + `torch`).
- Try the original 27-way multi-label GoEmotions task instead of the 6-class collapse.
- Add cross-validation, richer embeddings (FastText, `glove-wiki-gigaword-300`), or a second
  BiLSTM layer for extra capacity.
