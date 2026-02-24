import json
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

class MultiLabelPredictor:
    """
    Multi-label predictor for fashion attributes.
    Loads:
      - Keras model with sigmoid outputs (num_labels)
      - label_vocab.json: list of label strings
      - label_groups.json: dict with groups: gender/color/article (label strings)
    """
    def __init__(self, model_path: str, label_vocab_path: str, label_groups_path: str):
        self.model = tf.keras.models.load_model(model_path)

        with open(label_vocab_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        with open(label_groups_path, "r", encoding="utf-8") as f:
            self.groups = json.load(f)

        if not isinstance(self.labels, list) or len(self.labels) == 0:
            raise ValueError("label_vocab.json must be a non-empty JSON list.")

        if not isinstance(self.groups, dict) or len(self.groups) == 0:
            raise ValueError("label_groups.json must be a non-empty JSON dict.")

        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def _prepare(self, img_bytes: bytes) -> np.ndarray:
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        x = tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)
        return x.numpy()

    def predict(self, img_bytes: bytes, threshold: float = 0.5, top_k: int = 5):
        x = self._prepare(img_bytes)
        probs = self.model.predict(x, verbose=0)[0]  # (num_labels,)

        # Overall top-k labels
        top_k = max(1, min(int(top_k), len(probs)))
        idxs = probs.argsort()[::-1][:top_k]
        topk = [
            {"label": self.labels[int(i)], "prob": round(float(probs[int(i)]), 6)}
            for i in idxs
        ]

        # Active labels above threshold
        active_idxs = np.where(probs >= float(threshold))[0].tolist()
        active = [
            {"label": self.labels[int(i)], "prob": round(float(probs[int(i)]), 6)}
            for i in active_idxs
        ]
        active = sorted(active, key=lambda d: d["prob"], reverse=True)

        # Best label per group
        best_per_group = {}
        for gname, glabels in self.groups.items():
            # Filter labels that exist in vocab (safety)
            valid = [l for l in glabels if l in self.label_to_idx]
            if not valid:
                continue
            gidxs = [self.label_to_idx[l] for l in valid]
            best_i = max(gidxs, key=lambda i: probs[i])
            best_per_group[gname] = {
                "label": self.labels[int(best_i)],
                "prob": round(float(probs[int(best_i)]), 6)
            }

        return {
            "best_per_group": best_per_group,
            "active": active,
            "topk": topk
        }
