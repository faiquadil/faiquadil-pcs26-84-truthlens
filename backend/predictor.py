
"""
Fake News Detection — BiLSTM Prediction Module
Project Code: PCS26/84

Loads the saved BiLSTM checkpoint and predicts whether a
news article is classified as Fake or Real.

Note:
The prediction is a model classification, not factual verification.
"""

from pathlib import Path
import re

import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Project paths and reproducibility
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "bilstm_best_corrected.pt"
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# 2. BiLSTM model architecture
# ---------------------------------------------------------

class FakeNewsBiLSTM(nn.Module):
    """Bidirectional LSTM model matching the saved checkpoint."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        dropout: float,
        padding_idx: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        _, (hidden, _) = self.bilstm(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        return self.classifier(self.dropout(combined))


# ---------------------------------------------------------
# 3. Prediction interface
# ---------------------------------------------------------

class FakeNewsPredictor:
    """Loads the saved model and predicts Fake or Real."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.device = DEVICE

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        # Load checkpoint on CPU.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        required_keys = {
            "model_state_dict",
            "vocab",
            "config",
        }

        if not isinstance(checkpoint, dict):
            raise ValueError("The model checkpoint is not a dictionary.")

        missing_keys = required_keys - set(checkpoint.keys())

        if missing_keys:
            raise ValueError(
                f"Checkpoint is missing required keys: {missing_keys}"
            )

        self.vocab = checkpoint["vocab"]
        self.config = checkpoint["config"]

        # Read the actual vocabulary special tokens.
        if "<PAD>" not in self.vocab or "<UNK>" not in self.vocab:
            raise ValueError(
                "The vocabulary must contain <PAD> and <UNK> tokens."
            )

        self.pad_id = int(self.vocab["<PAD>"])
        self.unk_id = int(self.vocab["<UNK>"])

        self.max_length = int(
            self.config["max_sequence_length"]
        )
        embedding_dim = int(self.config["embedding_dim"])
        hidden_size = int(self.config["hidden_size"])

        self.label_mapping = self.config.get(
            "label_mapping",
            {"Fake": 0, "Real": 1},
        )

        num_classes = len(self.label_mapping)

        # The checkpoint configuration does not contain dropout.
        # Dropout is inactive during evaluation mode.
        dropout = 0.3

        self.id_to_label = {
            int(label_id): str(label)
            for label, label_id in self.label_mapping.items()
        }

        self.model = FakeNewsBiLSTM(
            vocab_size=len(self.vocab),
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=self.pad_id,
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.eval()

    # -----------------------------------------------------
    # 4. Input preprocessing
    # -----------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        """
        Tokenize and pad text.

        IMPORTANT:
        Verify that this tokenization exactly matches the
        preprocessing used during model training.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())

        token_ids = [
            self.vocab.get(token, self.unk_id)
            for token in tokens
        ]

        token_ids = token_ids[:self.max_length]

        if not token_ids:
            raise ValueError(
                "The supplied news text contains no usable tokens."
            )

        padding_length = self.max_length - len(token_ids)

        token_ids.extend([self.pad_id] * padding_length)

        return torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

    # -----------------------------------------------------
    # 5. Prediction
    # -----------------------------------------------------

    def predict(self, title: str = "", text: str = "") -> dict:
        """
        Predict the class of a news item.

        The article body is used when supplied; otherwise,
        the title is used as the model input.
        """
        if not isinstance(title, str) or not isinstance(text, str):
            raise TypeError(
                "Both title and text must be strings."
            )

        title = title.strip()
        text = text.strip()

        content = text if text else title

        if not content:
            raise ValueError(
                "Please provide a news title or article body."
            )

        input_ids = self._encode(content)

        with torch.inference_mode():
            logits = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_id = int(
            torch.argmax(probabilities).item()
        )

        predicted_label = self.id_to_label.get(
            predicted_id,
            str(predicted_id),
        )

        fake_id = self.label_mapping.get("Fake", 0)
        real_id = self.label_mapping.get("Real", 1)

        return {
            "label": predicted_label,
            "label_id": predicted_id,
            "fake_score": round(
                float(probabilities[fake_id].item()), 6
            ),
            "real_score": round(
                float(probabilities[real_id].item()), 6
            ),
            "note": (
                "These scores represent model outputs. "
                "They do not verify whether the news is factually true."
            ),
        }


# ---------------------------------------------------------
# 6. Local smoke test
# ---------------------------------------------------------

if __name__ == "__main__":
    predictor = FakeNewsPredictor()

    result = predictor.predict(
        title="Illustrative news headline",
        text=(
            "This is an illustrative article used to test "
            "whether the saved model loads and returns a prediction."
        ),
    )

    print("Model loaded successfully.")
    print("Prediction:", result)
