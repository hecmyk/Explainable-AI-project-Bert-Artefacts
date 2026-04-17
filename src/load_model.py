"""Model loading utilities."""

import torch
from transformers import AutoModel, AutoTokenizer


def load_model(model_name: str = "bert-base-uncased"):
    """Load tokenizer + pretrained BERT model with all hidden states."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model
