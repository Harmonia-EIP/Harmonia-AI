import torch
import torch.nn as nn
import os
from transformers import AutoModel

MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")

class TextToParams(nn.Module):
    def __init__(self, num_plugin_parameters=9):
        super().__init__()
        if int(num_plugin_parameters) <= 0:
            raise ValueError("num_plugin_parameters must be > 0")

        # 1. text encodeor with BERT model
        self.bert = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)  # nosec B615
        hidden_size = int(getattr(self.bert.config, "hidden_size", 128))

        # 2. maping
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, int(num_plugin_parameters)),
            nn.Sigmoid() # Output 0.0 - 1.0
        )

    def forward(self, input_ids, attention_mask):
        # text feature
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Take the "sentence embedding" (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # Predict parameters
        params = self.head(pooled_output)
        return params
