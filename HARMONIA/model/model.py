import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextToParams(nn.Module):
    def __init__(self, num_plugin_parameters=50):
        super().__init__()
        # 1. Text Encoder (Understand the prompt)
        # Small, pre-trained BERT model.
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")

        # 2. The "Mapper" (Map text meaning to Knobs)
        # Converts the 128 features from BERT into x knobs
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_plugin_parameters),
            nn.Sigmoid() # Forces output to be between 0.0 and 1.0
        )

    def forward(self, input_ids, attention_mask):
        # Get text features
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Take the "sentence embedding" (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # Predict parameters
        params = self.head(pooled_output)
        return params
