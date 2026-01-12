import torch
import torch.nn as nn
from transformers import AutoModel

class TextToParams(nn.Module):
    def __init__(self, num_plugin_parameters=9):
        super().__init__()
        # 1. text encodeor with BERT model
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")

        # 2. maping
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_plugin_parameters),
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
