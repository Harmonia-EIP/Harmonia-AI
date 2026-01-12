import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

    def forward(self, input_ids, attention_mask):
        # Get text features
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Take the "sentence embedding" (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # Predict parameters
        params = self.head(pooled_output)
        return params
