import torch
import json
from torch.utils.data import Dataset

class PresetDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['description']
        tokens = self.tokenizer(text, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
        # Ensure parameters are floats
        params = torch.tensor(item['parameters'], dtype=torch.float32)
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': params
        }
