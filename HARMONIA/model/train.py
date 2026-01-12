import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import TextToParams

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 8  # Parameter count, should match plugin
EPOCHS = 100
LR = 1e-4

# --- DATASET ---
class PresetDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Text Processing
        text = item['description']
        tokens = self.tokenizer(text, padding="max_length", max_length=32, truncation=True, return_tensors="pt")

        # Paramter Process
        # Ensure JSON parameters are a list of floats 0.0-1.0
        params = torch.tensor(item['parameters'], dtype=torch.float32)

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': params
        }

# --- TRAINING LOOP ---
def train():
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    dataset = PresetDataset("dataset/presets.json", tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TextToParams(num_plugin_parameters=PLUGIN_PARAM_COUNT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss() # Minimizing differrance between prediction and real knobs

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # Forward pass
            preds = model(batch['input_ids'], batch['attention_mask'])

            # Calculate error
            loss = loss_fn(preds, batch['labels'])

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.6f}")

    # Saving model
    torch.save(model.state_dict(), "my_plugin_ai.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
