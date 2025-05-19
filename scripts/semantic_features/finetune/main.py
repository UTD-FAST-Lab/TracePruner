from finetune_dataset import CallGraphFinetuneDataset
from finetune_model import BERT
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read the config file
with open("config.json", "r") as f:
    config = json.load(f)

train_dataset = CallGraphFinetuneDataset(config, "train")
test_dataset = CallGraphFinetuneDataset(config, "test")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BERT().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(2):
    model.train()
    for batch in train_loader:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(ids, mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} completed")

# Save the finetuned model
torch.save(model.state_dict(), "finetuned_codebert.pth")
