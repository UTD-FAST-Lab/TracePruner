from transformers import AutoModel
from torch import nn
import torch

class BERT(nn.Module):
    def __init__(self, num_classes=2):
        super(BERT, self).__init__()
        self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, ids, mask):
        _, emb = self.bert_model(ids, attention_mask=mask, return_dict=False)
        logits = self.classifier(emb)
        return logits, emb
