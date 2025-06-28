from transformers import T5EncoderModel, AutoModel
from torch import nn
import torch

# Constants (should be configurable elsewhere in real use)
MODELS_BATCH_SIZE = {"codebert": 32, "codet5": 32, "codet5_plus": 12}
NO_EPOCHS = 2
LR = 0.00001

class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.out = nn.Linear(768, 2)

    def forward(self, ids, mask):
        _, emb = self.encoder(ids, attention_mask=mask, return_dict=False)
        out = self.out(emb)
        return out


class CodeT5Classifier(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained('Salesforce/codet5p-770m')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.encoder.config.hidden_size, 2)

    # def _get_encoder(self, name):
    #     if name == 'codet5':
    #         return T5EncoderModel.from_pretrained('Salesforce/codet5-base')
    #     elif name == 'codet5_plus':
    #         return T5EncoderModel.from_pretrained('Salesforce/codet5p-770m')
    #     else:
    #         raise ValueError("Unsupported model name for CodeT5Classifier")

    def forward(self, ids, mask):
        emb = self.encoder(ids, attention_mask=mask, return_dict=False)[0][:, -1]  # Last token
        emb = self.dropout(emb)
        out = self.out(emb)
        return out


# class CodeAndStructClassifier(nn.Module):
#     def __init__(self, model_name='codet5', hidden_size=16, dropout_rate=0.25):
#         super().__init__()
#         self.encoder = self._get_encoder(model_name)
#         self.code_proj = nn.Linear(self.encoder.config.hidden_size, hidden_size)
#         self.struct_proj = nn.Linear(STRUCT_FEAT_DIM, hidden_size)
#         self.out = nn.Linear(2 * hidden_size, 2)
#         self.dropout = nn.Dropout(p=dropout_rate)

#     def _get_encoder(self, name):
#         if name == 'codet5':
#             return T5EncoderModel.from_pretrained('Salesforce/codet5-base')
#         elif name == 'codet5_plus':
#             return T5EncoderModel.from_pretrained('Salesforce/codet5p-770m')
#         elif name == 'codebert':
#             return AutoModel.from_pretrained('microsoft/codebert-base')
#         else:
#             raise ValueError("Unsupported model name for CodeAndStructClassifier")

#     def forward(self, ids, struct_feat, mask):
#         emb = self.encoder(ids, attention_mask=mask, return_dict=False)[0][:, -1]
#         h_code = self.code_proj(emb)
#         h_struct = self.struct_proj(struct_feat)
#         h = torch.cat([h_code, h_struct], dim=1)
#         h = self.dropout(h)
#         out = self.out(h)
#         return out, h  # also return embedding for downstream use


# class StaticOnlyClassifier(nn.Module):
#     def __init__(self, input_dim=11, hidden_size=16):
#         super().__init__()
#         self.encoder = nn.Linear(input_dim, hidden_size)
#         self.out = nn.Linear(hidden_size, 2)

#     def forward(self, struct):
#         h = self.encoder(struct)
#         return self.out(h)