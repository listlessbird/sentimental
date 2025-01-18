import torch
import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.projection = nn.Linear(768, 128)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled = outputs.pooler_output
        
        return self.projection(pooled)