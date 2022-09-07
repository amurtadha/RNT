import torch.nn as nn
from transformers import  AutoModel, AutoConfig

class RNT(nn.Module):
    def __init__(self, args):
        super(RNT, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        self.encoder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.to('cuda')
        layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        self.classifier = nn.Sequential(*layers)
    def forward(self, inputs):
        input_ids,token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        logits = self.classifier(pooled_output)
        return pooled_output, logits





