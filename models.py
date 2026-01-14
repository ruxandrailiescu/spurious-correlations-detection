import torch.nn as nn
import sentence_transformers


class SentenceEncoder(nn.Module):

    def __init__(self, model_name='Alibaba-NLP/gte-large-en-v1.5'):
        super().__init__()
        self.model_name = model_name
        self.encoder = sentence_transformers.SentenceTransformer(model_name, trust_remote_code=True)

    def encode_texts_batched(self, texts: list[str], device, bs=128):
        self.to(device)
        self.eval()
        embeddings = self.encoder.encode(
            texts, 
            batch_size=bs, 
            device=device, 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        return embeddings.cpu()
        
    def forward(self, x):
        device = next(self.parameters()).device
        return self.encode_texts_batched(x, device=device)
    