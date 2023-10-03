import torch
from sentence_transformers import SentenceTransformer


class LLM(torch.nn.Module):
    def __init__(self, transformer="paraphrase-MiniLM-L3-v2", n_classes=2):
        """A transformer based LLM class

        Args:
            transformer (str, optional): LLM name. Defaults to "paraphrase-MiniLM-L3-v2".
            n_classes (int, optional): number of classes. Defaults to 2.
        """
        super(LLM, self).__init__()
        self.transformer = SentenceTransformer(transformer)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.classifier = torch.nn.Linear(self.compute_dim(), n_classes)

    def compute_dim(self):
        """Get hidden dimension
        """
        return self.transformer[1].word_embedding_dimension

    def embed(self, ids, mask):
        """Get embedding of an item
        """
        data = self.transformer({'input_ids': ids, 'attention_mask': mask})
        return data['sentence_embedding']

    def forward(self, ids, mask):
        """Forward pass

        Args:
            ids (Torch.LongTensor): input_ids
            mask (Torch.LongTensor): attention mask

        Returns:
            torch.Tensor: logits
        """
        return self.classifier(
                self.dropout(self.embed(ids, mask)))
