import torch
from utils import read_data
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, tokenizer='bert-base-uncased', max_length=32):
        """Dataset Class

        Args:
            dir (str): _description_
            tokenizer (str, optional): tokenizer type. Defaults to 'bert-base-uncased'.
            max_length (int, optional): max sequence length. Defaults to 32.
        """
        self.text, self.labels = read_data(dir)
        self.input_ids, self.attention_mask = self.tokenize(tokenizer, max_length)

    def tokenize(self, tokenizer, max_length):
        """Tokenize given text

        Args:
            tokenizer (str): tokenizer type
            max_length (int): max sequence length

        Returns:
            (torch.Tensor, torch.Tensor): input ids and attention masks
        """
        print("Tokenizing..")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=True)
        output = tokenizer(
            self.text,
            padding=True,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt")
        return output['input_ids'], output['attention_mask']

    def __len__(self):
        """Length of dataset
        """
        return len(self.text)

    def __getitem__(self, idx):
        """Get input_ids, attention_mask and label for a given index
        """
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]