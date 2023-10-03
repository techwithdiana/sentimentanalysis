import torch
import transformers
from network import LLM
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Pipeline():
    def __init__(self, transformer="paraphrase-MiniLM-L3-v2", lr=0.0001):
        """Pipeline to train and infer for sentiment analysis

        Args:
            transformer (str, optional): LLM name.. Defaults to "paraphrase-MiniLM-L3-v2".
            lr (float, optional): learning rate. Defaults to 0.0001.
        """
        self.nnet = LLM(transformer)
        self.nnet.to("cuda")
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(
            params=self.nnet.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.schedular = None

    def train(self, trn_dataset, val_dataset, batch_size, n_epochs=100):
        """Train the network with given data

        Args:
            trn_dataset (torch.utils.data.Dataset): train dataset
            val_dataset (torch.utils.data.Dataset): validation dataset
            batch_size (int): batch size
            n_epochs (int, optional): _description_. Defaults to 100.
        """
        trn_loader = torch.utils.data.DataLoader(
            trn_dataset,
            batch_size=batch_size,
            collate_fn=None,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=None,
            shuffle=False,
            num_workers=3,
            pin_memory=True)

        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optim,
            num_warmup_steps=500,
            num_training_steps=len(trn_loader) * n_epochs)

        for epoch in range(n_epochs):
            self.nnet.train()
            if (epoch != 0 and epoch % 2 == 0) or epoch == n_epochs-1:
                pred = self.inference(val_loader)
                self.evaluate(val_dataset.labels, pred)
            pbar = tqdm(trn_loader)
            for data in pbar:
                self.optim.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.nnet(data[0].to("cuda"), data[1].to("cuda"))
                    loss = self.loss(logits, data[2].long().to("cuda"))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.scheduler.step()
                pbar.set_description(f"Loss: {loss.item(): .4f}")

    @torch.no_grad()
    def inference(self, data_loader):
        """Predict and calculate accuracy

        Args:
            data_loader (torch.utils.data.DataLoader): data loader for test set

        Returns:
            np.ndarray: predictions
        """
        self.nnet.eval()
        predictions = torch.zeros(len(data_loader.dataset))
        idx = 0
        for data in tqdm(data_loader, desc="Predicting.."):
            bsz = data[0].size(0)
            logits = self.nnet(data[0].to("cuda"), data[1].to("cuda"))
            predictions[idx: idx+bsz] = torch.argmax(logits, dim=1).cpu()
            idx += bsz
        return predictions.numpy()

    def evaluate(self, y_true, y_pred):
        """print accuracy

        Args:
            y_true (np.ndarray): ground truth
            y_pred (np.ndarray): predictions
        """
        print(f"Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")