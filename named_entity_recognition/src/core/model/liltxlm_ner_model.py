import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
import pytorch_lightning as pl
from transformers import LiltForTokenClassification

class LiltXLMNer(pl.LightningModule):
    def __init__(self, num_labels, learning_rate=5e-5, label2idx: dict = None):
        super().__init__()
        self.model = LiltForTokenClassification.from_pretrained("nielsr/lilt-xlm-roberta-base", num_labels=num_labels)
        self.learning_rate = learning_rate
        self.label2idx = label2idx
        #self.weight = torch.tensor([0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.weight = torch.tensor([0.5, 2.0, 2.0, 2.0])
        print(self.weight)
        self.loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        self.test_preds = []
        self.test_targets = []
        self.test_report = None

    def forward(self, input_ids, bbox, attention_mask, labels=None):
        return self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)        
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs = {key: val for key, val in batch.items() if key != 'labels' and key != 'token_type_ids'}  # Avoid passing token_type_ids
        labels = batch['labels']
        outputs = self(**inputs)
        loss = self.loss(outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1))
        preds = torch.argmax(outputs.logits, dim=2)
        preds = preds[inputs["attention_mask"].bool()]
        labels = labels[inputs["attention_mask"].bool()]

        # remove all predictions with label -100
        preds = preds[labels != -100]
        labels = labels[labels != -100]

        self.test_preds.append(preds.cpu().numpy())
        self.test_targets.append(labels.cpu().numpy())

        return {"loss": loss, "preds": preds, "targets": labels}
    

    def on_test_epoch_end(self):
        preds = np.concatenate(self.test_preds)
        targets = np.concatenate(self.test_targets)
        allowed_values = set(targets)
        filtered_dict = {key: value for key, value in self.label2idx.items() if value in allowed_values}        
        self.test_report = classification_report(targets, preds, target_names=list(filtered_dict.keys()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
