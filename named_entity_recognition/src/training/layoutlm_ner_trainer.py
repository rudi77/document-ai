import hydra
import pytorch_lightning as pl
from transformers import LayoutLMTokenizer
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

from core.dataset.ner_datamodule import LayoutLMDataModule
from core.model.layoutlm_ner_model import LayoutLMNER

@hydra.main(config_path="../core/config", config_name="config")
def main(cfg: DictConfig):
    csv_dir = cfg.data.csv_dir
    image_dir = cfg.data.image_dir
    batch_size = cfg.data.batch_size    
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    data_module = LayoutLMDataModule(csv_dir, image_dir, tokenizer, cfg.model.label2idx, batch_size, cfg.data.num_workers)

    model = LayoutLMNER(
        num_labels=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        label2idx=cfg.model.label2idx
    )

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='../weights',
        filename='ner-layoutLM-best-checkpoint',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback], # Add the callback
    )

    trainer.fit(model, data_module)
    
    # Load the best model
    model = LayoutLMNER.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        num_labels=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        label2idx=cfg.model.label2idx
    )
    # Test the model
    trainer.test(model, datamodule=data_module)
    print(model.test_report)

if __name__ == '__main__':
    main()
