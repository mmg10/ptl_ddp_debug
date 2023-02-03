import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning import loggers


from model1 import LitResnet as LitResnet1
from model2 import LitResnet as LitResnet2
from dataset import IntelDataModule

DEVICE = "gpu"
EPOCHS = 1
num_cpus = os.cpu_count()



def run_training1(datamodule):

    tb_logger = loggers.TensorBoardLogger(save_dir='./tensorboard1/')
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        strategy='ddp',
        devices=1,
        num_nodes=4,
        logger=[tb_logger],
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        # fast_dev_run=True
    )
    module = LitResnet1('resnet18', 0.02, 'Adam', num_classes=10)
    trainer.fit(module, datamodule)

def run_training2(datamodule):
    
    tb_logger = loggers.TensorBoardLogger(save_dir='./tensorboard2/')
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        strategy='ddp',
        devices=1,
        num_nodes=4,
        logger=[tb_logger],
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        # fast_dev_run=True
    )
    
    module = LitResnet2('resnet18', 0.02, 'Adam', num_classes=10)
    trainer.fit(module, datamodule)
    


if __name__ == "__main__":
    datamodule = IntelDataModule(num_workers=num_cpus, batch_size=512)
    datamodule.setup()
    print('============')
    print('No "sync_dist=True"')
    print('============')
    run_training1(datamodule)
    print('============')
    print('With "sync_dist=True"')
    print('============')
    run_training2(datamodule)

