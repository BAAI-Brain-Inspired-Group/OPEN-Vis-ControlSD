from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import *
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
modelarch_path = './models/cldm_v15.yaml'
resume_path = './image_log/checkpoint_deconv_down2_3/last.ckpt'
logger_path = 'shuimo_deconv2_3_test'
# dataset_name_list = ['MyDatasetCOCO','MyDatasetCOCO_canny','MyDatasetCOCO_val','MyDatasetCOCO_val_canny','MyDatasetBihua','MyDatasetBihuaCanny','MyDatasetShuimo','MyDatasetShuimoCanny']
dataset_name = 'MyDatasetShuimo'

checkpoint_path = f'image_log/checkpoint_down2_0/'
# Configs train
# resume_path = './models/control_sd15_ini.ckpt'
# resume_path = './image_log/checkpoint/last.ckpt'
# batch_size = 4
# logger_freq = 3000
# learning_rate = 1e-5
# resume_path = './image_log/checkpoint_deconv2/epoch=3-step=118288.ckpt'
# iftrain = True

# Configs test
batch_size = 1
logger_freq = 1
learning_rate = 0.
sd_locked = True
only_mid_control = False
max_epoch = 5
iftrain = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(modelarch_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
# dataset = MyDataset()
# dataset = MyDatasetHC()
dataset = eval(dataset_name)()
if iftrain:
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
else:
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
logger = ImageLogger(batch_frequency=logger_freq,split=logger_path)
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path, 
    save_top_k=-1,
    save_last=True,
    save_weights_only=False, 
    every_n_epochs=1,
)

# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
if iftrain:
    trainer = pl.Trainer(strategy='ddp',accelerator='gpu',devices=[0], precision=32, callbacks=[logger,checkpoint_callback])
else:
    trainer = pl.Trainer(strategy='ddp',accelerator='gpu',devices=[0], precision=32, callbacks=[logger],max_epochs=max_epoch)


# # Train!
trainer.fit(model, dataloader)
# trainer.validate(model, dataloader,ckpt_path=resume_path)
