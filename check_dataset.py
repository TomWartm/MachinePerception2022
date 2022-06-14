from hps_core.dataset.mixed_dataset import MixedDataset
from hps_core.core.config import update_hparams

hparams = update_hparams('configs/baseline.yaml')

data = MixedDataset(
    
    options=hparams.DATASET,
    ignore_3d=hparams.DATASET.IGNORE_3D,
    is_train=True
)


