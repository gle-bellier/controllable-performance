import matplotlib.pyplot as plt

from data.datamodule.datamodule import ContoursDataModule
from data.dataprocessor.minmax_processor import MinMaxProcessor

dm = ContoursDataModule(dataset_path="data/contours/expressive/extended/",
                        batch_size=32,
                        num_workers=0,
                        sample_length=1024,
                        data_aug=False)
dm.setup()

# get data processor
processor = MinMaxProcessor(
    data_range=[-1, 1],
    train_path="data/contours/expressive/extended/train.pickle",
    ddsp_path=None)

train = dm.train_dataloader()

for i, x in enumerate(train):
    plt.figure()
    z = processor(x)
    y = (processor - 1)(z)

    f0 = z[0, 0, :]
    lo = z[0, 1, :]

    plt.plot(f0)
    plt.plot(lo)
    plt.savefig(f"src/data/visualization/figures/train_{i}")
    plt.clf()