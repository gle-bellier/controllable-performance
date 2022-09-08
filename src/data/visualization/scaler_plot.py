from random import sample
import matplotlib.pyplot as plt

from data.datamodule.datamodule import ContoursDataModule
from data.dataprocessor.minmax_processor import MinMaxProcessor
from data.dataprocessor.quantile_processor import QuantileProcessor

dm = ContoursDataModule(dataset_path="data/contours/expressive/extended/",
                        batch_size=32,
                        num_workers=0,
                        sample_length=1024,
                        data_aug=False)
dm.setup()

# get data processor
# processor = MinMaxProcessor(
#     data_range=[-1, 1],
#     train_path="data/contours/expressive/extended/train.pickle",
#     ddsp_path=None)
processor = QuantileProcessor(
    data_range=[-1, 1],
    train_path="data/contours/expressive/extended/train.pickle",
    ddsp_path=None,
    output_distribution="normal",
    sample_length=1024)

train = dm.train_dataloader()

for i, x in enumerate(train):

    fig, axs = plt.subplots(3)
    z = processor(x)
    y = (processor - 1)(z)

    z_f0 = z[0, 0, :]
    z_lo = z[0, 1, :]
    rec_f0 = y[0, 0, :]
    rec_lo = y[0, 1, :]
    f0 = x[0, 0, :]
    lo = x[0, 1, :]

    axs[0].plot(f0)
    axs[0].plot(rec_f0)

    axs[1].plot(lo)
    axs[1].plot(rec_lo)

    axs[2].plot(z_f0)
    axs[2].plot(z_lo)

    plt.savefig(f"src/data/visualization/figures/train_{i}")
    plt.clf()
    plt.close()