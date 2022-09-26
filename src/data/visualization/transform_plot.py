from random import sample
import matplotlib.pyplot as plt

from data.datamodule.datamodule import ContoursDataModule
from data.transforms.blur import Blur

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
transform = Blur()
train = dm.train_dataloader()

for i, x in enumerate(train):

    fig, axs = plt.subplots(2)
    z = transform(x)

    z_f0 = z[0, 0, :]
    z_lo = z[0, 1, :]
    f0 = x[0, 0, :]
    lo = x[0, 1, :]

    axs[0].plot(f0)
    axs[0].plot(z_f0)

    axs[1].plot(lo)
    axs[1].plot(z_lo)

    plt.savefig(f"src/data/visualization/figures/train_{i}")
    plt.clf()
    plt.close()