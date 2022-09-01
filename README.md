# Controllable Performance with Diffusion Models

This project aims at using Denoising Diffusion Models for expressive musical performances generation. 

This project aims at using  _Diffusion Models_ for expressive musical performance generation. We want to generate samples of violin performances that include expressive elements such as _glissando_, _tremolo_, and _vibrato_. To do so we adopt a modular approach by using a two-stage framework:

- Controllable performance generation (our work)
- Sound synthesis with the _DDSP_ model introduced by Google Magenta.

Then, the main goal is to model fundamental frequency and loudness contours (time series) that are similar to the ones obtained in musical performances (with slight variations of pitch and loudness within a note, and smooth transitions between the notes). 
We also want our model to be controllable and we propose a way to condition the generation on MIDI files or contours from existing violin performances.

## Train Model:

Write an experiment config file `myconfig.yaml` and save it in the `configs/` folder.
Launch experiment with:
```
python src/train/trainer.py --config=configs/myconfigs.pickle
```
## Project Structure

### Scripts

- `data/`: data processing, conditioning transformations and Pytorch data modules.

- `edits/`: contours editing methods, i.e. controllable generation of expressive contours scripts.

- `eval/`: scripts for models evaluation and results analysis.

- `models/`: all models and neural networks layers used for this project.

- `samplers/`: samplers dedicated to denoising diffusion models.
- `sde/`: stochastic differential equations used for the diffusion process.
- `train/`: training handler and diffusion models training process.
- `utils/`: some useful tools.


### Tests

All the tests are available in the `tests/` folder and ensure the well behaving of the dataprocessing, models, neural network blocks, transforms...


### Data

The data used is not included in this repository but many rely on the violon performances samples included in the _URMP_ dataset.The folder should be organized as follows:

- `audio/` containing all the violin performances audio samples (wav files at 16kHz).
- `midi/` containing MIDI files for conditionnal generation.
- `contours/` containing the `expressive` folder where are located the train, validation and test datasets but also the `unexpressive` folder where the unexpressive contours (extracted from MIDI files) are stored.









## Installation

1. Clone this repository:

```bash
git clone https://github.com/gle-bellier/controllable-performance.git

```

2. Install requirements:

```bash
cd controllable-performance
pip install -r requirements.txt

```