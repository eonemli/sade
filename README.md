# SADE
**S**core-based **A**nomaly **DE**tection (sah-day)

## Introduction

SADE is a research project aimed at exploring and experimenting with denoising score matching for the purposes of anomaly detection in 3D brain MRIs.

## Installation

To get started with the SADE, follow these installation steps:

1. Clone the repository to your local machine.
2. Set up a Docker environment using the files present in the `docker` directory.
    - Running `build_docker.sh` will build the container and tag it with `<username>/pytorch_sde`
    - The `run_docker.sh` file outlines a potential way to run the container (mounted volumes should be changed)
3. Run `pip install -e .` from the parent directory

Note: The user may simply install the requirements in `docker/repo_requirements.txt`. Our model has been tested with Pytorch 2.1. Other library versions can be found in the [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-07.html#rel-23-07) for the NVIDIA container we base the dev image on.
## Usage

One-line quick run
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python main.py --project test --mode train --config configs/ve/toy_config.py --workdir workdir/test/
```
This will go through a dummy training run and save the outputs in a directory `workdir/test/`.

### Configurations

The `configs` directory contains configuration files that allow you to customize the project setup. Refer to the individual files within this directory for detailed configuration options and adapt them according to your experimental setups. SADE uses the `ml_collections` library which allows us to easily set configuration parameters in the command line e.g. `--config.training.batch_size=32`. All configurations are expected to be in the `configs` folder.

## Project Structure

The project is structured into several directories and files, each serving a specific purpose:

- `sade`: The main directory containing the core project components.
  - `configs`: Directory storing configuration files for training and evaluation.
  - `datasets`: Directory housing scripts for data preprocessing and loading.
  - `models`: Directory containing scripts defining various architectures and layers.
  - `optim.py`: Script defining optimization strategies for model training.
  - `losses.py`: Script defining score-based loss functions used during training.
  - `sde_lib.py`: Library script containing utilities for the SDEs that determine noise scales.
  - `debug.py`: Script assisting in debugging the project components.
- `tests`: Directory containing test scripts to ensure the robustness of the project components.
  - `test_dataloader.py`: Script containing tests for data loading functionalities.
  - `test_model.py`: Script housing tests for model components.
  - `test_train.py`: Script detailing tests for training functionalities.

## Datasets and Loaders

Our data loaders use MONAI for loading and transforming data. `filenames.py` contains functions that return a list of filenames according to the dataset specified. The MONAI dataloaders will automatically infer how to load the files according to the extensions (`.nii.gz` in our case). 

For our research we used the (ABCD)[https://nda.nih.gov/abcd/] and (HCPD)[https://www.humanconnectome.org/study/hcp-young-adult] datasets. The user will have to request access to these datasets from the linked sources. We do provide notebooks which may be used to generate the train/test splits we used for our training.  

## Extending the Project

The SADE project is designed with extensibility in mind. Follow these guidelines to add new components to the project:

### Adding New Models

To add new models, create a new file in the `models` directory defining your model. Ensure to register it as done for `ncsnpp3d.py` .

### Adding New Datasets

To incorporate new datasets, add appropriate data loading to `filenames.py`, `loaders.py` and any additional preprocessing to `transforms.py` in the `datasets` directory. Make sure to update the configuration files accordingly with the names and image sizes to support the new datasets.

## Testing

To ensure the robustness and reliability of the project, utilize the scripts in the `tests` directory. These scripts provide a framework for testing various components of the project, including data loading, model functionalities, and training procedures.

## License

The project is licensed under the terms mentioned in the `LICENSE` file. Refer to this file for detailed licensing information.
