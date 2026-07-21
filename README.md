# Deep Learning for Physical Property Predictions of CPO Polycrystalline Textures

BNN and MLP regressors trained to predict polycrystalline texture viscous anisotropy, as parameterized by the Hill orthotropic yield criterion. These regressors may also be used to predict texture rotations. To use our model, textures must be represented by their 21 independent elasticity tensor components. 

## Quickstart

To skip ahead and simply run our trained model on a prediction dataset:

1. Specify the model (**mlp** or **bnn**) and the path to the training database in <code>.env</code>.

2. Go to the respective <code>/config/{mlp or bnn}_pipeline_config.py</code> (mlp or bnn) file

3. Set:

```
train: bool = False

test: bool = False

predict: bool = True

```

4. Specify the paths to the prediction dataset, trained model, etc.:

```
prediction_dataset_path : str = "/home/sabria/scratch_sabria/vpsc-hill/data/poly.csv"
load_predictive_model: bool = True
pred_model_path: str = "/path/to/trained/model"
pred_param_store_path: str = "/path/to/trained/model/params/"   #In case you're using BNN

```

5. Run the command:

```
python train.py 
```

Depending on your available resources, it is advised that you tune the Slurm configuration in the <code>/config/launcher/launcher_config.py</code>

## Description

Our BNN and MLP regressors were trained on polycrystalline olivine textures generated with the [Viscoplastic Self-consistent (VPSC)](https://doi.org/10.11578/dc.20231103.5) model. To generate our training database, we simulated 300 different deformation paths of 20 Eeq = 0.1 steps, starting from a texture of 500 randomly oriented crystals. 

To be exact, our deformation paths consisted of 150 random combinations of pure sheer + axial extension veloctiy gradients, and 150 random combinations of pure sheer + axial compression veloctiy gradients. These textures were then augmented by 150 randomly generated rotations sampled from the orthorhombic fundamental zone.

From our modified version of the VPSC code, we obtained the textures's stress and strain rates to compute the Hill yield surface coefficients in order to describe their anisotropy (see [Signorelli, et al.](https://doi.org/10.46298/jtcam.6737)). We also obtained the textures' Cijkl elasticity tensor components to represent them, of which we only need 21 components thanks to their symmetry. 

Through this texture representation, we were able to predict both the Hill coefficients and the texture rotations well enough and in adequate time so that the results of our model may further be used in a 3D thermo-mechanical finite-element code developed to model large-scale geodynamical flows.

Our repository is built with <code>Hydra</code> to simplify executions over different configurations (hyperparameters, datasets, etc.) with a single line of code. The desired configuration can be specified by updating the <code>.py</code> filess within the <code>config</code> folder, or by running the appropreate command, as you will see in the **Execution** section. 

Slurm job submission is simplified thanks to the <code>Submitit</code> library, with example scripts in the <code>scripts</code> folder. This will allow users to experiment with different datasets, hyperparameter tunings, or even prediction targets. 

The repo structure is as follows:   

```
    ├── config      # Configurations go in here
    │   │── data
    │   │    └── data_config.py     # Data configuration
    │   │── launcher
    │   │    └── launcher_config.py     # Slurm configuration
    │   ├── model
    │   │    ├── bnn_config.py          # Model configuration if using BNN
    │   │    └── mlp_config.py          # Model configuration if using MLP
    │   ├── trainer
    │   │    ├── bnn_trainer_config.py      # Training configuration if using BNN
    │   │    └── mlp_trainer_config.py      # Training configuration if using MLP
    │   ├── bnn_pipeline_config.py       # Pipeline config (directories, filepaths, etc.) if using BNN
    │   │── mlp_pipeline_config.py       # Pipeline config if using MLP
    ├── data        # Datamodule instantiation with helper functions
    │   ├── anisotropy_datamodule.py
    │   ├── csv_dataset.py
    │   └── dataloaders.py
    ├── models      # Networks used in this work and their components
    │   ├── bnn
    │   │   ├── bnn_components.py
    │   │   ├── bnn_predict.py    
    │   │   ├── bnn_trainer.py    
    │   │   └── network.py
    │   ├── mlp
    │   │   ├── mlp_trainer.py
    │   │   └── network.py    
    ├── utils
    │   ├── __init__.py
    │   └── plotting.py
    ├── scripts     # Submission scripts
    │   └── multirun.sh
    ├── requirements.txt
    ├── .env        # Please specify model config (bnn or mlp) and directory to your training database here
    ├── bnn_training_pipeline.py
    ├── mlp_training_pipeline.py
    └── train.py        # Main file to execute

```

To reproduce our results (...) 

## Getting Started

### Dependencies

Our work was mainly developed in Python 3.10 using Pytorch 2.11 and Pyro 1.9, the probabilistic programming language built on top of it. Dependencies can be found in the <code>requirements.txt</code> file accompanying our repo. 

### Installing

Simply install the required libraries with pip using the command:

```
pip install -r requirements.txt
```

### Execution

First, you must specify the model (**mlp** or **bnn**) and the path to your training database in <code>.env</code>.

In case you want to train a model from scratch, you must specify your desired configurations in the config folder as described above. Depending on the type of network that you want to train, in the respective <code>pipeline_config.py</code> file, make sure to set 

```
train: bool = True
```
If you wish to subsequently perform a prediction with your trained model, set

```
predict: bool = True
```

If you wish to use a pre-trained model, please specify this with 

```
train: bool = False

predict: bool = True
load_model: bool = True
load_dir: str = "directory/to/your/model"

```

The execution command, for either/both training and predictions with your desired configuration and hyperparameters:

```
python train.py 
```

In case you want to train on a number of hyperparameter combinations, you must add the following command to your submission script: 

```
# Example choice of multiple hyperparam combinations
HYPERPARAMETERS="model.hidden_dim=choice(42,84), datamodule.batch_size=choice(256,512)"  

(...)   # Load env, etc.

python train.py --multirun $HYPERPARAMETERS &
wait

```

Outputs along with logs will appear in a separate folder. 

## Authors

By Audrey Sabri for Géosciences Montpellier under the supervision of [Andrea Tommasi](https://www.gm.umontpellier.fr/annuaire/tommasi-andrea/) and [Nestor Cerpa](https://nestorcerpa.wordpress.com/). 

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

VPSC code can be found at
* [vpsc](https://github.com/lanl/vpsc_code)

Hydra template used belongs to 
* [hyperparameter-search-template](https://github.com/NKI-AI/hyperparameter-search-template)