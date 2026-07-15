 Deep Learning for Physical Property Predictions of Polycristalline Textures

BNN and MLP regressors trained to predict texture viscous anisotropy, as parameterized by the Hill orthotropic yield criterion. These regressors may also be used to predict texture rotations. To use our model, textures must be represented by their 21 independent elasticity tensor components. 

## Description

Our BNN and MLP regressors were trained on polycristalline olivine textures generated with the Viscoplastic Self-consistent (VPSC) model. To generate our training database, we simulated 300 different deformation paths of 20 Eeq = 0.1 steps, starting from a texture of 500 randomly oriented crystals. 

To be exact, our deformation paths consisted of 150 random combinations of pure sheer + axial extension veloctiy gradients, and 150 random combinations of pure sheer + axial compression veloctiy gradients. These textures were then augmented by 150 randomly generated rotations sampled from the orthorhombic fundamental zone.

From our modified version of the VPSC code, we obtained the textures's stress and strain rates to compute the Hill yield surface coefficients in order to describe their anisotropy (see Signorelli, et. all). We also obtained the textures' Cijkl elasticity tensor components to represent them, of which we only need 21 components thanks to their symmetry. 

Through this texture representation, we were able to predict both the Hill coefficients and the texture rotations well enough and in adequate time so that the results of our model may further be used in a 3D thermo-mechanical finite-element code developed to model large-scale geodynamical flows.

Our repository is built with Hydra to simplify executions over different configurations (hyperparameters, datasets, etc.) with a single line of code. The desired configuration can be specified by updating the .py folders within the "config" folder, or by running the appropreate command, as you will see in the "Execution" section. Slurm job submission is simplified thanks to the Submitit library, with example scripts in the "scripts" folder. This will allow users to experiment with different datasets, hyperparameter tunings, or even target variables. The repo structure is as follows:   

{
    ├── config      # Configurations go in here
    │   │── data
    │   │    └── data_config.py     # Data configuration
    │   │── launcher
    │   │    └── launcher_config.py     # Slurm configuration
    │   │── logging
    │   │    └── logging_config.py      # Logging configuration for MLP 
    │   ├── model
    │   │    ├── bnn_config.py          # Model configuration if using BNN
    │   │    └── mlp_config.py          # Model configuration if using MLP
    │   ├── trainer
    │   │    ├── bnn_trainer_config.py      # Training configuration if using BNN
    │   │    └── mlp_trainer_config.py      # Training configuration if using MLP
    │   ├── bnn_hpo_config.py       # Pipeline config (directories, filenames, etc.) if using BNN
    │   │── mlp_hpo_config.py       # Pipeline config if using MLP
    ├── data        # Datamodule instantiation with helper functions
    │   ├── anisotropy_datamodule.py
    │   ├── csv_dataset.py
    │   └── dataloaders.py
    ├── loggers     # Logger utilities for MLP Lightning framework
    │   └── tensorboard_utils.py
    ├── models      # Networks used in this work and their components
    │   ├── bnn
    │   │   ├── bnn_components.py
    │   │   ├── bnn_predict.py    
    │   │   ├── bnn_trainer.py    
    │   │   └── bnn.py
    │   ├── mlp
    │   │   ├── mlp_lightning_module.py
    │   │   └── mlp.py    
    ├── utils
    │   ├── __init__.py
    │   └── plotting.py
    ├── scripts     
    │   ├── bnn_hpo.sh
    │   └── mlp_hpo.sh
    ├── environment.yml
    ├── .env
    ├── bnn_training_pipeline.py
    ├── mlp_training_pipeline.py
    └── train.py        # Main file to execute

}

To reproduce our results (...) 

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Execution

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)