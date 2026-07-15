"""Main training script"""
import os

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

from bnn_training_pipeline import train

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# see .env.example file
dotenv.load_dotenv(override=True)

@hydra.main(config_path=None, config_name=os.environ["MAIN_CONFIG"])
def main(config: DictConfig):
    """Here we call the training pipeline and perhaps do some extra stuff"""
    print(OmegaConf.to_yaml(config))
    # Train model
    return train(config)

if __name__ == "__main__":
    # import the configs only here, since we want dotenv to run before the configs -- to register environment variables
    from config.bnn_hpo_config import register_configs
    
    register_configs()  # register hydra configs

    main()  