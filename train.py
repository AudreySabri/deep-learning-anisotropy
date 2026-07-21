"""Main training script"""
import os

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

from bnn_training_pipeline import train_bnn
from mlp_training_pipeline import train_mlp


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# see .env.example file
dotenv.load_dotenv(override=True)

@hydra.main(config_path=None, config_name=os.environ["MAIN_CONFIG"])
def main(config: DictConfig):
    """Here we call the training pipeline and perhaps do some extra stuff"""
    print(OmegaConf.to_yaml(config))
    # Train model
    if os.environ["MAIN_CONFIG"] == "bnn":
        return train_bnn(config)
    
    if os.environ["MAIN_CONFIG"] == "mlp":
        return train_mlp(config)

if __name__ == "__main__":
    # import the configs only here, since we want dotenv to run before the configs -- to register environment variables
    from config.bnn_pipeline_config import register_bnn_configs
    from config.mlp_pipeline_config import register_mlp_configs
    
    if os.environ["MAIN_CONFIG"] == "bnn":
        register_bnn_configs()  # register bnn hydra configs

    if os.environ["MAIN_CONFIG"] == "mlp":
        register_mlp_configs()  # register mlp hydra configs

    main()  