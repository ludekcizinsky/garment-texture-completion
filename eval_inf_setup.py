import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()  

from helpers.evaluators import get_best_inference_setup_results

@hydra.main(config_path="configs", config_name="evaluation/inference_setup.yaml", version_base="1.1")
def evaluate(cfg: DictConfig):

    print("-"*100)
    print(OmegaConf.to_yaml(cfg.evaluation))
    print("-"*100)
    get_best_inference_setup_results(cfg.evaluation)

if __name__ == "__main__":
    evaluate()