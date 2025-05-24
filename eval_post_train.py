from helpers.evaluators import run_post_train_evaluation
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="evaluation/post_train.yaml", version_base="1.2")
def evaluate(cfg: DictConfig):

    print("-"*100)
    print(OmegaConf.to_yaml(cfg.evaluation))
    print("-"*100)

    run_post_train_evaluation(cfg.evaluation.run_name, cfg.evaluation.run_id, cfg.evaluation.entity, cfg.evaluation.project)


if __name__ == "__main__":
    evaluate()
