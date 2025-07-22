from dataclasses import dataclass


@dataclass
class Config:
    output_dir: str
    model_path: str
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    exp_name: str
    dataset_name: str
    eval_every: int
    use_liger: bool = False
    seed: int = 1
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    use_packing: bool = True
    max_length: int = 640
    max_prompt_length: int = 512
    debug: bool = False
    lr: float = 1e-5
    optimizer: str = 'AdamW'
    warm_up_steps: int = 0
    num_train_epochs: int = 1
    project_name: str = "rlhf-training"
    average_log_prob: bool = False


@dataclass
class DPOConfig(Config):
    beta: float = 0.1
    eps: float = 0.25

@dataclass
class OnlineDPOConfig(Config):
    reward_model_path: str = ""
    beta: float = 0.05
    n_iters: int = 1000

@dataclass
class RPOConfig(DPOConfig):
    eta: float = 0.005

@dataclass
class LengthDPOConfig(DPOConfig):
    alpha: float = 0.01

@dataclass
class GRPOConfig(Config):
    reward_model_path: str = ""
    beta: float = 0.03
    rloo_k: int = 2
    num_iters: int = 600

@dataclass
class EvaluateConfig:
    judge_path: str  = ""
    dataset_name: str = ""
    model_path: str = ""
    key_name: str = ""
    ref_model_path: str = ""
    seed: int = 1
    batch_size: int = 8
    num_samples: int = 256
    temperature: float = 0.7
    top_p: float  = 1.0
    max_new_tokens: int = 128
    split: str = 'train'
    implicit: bool = False

@dataclass
class WESOConfig(Config):
    local_run_dir: str = ""
    base_weak_model_path: str = ""
    aligned_weak_model_path: str = ""
    gamma: float = 0.1
    load_reference_logprobs: str = None
    