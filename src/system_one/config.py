from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    backbone: str = "scratch"
    pretrained_name: str = "HuggingFaceTB/SmolLM2-135M"
    vocab_size: int = 0
    max_seq_len: int = 128
    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 4
    intermediate_size: int = 352
    dropout: float = 0.1
    rope_theta: float = 10000.0
    num_classes: int = 5


@dataclass
class DecisionConfig:
    mode: str = "fixed"
    pooling: str = "decision_token"
    max_options: int = 8
    temperature_init: float = 1.0
    min_temperature: float = 0.05
    learn_temperature: bool = True


@dataclass
class CalibrationConfig:
    enabled: bool = True
    hidden_size: int = 128


@dataclass
class PolicyConfig:
    route_threshold: float = 0.85
    clarify_threshold: float = 0.50
    min_margin: float = 0.15
    max_entropy: float = 1.0
    ood_threshold: float = 0.0


@dataclass
class DataConfig:
    tokenizer: str = "simple"
    labels_path: str = "data/labels.json"
    min_freq: int = 1


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 32
    grad_accum_steps: int = 1
    head_warmup_epochs: int = 0
    unfreeze_fraction: float = 1.0
    lr: float = 1e-3
    backbone_lr: float = 1e-3
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    alpha_brier: float = 0.5
    beta_conf: float = 0.5
    patience: int = 6
    monitor: str = "val_f1"
    amp_dtype: str = "none"
    device: str = "auto"

    def __post_init__(self):
        self.betas = tuple(self.betas)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Invalid config: {msg}")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        unknown = set(d) - set(SECTIONS)
        _check(not unknown, f"unknown sections {sorted(unknown)}")
        cfg = cls(**{k: SECTIONS[k](**v) for k, v in d.items()})
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        m, d, t, p = self.model, self.decision, self.training, self.policy
        _check(m.backbone in ("scratch", "pretrained"), "model.backbone must be scratch|pretrained")
        if m.backbone == "scratch":
            _check(m.hidden_size % m.num_heads == 0, "hidden_size must be divisible by num_heads")
            _check((m.hidden_size // m.num_heads) % 2 == 0, "head dim must be even for RoPE")
        _check(m.num_classes > 1, "num_classes must be > 1")
        _check(m.max_seq_len > 0, "max_seq_len must be > 0")
        _check(m.vocab_size >= 0, "vocab_size must be >= 0")
        _check(0.0 <= m.dropout < 1.0, "dropout must be in [0,1)")
        _check(d.mode in ("fixed", "dynamic"), "decision.mode must be fixed|dynamic")
        _check(d.pooling in ("last", "decision_token", "mean", "attention"), "bad decision.pooling")
        _check(d.max_options >= 2, "max_options must be >= 2")
        _check(d.temperature_init > 0 and d.min_temperature > 0, "temperature must be > 0")
        _check(self.data.tokenizer in ("simple", "hf"), "data.tokenizer must be simple|hf")
        _check(0.0 <= t.unfreeze_fraction <= 1.0, "unfreeze_fraction must be in [0,1]")
        _check(0 <= t.head_warmup_epochs <= t.epochs, "head_warmup_epochs must be in [0,epochs]")
        _check(t.amp_dtype in ("none", "bfloat16", "float16"), "bad amp_dtype")
        _check(t.grad_accum_steps >= 1 and t.batch_size >= 1, "bad batch settings")
        _check(p.clarify_threshold <= p.route_threshold, "clarify_threshold must be <= route_threshold")


SECTIONS = {
    "model": ModelConfig,
    "decision": DecisionConfig,
    "calibration": CalibrationConfig,
    "policy": PolicyConfig,
    "data": DataConfig,
    "training": TrainConfig,
}


def load_config(path: str) -> Config:
    import yaml

    with open(path, encoding="utf-8") as f:
        return Config.from_dict(yaml.safe_load(f) or {})
