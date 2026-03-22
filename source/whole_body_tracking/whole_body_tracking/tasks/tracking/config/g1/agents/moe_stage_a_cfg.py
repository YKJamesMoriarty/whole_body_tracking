from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1FlatMoEStageAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 200
    experiment_name = "g1_flat_moe_stage_a"
    empirical_normalization = False

    policy = RslRlPpoActorCriticRecurrentCfg(
        # match mimic-style exploration scale
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # encourage exploration early (mimic uses 0.005)
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
