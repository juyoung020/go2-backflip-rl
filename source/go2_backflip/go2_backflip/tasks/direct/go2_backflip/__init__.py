"""Go2 backflip curriculum tasks (Stage 1-4)."""

import gymnasium as gym

from . import agents

_STAGES = {
    "Isaac-Backflip-Go2-Stage1-Direct-v0": "Go2BackflipStage1Cfg",
    "Isaac-Backflip-Go2-Stage2-Direct-v0": "Go2BackflipStage2Cfg",
    "Isaac-Backflip-Go2-Stage3-Direct-v0": "Go2BackflipStage3Cfg",
    "Isaac-Backflip-Go2-Stage4-Direct-v0": "Go2BackflipStage4Cfg",
    "Isaac-Backflip-Go2-Stage5-Direct-v0": "Go2BackflipStage5Cfg",
    "Isaac-Backflip-Go2-Stage6-Direct-v0": "Go2BackflipStage6Cfg",
    # alias: the full-backflip task
    "Isaac-Backflip-Unitree-Go2-Direct-v0": "Go2BackflipStage3Cfg",
}

for task_id, cfg_name in _STAGES.items():
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.go2_backflip_env:Go2BackflipEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_backflip_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BackflipPPORunnerCfg",
        },
    )
