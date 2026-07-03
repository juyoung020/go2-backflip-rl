# Export the MuJoCo-finetuned rsl_rl checkpoint's actor as ONNX.
import torch
import torch.nn as nn

ck = torch.load("logs/mujoco_finetune/model_5697.pt", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
net = nn.Sequential(
    nn.Linear(43, 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 12),
)
new_sd = {}
for k in sd:
    if k.startswith("actor."):
        parts = k.split(".")  # actor.<idx>.<param>
        new_sd[f"{parts[1]}.{parts[2]}"] = sd[k]
net.load_state_dict(new_sd)
net.eval()
torch.onnx.export(net, torch.zeros(1, 43), "logs/mujoco_finetune/policy_mj.onnx",
                  input_names=["obs"], output_names=["actions"], dynamo=False)
print("exported logs/mujoco_finetune/policy_mj.onnx")
