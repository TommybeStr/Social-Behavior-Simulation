import torch
state = torch.load("/home/zss/Social_Behavior_Simulation/checkpoints/default/renew_10.29/global_step_1/cls_head0.pt", map_location="cpu")
print("Keys:", state.keys())
print("Weight shape:", state["weight"].shape)
print("Bias shape:", state["bias"].shape)