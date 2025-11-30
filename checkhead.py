from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/home/zss/Social_Behavior_Simulation/Qwen2.5-3B-Instruct")
print(model.config.num_hidden_layers)  # 例如输出 32