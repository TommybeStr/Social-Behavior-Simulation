from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"  # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("/home/zss/Social_Behavior_Simulation/checkpoints/default/global_step_2079")
model = AutoModelForCausalLM.from_pretrained("/home/zss/Social_Behavior_Simulation/checkpoints/default/global_step_2079", device_map="auto").eval()

# 初始化多轮对话的消息列表
messages = [
    {"role": "system", "content": "你是一个社交媒体互动预测专家，能够根据输入博文的具体内容，预测该条博文的互动情况。你现在收到的输入包括以下字段：- user_name: 原始发布者用户名 - user_interests: 原始发布者兴趣 - content: 博文正文 - depth: 博文在网络中的深度 - historical_interactors: 历史活跃用户 - potential_interactors: 潜在活跃用户列表（你只能从中选人进行预测）你必须严格按照以下格式输出，不允许包含任何解释性内容，也不要展示推理过程：[{\\\"user_name\\\": \\\"用户名（来自potential_interactors）\\\", \\\"content\\\": \\\"预测的评论内容\\\", \\\"type\\\": \\\"评论 或 转发\\\"}, ...] 注意事项：1. 你必须且只能从 potential_interactors 中选择用户填入输出结果；2. type 字段只能为 \\\"评论\\\" 或 \\\"转发\\\"；3. 不允许添加任何说明、理由、分析等内容；4. 输出必须是且只含一个合法的JSON数组结构。\'"}
    #{"role": "system", "content": "你是通义千问，一个人工智能助手"}
]

print("欢迎使用Qwen多轮对话助手。输入 'exit' 结束对话。")
while True:
    # 获取用户输入
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("对话结束。")
        break

    # 添加用户消息到对话历史
    messages.append({"role": "user", "content": user_input})

    # 将消息列表转换为文本格式，添加assistant的生成提示
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 模型生成回复
    generated_ids = model.generate(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,  # 🔧 显式传入 attention mask
    max_new_tokens=2048
)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 打印模型回复
    print(f"Qwen: {response}")

    # 添加模型回复到对话历史
    messages.append({"role": "assistant", "content": response})
