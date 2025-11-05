from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"  # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("/home/zss/Social_Behavior_Simulation/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/home/zss/Social_Behavior_Simulation/Qwen2.5-3B-Instruct", device_map="auto").eval()


PSEP_TOKEN = "<|psep|>"

# <POTENTIAL_SPANS> 成对包裹（明确放在 user.content 末尾）
PSEP_BLOCK_START = "\n<POTENTIAL_SPANS>\n"
PSEP_BLOCK_END   = "\n</POTENTIAL_SPANS>\n"

# 评论内容定位哨兵（供训练器精确对齐到“评论的 content 文本”）
CSTART_TOKEN = "<|cstart|>"
CEND_TOKEN   = "<|cend|>"
SYSTEM_PROMPT = f'''你是社交媒体互动预测专家。请基于给定的单条样本（一个 JSON 对象）对「每个候选互动对象」分别进行三分类预测：0=无互动，1=评论，2=转发微博；若为评论或转发，可给出简短内容。最终仅输出一个覆盖全部候选的 JSON 数组，顺序需与候选顺序严格一致。

【输入字段（单样本 JSON）】
- user_name：作者
- interests：作者兴趣（数组）
- content：正文文本。注意：正文末尾会追加一个特殊段落 `<POTENTIAL_SPANS>`，用于提供候选人信息。
- historical_interactors：与作者历史上发生过互动的用户名列表

【关于 <POTENTIAL_SPANS>】
- `<POTENTIAL_SPANS>` 紧跟在 content 末尾，并以 `</POTENTIAL_SPANS>` 结束（严格成对）。
- 其中每个候选用成对分隔符 `{PSEP_TOKEN}` 包裹：`{PSEP_TOKEN}{{候选JSON}}{PSEP_TOKEN}`。
- 候选 JSON 严格包含：`{{"user_name": "...", "interests": [...], "depth": 整数}}`。
- 这些候选块的先后顺序即为评分类与输出顺序的唯一依据；禁止重排、丢失或增添。

【唯一输出（严格格式）】
- 输出一个 JSON 数组，长度等于候选数量，顺序与 `<POTENTIAL_SPANS>` 中候选块顺序一致。
- 数组中每个元素的结构为：
  `{{"user_name": "...", "content": "{CSTART_TOKEN}...{CEND_TOKEN}", "type": 0/1/2}}`
  - type：0=无互动；1=评论；2=转发微博
  - content：
      * 当 type=1/2 时，用 `{CSTART_TOKEN}评论或转发语{CEND_TOKEN}` 包裹（允许为空内容但必须保留这对标记）；
      * 当 type=0 时，必须输出 `"{CSTART_TOKEN}{CEND_TOKEN}"`（空内容的成对标记）。
- 严禁输出 `{CSTART_TOKEN}` 或 `{CEND_TOKEN}` 之外的自造标记。

【判定要点（逐候选独立打分）】
1) 兴趣匹配：作者 interests 与候选 interests 的重合或主题相关度。
2) 历史关系：historical_interactors 是否包含该候选及其强/弱关系迹象。
3) 语义关联：正文 content 与候选兴趣或历史话题的语义相关度。
4) 层级 depth 衰减：`depth` 越深，互动概率通常越低，应体现衰减趋势。
5) 不臆测外部事实：仅依据样本提供的信息进行判断。

【硬性规则】
- 只能使用 `<POTENTIAL_SPANS>` 中出现的 `user_name`，严禁虚构或遗漏。
- 顺序必须与 `<POTENTIAL_SPANS>` 中候选顺序完全一致。
- 仅输出上述 JSON 数组；不得输出任何解释、前后缀或额外文本。
- JSON 必须可解析（双引号、逗号与括号合法）。
'''
# 初始化多轮对话的消息列表
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}

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
    max_new_tokens=4096
)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 打印模型回复
    print(f"Qwen: {response}")

    # 添加模型回复到对话历史
    messages.append({"role": "assistant", "content": response})
