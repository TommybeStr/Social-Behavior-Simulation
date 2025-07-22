import json
import pandas as pd

# 1. 读取整理后的 JSON 文件
with open('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/rebuild_data.json', 'r', encoding='utf-8') as f:
    posts = json.load(f)

# 2. 构建潜在活跃用户映射：对每位博主，收集其所有帖子下的互动用户（包括博主自己）
potential_map = {}
for post in posts:
    author = post['user']
    pset = set()
    # 加入博主自己
    pset.add((post['user'], tuple(post['interests'])))
    
    def collect_interactors(node):
        for child in node.get('replies', []):
            pset.add((child['user'], tuple(child['interests'])))
            collect_interactors(child)
    
    collect_interactors(post)
    potential_map[author] = [
        {"user_name": u, "user_interests": list(interests)}
        for u, interests in pset
    ]

# 3. 遍历所有节点，构造 SFT 问答对
records = []

def traverse(node, root_author, parent=None):
    # 历史活跃用户
    if parent is None:
        hist = []
    else:
        hist = [{"user_name": parent['user'], "user_interests": parent['interests']}]
    
    # 输出列表：当前节点的所有直接回复或转发
    outputs = [
        {"user_name": child['user'], "content": child['content'], "type": child['type']}
        for child in node.get('replies', [])
    ]
    
    # 构造一条记录
    records.append({
        "user_name": node['user'],
        "user_interests": node['interests'],
        "content": node['content'],
        "depth": node['depth'],
        "historical_interactors": hist,
        "potential_interactors": potential_map[root_author],
        "output": outputs
    })
    
    # 递归处理子节点
    for child in node.get('replies', []):
        traverse(child, root_author, parent=node)

# 对每条博文及其子节点应用
for post in posts:
    traverse(post, root_author=post['user'])

# 4. 转成 DataFrame 并保存为 Parquet和json
df = pd.DataFrame(records)

df.to_parquet('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/sft_file_raw.parquet', engine='pyarrow', index=False)
with open('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/sft_file_raw.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
