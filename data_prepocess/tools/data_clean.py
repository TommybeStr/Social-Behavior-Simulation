import json
import pandas as pd
import argparse
from collections import defaultdict

SYSTEM_PROMPT = (
    "你是一个社交媒体互动预测专家，能够根据博文或评论的信息预测其下的评论情况。"
)

# 加载用户兴趣映射
def load_user_interest_map(interest_file_path, mapping_file_path):
    with open(interest_file_path, 'r', encoding='utf-8') as f1:
        interest_data = json.load(f1)
    with open(mapping_file_path, 'r', encoding='utf-8') as f2:
        mapping_data = json.load(f2)

    real_to_interest = {}
    for real_id, anon_id in mapping_data.items():
        anon_entry = interest_data.get(str(anon_id), {})
        interests = anon_entry.get('user_interests', [])
        real_to_interest[str(real_id)] = interests
    return real_to_interest


def collect_all_interactors(records):
    interactors = defaultdict(set)

    def dfs(node, root_user):
        for c in node.get('comments', []) or []:
            uid = str(c.get('user', {}).get('id', ''))
            if uid:
                interactors[root_user].add(uid)
            dfs(c, root_user)

    for record in records:
        root = str(record.get('user', {}).get('id', ''))
        dfs(record, root)
    return interactors


def make_question_dict(node, ancestors, potential, interest_map):
    text = node.get('text_raw') or node.get('text') or ''
    nature = '转发' if text == ('转发微博' or '转发') else ('原创博文' if not ancestors else '评论')
    user_id = str(node.get('user', {}).get('id', ''))
    hist_ids = [str(a) for a in ancestors]
    pot_ids = sorted(potential)
    depth = len(ancestors)
    interests = interest_map.get(user_id, [])
    return {
        '用户id': user_id,
        '文本内容': text,
        '文本性质': nature,
        '历史活跃用户': hist_ids,
        '潜在活跃用户': pot_ids,
        '用户兴趣': interests,
        'depth': depth
    }


def make_answer(node, interest_map):
    children = node.get('comments', []) or []
    structured = []
    for c in children:
        c_text = c.get('text_raw') or c.get('text') or ''
        c_nature = '转发' if c_text == ('转发微博' or '转发') else '评论'
        c_user = str(c.get('user', {}).get('id', ''))
        c_interest = interest_map.get(c_user, [])
        structured.append({
            '用户id': c_user,
            '文本内容': c_text,
            '文本性质': c_nature,
            '用户兴趣': c_interest
        })
    return structured


def collect_conversation_bfs(root, potential, interest_map):
    qas = []
    queue = [(root, [])]
    while queue:
        next_queue = []
        for node, ancestors in queue:
            depth = len(ancestors)
            q_dict = make_question_dict(node, ancestors, potential, interest_map)
            a_list = make_answer(node, interest_map)
            qas.append((q_dict, a_list, depth))
            parent_id = node.get('user', {}).get('id', '')
            for child in node.get('comments', []) or []:
                next_queue.append((child, ancestors + [parent_id]))
        queue = next_queue
    return qas


def main(input_file, output_jsonl, output_parquet):
    # 加载微博数据
    with open(input_file, 'r', encoding='utf-8') as fin:
        records = [json.loads(line) for line in fin]

    # 加载兴趣映射
    interest_map = load_user_interest_map(
        '/home/zss/Social_Behavior_Simulation/data_prepocess/ruby_face_cream_profile.graph.anon',
        '/home/zss/Social_Behavior_Simulation/data_prepocess/id_dict.json'
    )

    # 提取潜在活跃用户
    interactors = collect_all_interactors(records)

    samples = []
    for record in records:
        root_user = str(record.get('user', {}).get('id', ''))
        potential = interactors.get(root_user, set())
        qas = collect_conversation_bfs(record, potential, interest_map)

        messages = []
        messages.append({
            'role': 'system',
            'content': json.dumps(SYSTEM_PROMPT, ensure_ascii=False),
            'loss': 0,
            'depth': 0
        })

        for q_dict, a_list, depth in qas:
            messages.append({
                'role': 'user',
                'content': json.dumps(q_dict, ensure_ascii=False),
                'loss': 0,
                'depth': depth
            })
            messages.append({
                'role': 'assistant',
                'content': json.dumps(a_list, ensure_ascii=False),
                'loss': 1,
                'depth': depth + 1
            })

        total = ''.join(
            json.dumps(msg['content'], ensure_ascii=False) if not isinstance(msg['content'], str) else msg['content']
            for msg in messages
        )
        seq_len = min(len(total), 8192)
        samples.append({
            'id': record.get('id'),
            'messages': messages,
            'seq_len': seq_len
        })

    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for item in samples:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    pretty = output_jsonl.replace('.jsonl', '_pretty.json')
    with open(pretty, 'w', encoding='utf-8') as fout:
        json.dump(samples, fout, ensure_ascii=False, indent=2)

    df = pd.DataFrame([
        {
            'id': s['id'],
            'messages': json.dumps(s['messages'], ensure_ascii=False),
            'seq_len': s['seq_len']
        }
        for s in samples
    ])
    df.to_parquet(output_parquet, index=False)

    print(f"共生成 {len(samples)} 条训练样本")
    print(f"JSONL 输出: {output_jsonl}")
    print(f"Pretty JSON 输出: {pretty}")
    print(f"Parquet 输出: {output_parquet}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将微博树状评论展开成单条 SFT 样本并输出 JSONL/Parquet')
    parser.add_argument('input', help='输入 JSONL 文件路径')
    parser.add_argument('json_output', help='输出 SFT JSONL 文件路径')
    parser.add_argument('parquet_output', help='输出 Parquet 文件路径')
    args = parser.parse_args()
    main(args.input, args.json_output, args.parquet_output)
