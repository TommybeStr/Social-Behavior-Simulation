import json
import pandas as pd
import argparse
from collections import defaultdict

SYSTEM_PROMPT = (
    "你是一个社交媒体互动预测专家，能够根据博文或评论的信息预测其下的评论情况。"
)

def load_user_profile_map(interest_file_path, mapping_file_path):
    with open(interest_file_path, 'r', encoding='utf-8') as f1:
        interest_data = json.load(f1)
    with open(mapping_file_path, 'r', encoding='utf-8') as f2:
        mapping_data = json.load(f2)

    profile_map = {}
    for real_id, anon_id in mapping_data.items():
        anon_entry = interest_data.get(str(anon_id), {})
        interests = anon_entry.get('user_interests', [])
        description = anon_entry.get('user_description', "")
        profile_map[str(real_id)] = {
            "interests": interests,
            "description": description
        }
    return profile_map


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


def split_potential_users(potentials, num_splits=10):
    pot_list = sorted(potentials)
    n = len(pot_list)
    if n == 0:
        return []
    avg = max(1, n // num_splits)
    chunks = [set(pot_list[i:i+avg]) for i in range(0, n, avg)]
    return chunks[:num_splits]


def make_question_dict(node, ancestors, potential, profile_map):
    text = node.get('text_raw') or node.get('text') or ''
    nature = '转发' if text in ('转发微博', '转发') else ('原创博文' if not ancestors else '评论')
    user_id = str(node.get('user', {}).get('id', ''))
    hist_ids = [str(a) for a in ancestors]
    # 构建结构化潜在活跃用户列表
    pot_profiles = []
    for pid in sorted(potential):
        prof = profile_map.get(pid, {})
        pot_profiles.append({
            '用户id': pid,
            '用户兴趣': prof.get('interests', []),
            '用户简介': prof.get('description', '')
        })
    interests = profile_map.get(user_id, {}).get("interests", [])
    description = profile_map.get(user_id, {}).get("description", "")
    return {
        '用户id': user_id,
        '用户兴趣': interests,
        '用户简介': description,
        '文本内容': text,
        '文本性质': nature,
        '历史活跃用户': hist_ids,
        '潜在活跃用户': pot_profiles
    }


def make_answer_list(children, subset, profile_map):
    results = []
    for c in children:
        c_user = str(c.get('user', {}).get('id', ''))
        if c_user not in subset:
            continue
        c_text = c.get('text_raw') or c.get('text') or ''
        c_nature = '转发' if c_text in ('转发微博', '转发') else '评论'
        profile = profile_map.get(c_user, {})
        results.append({
            '用户id': c_user,
            '用户兴趣': profile.get("interests", []),
            '用户简介': profile.get("description", ""),
            '文本内容': c_text,
            '文本性质': c_nature
        })
    return results


def traverse_conversation(node, ancestors, subset, profile_map, messages):
    depth = len(ancestors)
    q_dict = make_question_dict(node, ancestors, subset, profile_map)
    children = node.get('comments', []) or []
    a_list = make_answer_list(children, subset, profile_map)

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

    parent_id = str(node.get('user', {}).get('id', ''))
    for child in children:
        c_user = str(child.get('user', {}).get('id', ''))
        if c_user in subset:
            traverse_conversation(child, ancestors + [parent_id], subset, profile_map, messages)


def main(input_file, output_jsonl, output_parquet):
    with open(input_file, 'r', encoding='utf-8') as fin:
        records = [json.loads(line) for line in fin if line.strip()]

    profile_map = load_user_profile_map(
        '/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/ruby_face_cream_profile.graph.anon',
        '/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/id_dict.json'
    )

    interactors = collect_all_interactors(records)

    samples = []
    for record in records:
        root_user = str(record.get('user', {}).get('id', ''))
        potential = interactors.get(root_user, set())
        subsets = split_potential_users(potential, num_splits=100)

        for part_idx, subset in enumerate(subsets):
            messages = [
                {
                    'role': 'system',
                    'content': json.dumps(SYSTEM_PROMPT, ensure_ascii=False),
                    'loss': 0,
                    'depth': 0
                }
            ]
            traverse_conversation(record, [], subset, profile_map, messages)
            total = ''.join(json.dumps(m['content'], ensure_ascii=False) for m in messages)
            seq_len = min(len(total), 8192)
            samples.append({
                'id': f"{record.get('id')}_part{part_idx}",
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
        } for s in samples
    ])
    df.to_parquet(output_parquet, index=False)

    print(f"共生成 {len(samples)} 条多层级样本")
    print(f"JSONL 输出: {output_jsonl}")
    print(f"Pretty JSON 输出: {pretty}")
    print(f"Parquet 输出: {output_parquet}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成包含多层级评论结构的样本，每份潜在活跃用户为1/10，包含兴趣与简介')
    parser.add_argument('input', help='输入 JSONL 文件路径')
    parser.add_argument('json_output', help='输出 JSONL 文件路径')
    parser.add_argument('parquet_output', help='输出 Parquet 文件路径')
    args = parser.parse_args()
    main(args.input, args.json_output, args.parquet_output)
