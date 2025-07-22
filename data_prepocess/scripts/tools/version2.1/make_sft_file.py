import json
import pandas as pd
import argparse
from collections import defaultdict, deque

SYSTEM_PROMPT = ("你是一个社交媒体互动预测专家，能够根据输入博文的具体内容，预测该条博文的互动情况。你现在收到的输入包括以下字段：- user_name: 原始发布者用户名 - user_interests: 原始发布者兴趣 - content: 博文正文 - depth: 博文在网络中的深度 - historical_interactors: 历史活跃用户 - potential_interactors: 潜在活跃用户列表（你只能从中选人进行预测）你必须严格按照以下格式输出，不允许包含任何解释性内容，也不要展示推理过程：[{\"user_name\": \"用户名（来自potential_interactors）\", \"content\": \"预测的评论内容\", \"type\": \"评论 或 转发\"}, ...] 注意事项：1. 你必须且只能从 potential_interactors 中选择用户填入输出结果；2. type 字段只能为 \"评论\" 或 \"转发\"；3. 不允许添加任何说明、理由、分析等内容；4. 输出必须是且只含一个合法的JSON数组结构。")


def collect_all_nodes_dfs(node):
    """深度优先收集所有节点"""
    nodes = [node]
    for child in node.get('replies', []) or []:
        nodes.extend(collect_all_nodes_dfs(child))
    return nodes


def split_tree_into_chunks_by_path_accumulating(root_node, max_users_per_chunk=10):
    """
    每次完整走完一条从当前节点到叶子的路径后，判断是否超过用户数限制，控制chunk切割。
    保证不会重复添加节点。
    """
    chunks = []
    current_chunk_nodes = []
    current_chunk_users = set()
    seen_node_ids = set()  # 记录当前chunk中已经添加的节点

    def dfs(node, path_nodes, path_users):
        nonlocal current_chunk_nodes, current_chunk_users, seen_node_ids

        user = str(node.get('user'))
        new_path_nodes = path_nodes + [node]
        new_path_users = set(path_users)
        if user:
            new_path_users.add(user)

        # 到达叶子节点
        if not node.get('replies'):
            combined_users = current_chunk_users.union(new_path_users)
            if len(combined_users) > max_users_per_chunk:
                # 超过用户数，保存当前chunk
                if current_chunk_nodes:
                    chunks.append({
                        'nodes': current_chunk_nodes,
                        'users': current_chunk_users
                    })
                # 启动新chunk
                current_chunk_nodes = []
                current_chunk_users = set()
                seen_node_ids = set()

            # 加入当前路径中未添加过的节点
            for n in new_path_nodes:
                node_id = id(n)
                if node_id not in seen_node_ids:
                    current_chunk_nodes.append(n)
                    seen_node_ids.add(node_id)

            current_chunk_users.update(new_path_users)
            return

        # 非叶子节点，继续向下
        for child in node.get('replies', []) or []:
            dfs(child, new_path_nodes, new_path_users)

    dfs(root_node, [], set())

    # 收尾
    if current_chunk_nodes:
        chunks.append({
            'nodes': current_chunk_nodes,
            'users': current_chunk_users
        })

    return chunks


def make_question_dict(node, ancestors, potential):
    content = node.get('content')
    user_name = str(node.get('user'))
    hist_ids = [str(a) for a in ancestors]
    
    # 构建结构化潜在活跃用户列表
    pot_profiles = []
    for pid in sorted(potential):
        pot_profiles.append({
            'user_name': pid,
            '用户兴趣': node.get('interests'),
        })
    
    interests = node.get('interests')
    depth = node.get('depth')
    
    return {
        'user_name': user_name,
        'interests': interests,
        'content': content,
        'depth': depth,
        'historical_interactors': hist_ids,
        'potential_interactors': pot_profiles
    }


def make_answer_list(children):
    results = []
    for c in children:
        c_user = str(c.get('user'))
        c_content = c.get('content')
        if not c_content: 
            c_content = "以上用户都不感兴趣，没有发生任何交互"
        c_nature = c.get('type')
        results.append({
            'user_name': c_user,
            'content': c_content,
            'type': c_nature
        })
    return results


from collections import defaultdict, deque
import json

def process_chunk_breadth_first(chunk_nodes, chunk_users):
    """
    对一个chunk内的节点进行广度优先处理，生成消息对
    """
    messages = []

    # ID映射，确保引用来自chunk_nodes内部（避免直接用树原始child对象）
    id_to_node = {id(node): node for node in chunk_nodes}
    chunk_node_ids = set(id_to_node.keys())

    # 构建 node_children 映射
    node_children = defaultdict(list)
    for node in chunk_nodes:
        node_id = id(node)
        for child in node.get('replies', []) or []:
            child_id = id(child)
            if child_id in chunk_node_ids:
                node_children[node_id].append(id_to_node[child_id])  # ⚠️ 使用chunk内的引用

    # 找根节点（未被任何人引用的）
    referenced_ids = {id(c) for children in node_children.values() for c in children}
    root_nodes = [n for n in chunk_nodes if id(n) not in referenced_ids]
    if not root_nodes and chunk_nodes:
        root_nodes = [chunk_nodes[0]]

    # 广度优先生成对话
    queue = deque([(node, []) for node in root_nodes])
    while queue:
        current_level_nodes = []
        for _ in range(len(queue)):
            node, ancestors = queue.popleft()
            current_level_nodes.append((node, ancestors))
            user_id = str(node.get('user'))
            for child in node_children.get(id(node), []):
                queue.append((child, ancestors + [user_id]))

        for node, ancestors in current_level_nodes:
            q_dict = make_question_dict(node, ancestors, chunk_users)
            children = node_children.get(id(node), [])
            a_list = make_answer_list(children)

            if not a_list:
                a_list = ["以上用户都不感兴趣，没有发生任何交互"]

            messages.append({
                'role': 'user',
                'content': json.dumps(q_dict, ensure_ascii=False),
                'loss': 0,
            })
            messages.append({
                'role': 'assistant',
                'content': json.dumps(a_list, ensure_ascii=False),
                'loss': 1,
            })

    return messages




def main(input_file, output_jsonl, output_parquet):
    with open(input_file, 'r', encoding='utf-8') as fin:
        records = json.load(fin)

    samples = []
    
    for record in records:
        # 将每棵树分割成多个chunk
        chunks = split_tree_into_chunks_by_path_accumulating(record, max_users_per_chunk=10)
        
        for chunk_idx, chunk in enumerate(chunks):
            messages = [
                {
                    'role': 'system',
                    'content': json.dumps(SYSTEM_PROMPT, ensure_ascii=False),
                    'loss': 0,
                    'depth': 0
                }
            ]
            
            # 处理当前chunk，生成消息对
            chunk_messages = process_chunk_breadth_first(chunk['nodes'], chunk['users'])
            messages.extend(chunk_messages)
            
            samples.append({
                'id': f"{record.get('id')}_chunk_{chunk_idx}",
                'messages': messages,
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
        } for s in samples
    ])
    df.to_parquet(output_parquet, index=False)

    print(f"共生成 {len(samples)} 条chunk样本")
    print(f"JSONL 输出: {output_jsonl}")
    print(f"Pretty JSON 输出: {pretty}")
    print(f"Parquet 输出: {output_parquet}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成包含多层级评论结构的样本,包含兴趣与简介')
    parser.add_argument('--input', help='输入 JSONL 文件路径')
    parser.add_argument('--json_output', help='输出 JSONL 文件路径')
    parser.add_argument('--parquet_output', help='输出 Parquet 文件路径')
    args = parser.parse_args()
    main(args.input, args.json_output, args.parquet_output)