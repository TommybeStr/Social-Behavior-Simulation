import json
import pandas as pd
import argparse


def generate_qa(node, ancestors):
    """
    对单个节点生成 Q/A 对及其子节点的 Q/A 列表。
    ancestors: 上级节点的用户 id 列表，用于计算历史活跃用户
    返回值：列表，每项为 {'question': ..., 'answer': ...}
    """
    # 获取文本内容
    text = node.get('text_raw')
    # 判断文本性质
    if text == '快转微博':
        nature = '转发'
    else:
        # ancestors 为空表示顶层博文
        nature = '原创博文' if not ancestors else '评论'

    # 当前节点用户 ID
    user_id = node.get('user', {}).get('id', '')

    # 计算历史活跃用户：将祖先用户ID转换为字符串，并剔除自身
    hist_ids = [str(aid) for aid in ancestors if str(aid) != str(user_id)]
    hist_str = '无' if not hist_ids else ','.join(hist_ids)

    # 子回复列表
    children = node.get('comments', []) or []

    qa_pairs = []
    # 如果存在一级子回复，则生成当前节点的 Q/A 对
    if children:
        question = (
            "请你根据以下信息预测这条博文或评论下的评论情况：\n"
            f"用户id：{user_id}\n"
            f"文本内容：{text}\n"
            f"文本性质：{nature}\n"
            f"历史活跃用户：{hist_str}"
        )
        answer_parts = []
        for idx, c in enumerate(children, 1):
            c_text = c.get('text_raw')
            c_nature = '转发' if c_text == '快转微博' else '评论'
            c_user_id = c.get('user', {}).get('id', '')
            part = (
                f"评论{idx}：用户id：{c_user_id}\n"
                f"文本内容：{c_text}\n"
                f"文本性质：{c_nature}"
            )
            answer_parts.append(part)
        answer = "\n\n".join(answer_parts)
        qa_pairs.append({'question': question, 'answer': answer})

    # 对每个一级回复递归生成其 Q/A，传入新的祖先列表
    for child in children:
        qa_pairs.extend(generate_qa(child, ancestors + [user_id]))

    return qa_pairs


def main(input_file, output_json, output_parquet):
    all_qa = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            record = json.loads(line)
            # 首次调用时 ancestors 为空
            all_qa.extend(generate_qa(record, []))

    # 保存为 JSON
    with open(output_json, 'w', encoding='utf-8') as fout:
        json.dump(all_qa, fout, ensure_ascii=False, indent=2)

    # 保存为 Parquet
    df = pd.DataFrame(all_qa)
    df.to_parquet(output_parquet, index=False)

    print(f"共生成 {len(all_qa)} 条 SFT 问答对")
    print(f"JSON 输出：{output_json}")
    print(f"Parquet 输出：{output_parquet}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='递归拆分微博 JSONL 为 SFT Q/A 对并输出 JSON/Parquet')
    parser.add_argument('input', help='输入 JSONL 文件路径')
    parser.add_argument('json_output', help='输出 JSON 文件路径')
    parser.add_argument('parquet_output', help='输出 Parquet 文件路径')
    args = parser.parse_args()
    main(args.input, args.json_output, args.parquet_output)