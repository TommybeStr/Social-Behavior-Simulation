import json

# 读取 JSONL 文件的前 50 行
def read_jsonl(file_path, num_lines=463):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(file.readline()) for _ in range(num_lines)]
    return lines

# 计算每个 depth 下 gold 为空的数量
def count_empty_gold(lines):
    depth_gold_count = {}
    num=0
    for line in lines:
        depth = line.get("depth", -1)  # 获取当前行的 depth
        gold = line.get("gold", [])  # 获取当前行的 gold

        if not gold:  # 如果 gold 为空
            if depth in depth_gold_count:
                depth_gold_count[depth] += 1
                
            else:
                depth_gold_count[depth] = 1
                
        num+=1
    return depth_gold_count

# 主函数
def main(file_path):
    lines = read_jsonl(file_path)
    empty_gold_count = count_empty_gold(lines)
    
    print("每个 depth 下 gold 为空的数量:")
    for depth, count in empty_gold_count.items():
        print(f"Depth {depth}: {count}")

# 示例使用
file_path = '/home/zss/Social_Behavior_Simulation/checkpoints/evaluate/sft_evaluate_detail.jsonl'  # 请替换为你的文件路径
main(file_path)
