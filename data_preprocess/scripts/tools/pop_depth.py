import pyarrow as pa
import pyarrow.parquet as pq
import json

def remove_depth_from_messages(messages):
    """
    移除 messages 中每条消息的 'depth' 字段
    """
    for message in messages:
        message.pop('depth', None)  # 移除 'depth' 字段
    return messages

def process_parquet(input_parquet_path, output_parquet_path):
    # 读取 Parquet 文件
    table = pq.read_table(input_parquet_path)
    
    # 获取所有的行数据
    rows = table.to_pandas()
    
    # 处理每一行
    processed_rows = []
    for index, row in rows.iterrows():
        # 处理 row['messages']，检查其类型并转换
        messages = row['messages']
        
        # 如果是 ndarray 类型，直接处理为列表
        if isinstance(messages, pa.lib.Tensor):
            messages = messages.to_pandas().tolist()
        # 如果是字符串类型，尝试加载为 JSON
        elif isinstance(messages, str):
            messages = json.loads(messages)
        
        # 移除 messages 中每个字典的 'depth' 字段
        messages = remove_depth_from_messages(messages)
        
        # 更新处理后的行数据
        processed_rows.append({
            'id': row['id'],  # 保留 id 字段
            'messages': json.dumps(messages, ensure_ascii=False)  # 重新序列化 messages 字段
        })
    
    # 创建一个新的 Parquet 表
    new_table = pa.table(processed_rows)
    
    # 写回新的 Parquet 文件
    pq.write_table(new_table, output_parquet_path)

# 示例：使用脚本读取一个 Parquet 文件并去除其中的 'depth' 字段
input_parquet_path = '/home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/split_full_depth_removed_val.parquet'  # 输入文件路径
output_parquet_path = '/home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/split_full_depth_removed_val.parquet'  # 输出文件路径

process_parquet(input_parquet_path, output_parquet_path)
