import json
target_ids = {
    #"4985835172660478", "4978211733635161", "4977128750189194",
    #"4936520266422249", "4981905626500053", "4854240176512214",
    #"4984032418334725", "4968720760047790"
"4863694426605567"
}
matched = []
input_file='/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/Ruby_Face_Cream_r0.jsonl'
with open(input_file, 'r', encoding='utf-8') as fin:
        records = [json.loads(line) for line in fin if line.strip()]
        for record in records:
                id = record.get('idstr')
                if id in target_ids:
                        matched.append(record)

with open('/home/zss/Social_Behavior_Simulation/item.json', 'w', encoding='utf-8') as fout:
        json.dump(matched, fout, ensure_ascii=False, indent=2)
            