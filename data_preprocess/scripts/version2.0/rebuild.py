#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from collections import defaultdict

def load_user_profile_map(interest_file_path, mapping_file_path):
    """
    载入用户兴趣映射：real_id -> {"interests": [...]}
    """
    with open(interest_file_path, 'r', encoding='utf-8') as f1:
        interest_data = json.load(f1)
    with open(mapping_file_path, 'r', encoding='utf-8') as f2:
        mapping_data = json.load(f2)

    profile_map = {}
    for real_id, anon_id in mapping_data.items():
        anon_entry = interest_data.get(str(anon_id), {})
        interests = anon_entry.get('user_interests', [])
        profile_map[str(real_id)] = {
            "interests": interests
        }
    return profile_map

def is_image_comment(text):
    """
    判断是否为图片评论
    """
    return isinstance(text, str) and text.strip().startswith("图片评论")


def build_comment_hierarchy(comments, profile_map):
    """
    扁平化创建所有评论节点，并根据 reply_comment 关系构建多层嵌套
    """
    nodes = {}
    children = defaultdict(list)

    # 1) 创建节点
    for c in comments:
        text = c.get('text_raw', '')
        if is_image_comment(text):
            continue
        cid = c['id']
        uid = str(c['user']['id'])
        nodes[cid] = {
            "id": cid,
            "user_id": c['user']['id'],
            "user": c['user']['screen_name'],
            "interests": profile_map.get(uid, {}).get("interests", []),
            "content": c['text_raw'],
            "type": "评论",
            "depth": None,
            "replies": []
        }
        # 二级及以上评论也先扁平记录
        for sub in c.get('comments', []):
            sub_text = sub.get('text_raw', '')
            if is_image_comment(sub_text):
                continue
            sid = sub['id']
            suid = str(sub['user']['id'])
            nodes[sid] = {
                "id": sid,
                "user_id": c['user']['id'],
                "user": sub['user']['screen_name'],
                "interests": profile_map.get(suid, {}).get("interests", []),
                "content": sub['text_raw'],
                "type": "评论",
                "depth": None,
                "replies": []
            }
            # reply_comment.id 指向真正的父评论，否则归到当前一级评论
            parent_id = sub.get('reply_comment', {}).get('id', cid)
            children[parent_id].append(sid)

    # 2) 递归挂载并设置 depth
    def attach(parent_id, parent_node):
        for child_id in children.get(parent_id, []):
            child = nodes[child_id]
            child["depth"] = parent_node["depth"] + 1
            parent_node["replies"].append(child)
            attach(child_id, child)

    # 3) 处理所有一级评论
    roots = []
    for c in comments:
        if is_image_comment(c.get('text_raw', '')):
            continue
        root = nodes[c['id']]
        root["depth"] = 1
        roots.append(root)
        attach(c['id'], root)

    return roots

def process_reposts(reposts,profile_map):
    """
    将转发扁平化为 depth=1 的列表
    """
    out = []
    for r in reposts:
        text = r.get("text_law", r.get("text_raw", ""))
        if is_image_comment(text):
            continue
        uid = str(r['user']['id'])
        out.append({
            "id": r["id"],
            "user_id": r['user']['id'],
            "user": r["user"]["screen_name"],
            "interests": profile_map.get(uid, {}).get("interests", []),
            "content": r.get("text_law", r.get("text_raw", "")),
            "type": "转发微博",
            "depth": 1,
            "replies": []
        })
    return out

def process_single_post(post,profile_map):
    """
    构建单条博文及其下所有评论、转发的嵌套结构
    """
    uid = str(post['user']['id'])
    post_node = {
        "id": post["id"],
        "user_id": post['user']['id'],
        "user": post["user"]["screen_name"],
        "interests": profile_map.get(uid, {}).get("interests", []),
        "content": post.get("text_raw", ""),
        "type": "原始博文",
        "depth": 0,
        "replies": []
    }

    # 评论部分（key 名为 'comments'）
    comments = post.get("comments", [])
    if comments:
        post_node["replies"].extend(build_comment_hierarchy(comments,profile_map))

    # 转发部分
    reposts = post.get("reposts", [])
    if reposts:
        post_node["replies"].extend(process_reposts(reposts,profile_map))

    return post_node

def main(input_path, output_path):
    profile_map = load_user_profile_map('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/ruby_face_cream_profile.graph.anon',  '/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/id_dict.json')
    all_posts = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            node = process_single_post(record, profile_map)
            all_posts.append(node)

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(all_posts, fout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input.jsonl output.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
