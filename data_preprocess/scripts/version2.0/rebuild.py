#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta

# ======================
# 新增：时间解析与一周窗口筛选
# ======================
DATE_FMT_CANDIDATES = [
    "%a %b %d %H:%M:%S %z %Y",   # Wed Dec 20 10:35:09 +0800 2023
    "%a %b %e %H:%M:%S %z %Y",   # 部分平台支持%e（空格补齐的日期）
]

def parse_created_at(s):
    """
    尝试解析类似 'Wed Dec 20 10:35:09 +0800 2023' 的时间字符串。
    成功返回带tzinfo的datetime；失败返回None（不改变原有构造逻辑的稳定性）。
    """
    if not isinstance(s, str):
        return None
    for fmt in DATE_FMT_CANDIDATES:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # 宽松兜底：把连续空格压成单空格再试一次主格式
    try:
        s2 = " ".join(s.split())
        return datetime.strptime(s2, DATE_FMT_CANDIDATES[0])
    except Exception:
        return None


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
                "user_id": c['user']['id'],  # 保持原逻辑不改动
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

def process_reposts(reposts, profile_map):
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

def process_single_post(post, profile_map):
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
        post_node["replies"].extend(build_comment_hierarchy(comments, profile_map))

    # 转发部分
    reposts = post.get("reposts", [])
    if reposts:
        post_node["replies"].extend(process_reposts(reposts, profile_map))

    return post_node

def main(input_path, output_path):
    profile_map = load_user_profile_map(
        '/home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_raw/ruby_face_cream_profile.graph.anon',
        '/home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_raw/id_dict.json'
    )

    # ===========
    # 第一遍：读入全部记录，收集 (user_id, created_at_dt)，并计算每个博主的最新时间
    # ===========
    raw_records = []  # 保存 (record_dict, created_dt_or_None, user_id_str)
    latest_by_user = {}  # user_id_str -> latest_dt

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user = record.get("user", {}) or {}
            uid_str = str(user.get("id"))
            created_dt = parse_created_at(record.get("created_at"))

            raw_records.append((record, created_dt, uid_str))

            if created_dt is not None and uid_str != "None":
                prev = latest_by_user.get(uid_str)
                if (prev is None) or (created_dt > prev):
                    latest_by_user[uid_str] = created_dt

    # ===========
    # 第二遍：仅对“在该博主最新时间点向前7天内”的博文进行构造
    # 规则：若该博主存在有效latest_dt且当前记录created_dt有效，
    #       则要求 created_dt >= latest_dt - 7天 才保留；
    #       若二者任一无效，则为不破坏旧逻辑，默认保留（可按需改为丢弃，这里遵循“其他逻辑不更改”的温和策略）。
    # ===========
    one_week = timedelta(days=7)
    all_posts = []

    for record, created_dt, uid_str in raw_records:
        latest_dt = latest_by_user.get(uid_str)

        keep = True
        if (latest_dt is not None) and (created_dt is not None):
            keep = (created_dt >= (latest_dt - one_week))

        # 若无法判断时间（某条缺失created_at或该用户没有可用latest），保持原有构造行为：保留
        if not keep:
            continue

        node = process_single_post(record, profile_map)
        all_posts.append(node)

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(all_posts, fout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input.jsonl output.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
