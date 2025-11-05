# -*- coding: utf-8 -*-
"""
stats_from_logs.py  —  评估明细（--jsonl_detail）综合统计（仅读日志即可）

输出板块：
(I)  按 depth 的候选/真实/预测条目计数（面板）
(II) 以 step_reward 为准的 F1（gold 非空样本）与缺失统计
(III)树内合并（bag，不去重）节点级 F1（同树同层先累加计数再计算）
(IV) 图级评估（树级合并）：
     - rel_error（边权向量的相对误差）
     - 边存在性的精确率/召回率/边F1（集合口径）
     - 出/入度分布相似性（JSD、Spearman）
     - 边权强度分布（KS 距离、Gini、均值相对差）
(V)  Micro-averaged 节点级 F1（逐条算 TP/FP/FN，再在 depth 汇总）支持 set/bag
(VI) Micro-averaged 边级 F1（严格 child→parent）支持 set/bag

用法：
python stats_from_logs.py --input rollout_io_gold.jsonl [--depths 1 2] [--undirected_graph] [--micro_mode set|bag]
"""

import json
import argparse
from collections import defaultdict, Counter
import math
import re

# ===== 旧格式兼容 =====
NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"
_INTERACT_HEAD_RE = re.compile(r'^\s*\[\s*INTERACT\s*=\s*([01])\s*\]\s*(?:\r?\n)?', re.IGNORECASE)

def _peel_interact_header(s: str):
    if not isinstance(s, str):
        return None, ""
    m = _INTERACT_HEAD_RE.match(s)
    if not m:
        return None, s.strip()
    tag = int(m.group(1))
    body = s[m.end():].strip()
    return tag, body

# ===== 读 JSON/JSONL =====
def iter_records(path):
    """支持 JSONL 或 JSON 数组文件。"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict):
                            yield rec
                return
            except Exception:
                pass
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec

def _gold_list(x):
    return x if isinstance(x, list) else []

def _parse_output_list(output_text):
    """
    解析 output_text 为 list[dict]（若旧字符串格式则尽力解析）。
    返回 (items, old_format_failed)
    """
    if isinstance(output_text, list):
        return [it for it in output_text if isinstance(it, dict)], False

    s = (output_text or "").strip()
    if not s:
        return [], True
    tag, body = _peel_interact_header(s)
    if tag == 0:
        return [], False
    if tag == 1:
        s = body
    if not s or s == "[]" or s == NO_INTERACTION_STR:
        return [], False
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [it for it in obj if isinstance(it, dict)], False
        return [], True
    except Exception:
        return [], True

# ===== Bag-F1（树内合并；不去重） =====
def bag_f1(pred_ctr: Counter, gold_ctr: Counter, eps=1e-8):
    names = set(pred_ctr) | set(gold_ctr)
    hit = sum(min(pred_ctr.get(n,0), gold_ctr.get(n,0)) for n in names)
    pred_sum = sum(pred_ctr.values())
    gold_sum = sum(gold_ctr.values())
    if pred_sum == 0 or gold_sum == 0:
        return 0.0
    prec = hit / (pred_sum + eps)
    rec  = hit / (gold_sum + eps)
    return 0.0 if (prec+rec)==0 else 2.0 * prec * rec / (prec + rec + eps)

# ===== 度分布/权重分布度量 =====
from math import log2

def _normalize(dist): 
    s = float(sum(dist.values())) or 1.0
    return {k:v/s for k,v in dist.items()}

def _jsd(p, q):
    # p,q: dict[node]->prob
    keys = set(p)|set(q)
    P = [p.get(k,0.0) for k in keys]
    Q = [q.get(k,0.0) for k in keys]
    M = [(pi+qi)/2 for pi,qi in zip(P,Q)]
    def _kl(A,B):
        s=0.0
        for a,b in zip(A,B):
            if a>0 and b>0: s += a*log2(a/b)
        return s
    return 0.5*_kl(P,M)+0.5*_kl(Q,M)

def _spearman(x, y):
    # x,y: dict[node]->value
    keys = list(set(x)|set(y))
    xv = [x.get(k,0.0) for k in keys]
    yv = [y.get(k,0.0) for k in keys]
    def _ranks(arr):
        order = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0]*len(arr)
        i=0
        while i<len(arr):
            j=i
            while j+1<len(arr) and arr[order[j+1]]==arr[order[i]]: j+=1
            r = (i+j)/2+1
            for k in range(i,j+1): ranks[order[k]]=r
            i=j+1
        return ranks
    rx, ry = _ranks(xv), _ranks(yv)
    n=len(rx); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    den=(sum((r-mx)**2 for r in rx)*sum((r-my)**2 for r in ry))**0.5
    return 0.0 if den==0 else num/den

def _gini(xs):
    xs = sorted([x for x in xs if x>=0])
    if not xs: return 0.0
    n=len(xs); s=sum(xs)
    if s==0: return 0.0
    cum=0.0; num=0.0
    for i,x in enumerate(xs,1):
        cum += x
        num += cum
    return 2*num/(n*s) - (n+1)/n

def _ks_dist(a, b):
    if not a and not b: return 0.0
    sa = sorted(a); sb = sorted(b)
    ia=ib=0; na=len(sa); nb=len(sb)
    D=0.0
    points = sorted(set(sa+sb))
    for t in points:
        while ia<na and sa[ia]<=t: ia+=1
        while ib<nb and sb[ib]<=t: ib+=1
        Fa = ia/na if na>0 else 0.0
        Fb = ib/nb if nb>0 else 0.0
        D = max(D, abs(Fa-Fb))
    return D

# ===== 主程序 =====
def main():
    ap = argparse.ArgumentParser(
        description="从评估明细日志统计：面板 + step_rewardF1 + 树内合并(bag)F1 + 图级 + Micro节点/边级F1。"
    )
    ap.add_argument("--input", required=True, help="评估明细 JSON/JSONL（--jsonl_detail 输出）")
    ap.add_argument("--depths", type=int, nargs="*", default=None, help="仅统计这些 report_depth（如：--depths 1 2）")
    ap.add_argument("--undirected_graph", action="store_true", help="图评估按无向边（默认有向）")
    ap.add_argument("--micro_mode", choices=["set","bag"], default="set",
                    help="逐行（micro）口径：set=去重，bag=不去重（按次数）")
    args = ap.parse_args()

    allow_depths = set(args.depths) if args.depths else None
    undirected = bool(args.undirected_graph)
    micro_mode = args.micro_mode

    # (I) 面板计数
    total_cands_by_depth   = defaultdict(int)
    gold_type0_by_depth    = defaultdict(int)
    gold_typepos_by_depth  = defaultdict(int)   # gold 的 (1|2) 合并正例（仅在候选集合内对齐）
    pred_t0_by_depth       = defaultdict(int)
    pred_t1_by_depth       = defaultdict(int)
    pred_t2_by_depth       = defaultdict(int)

    # (II) step_reward F1
    reward_vals_by_depth     = defaultdict(list)  # gold 非空样本，用 step_reward
    reward_missing_by_depth  = defaultdict(int)

    # (III) 树内合并（bag，不去重）
    tree_depth_gold_bag = defaultdict(lambda: defaultdict(Counter))
    tree_depth_pred_bag = defaultdict(lambda: defaultdict(Counter))

    # (IV) 图级（树级合并）
    tree_edges_gold = defaultdict(Counter)  # (child,parent)->count
    tree_edges_pred = defaultdict(Counter)

    # (V) Node-Micro（逐行）
    tp_by_depth = defaultdict(int)
    fp_by_depth = defaultdict(int)
    fn_by_depth = defaultdict(int)

    # (VI) Edge-Micro（逐行）
    edge_tp_by_depth = defaultdict(int)
    edge_fp_by_depth = defaultdict(int)
    edge_fn_by_depth = defaultdict(int)

    total_rows = 0
    skipped_bad_depth = 0

    for rec in iter_records(args.input):
        total_rows += 1

        # depth
        rd = rec.get("report_depth")
        try:
            rd = int(rd)
        except Exception:
            skipped_bad_depth += 1
            continue
        if allow_depths and rd not in allow_depths:
            continue

        gid = rec.get("group_id") or rec.get("record_id") or "NA"
        parent = (rec.get("cond_key") or "").strip()

        # gold（严格父键下的子用户名列表）
        gold_list = _gold_list(rec.get("gold"))
        gold_step = [g.strip() for g in gold_list if isinstance(g,str) and g.strip()]
        gold_ctr_step = Counter(gold_step)  # bag 计数

        # output_text -> items
        items, _ = _parse_output_list(rec.get("output_text"))
        num_cands = len(items)
        total_cands_by_depth[rd] += num_cands

        # 预测类型计数 + 节点级正例列表/计数
        t0 = t1 = t2 = 0
        pred_step_names = []
        pred_ctr_step = Counter()
        for it in items:
            try:
                t = int(it.get("type", 0))
            except Exception:
                t = 0
            if t == 0: t0 += 1
            elif t == 1: t1 += 1
            elif t == 2: t2 += 1

            name = (it.get("user_name") or "").strip()
            if name:
                # 面板
                if t in (1,2):
                    pred_step_names.append(name)
                    pred_ctr_step[name] += 1

        pred_t0_by_depth[rd] += t0
        pred_t1_by_depth[rd] += t1
        pred_t2_by_depth[rd] += t2

        # gold 在候选集合内对齐：统计 gold 正/负条目数（面板）
        if num_cands > 0:
            gold_set = set(gold_step)
            gold_hits = sum(1 for n in [(it.get("user_name") or "").strip() for it in items if (it.get("user_name") or "").strip()] if n in gold_set)
            gold_pos = gold_hits
            gold_neg = max(0, num_cands - gold_hits)
        else:
            gold_pos = 0
            gold_neg = 0
        gold_typepos_by_depth[rd] += gold_pos
        gold_type0_by_depth[rd]   += gold_neg

        # (II) step_reward 仅 gold 非空计入
        if len(gold_step) > 0:
            sr = rec.get("step_reward", None)
            if sr is None:
                reward_missing_by_depth[rd] += 1
            else:
                try:
                    reward_vals_by_depth[rd].append(float(sr))
                except Exception:
                    reward_missing_by_depth[rd] += 1

        # (III) 树内合并（bag，不去重）
        tree_depth_gold_bag[gid][rd].update(gold_ctr_step)
        tree_depth_pred_bag[gid][rd].update(pred_ctr_step)

        # (IV) 图级：累加边 (child,parent) 计数
        if parent:
            # gold 边
            for child, cnt in gold_ctr_step.items():
                if undirected:
                    a,b = sorted([child, parent])
                    tree_edges_gold[gid][(a,b)] += cnt
                else:
                    tree_edges_gold[gid][(child,parent)] += cnt
            # pred 边（仅正例）
            for child, cnt in pred_ctr_step.items():
                if undirected:
                    a,b = sorted([child, parent])
                    tree_edges_pred[gid][(a,b)] += cnt
                else:
                    tree_edges_pred[gid][(child,parent)] += cnt

        # (V) Node-Micro：逐条 TP/FP/FN（节点级）
        if micro_mode == "set":
            G = set(gold_step)
            P = set(pred_step_names)
            TP = len(P & G)
            FP = len(P - G)
            FN = len(G - P)
        else:  # bag
            cg = Counter(gold_step)
            cp = Counter(pred_step_names)
            names = set(cg) | set(cp)
            TP = sum(min(cp[n], cg[n]) for n in names)
            FP = sum(max(cp[n]-cg.get(n,0), 0) for n in names)
            FN = sum(max(cg[n]-cp.get(n,0), 0) for n in names)
        tp_by_depth[rd] += TP
        fp_by_depth[rd] += FP
        fn_by_depth[rd] += FN

        # (VI) Edge-Micro：逐条 TP/FP/FN（边级，严格 child→parent）
        if parent:
            if micro_mode == "set":
                Ge = set((g,parent) for g in gold_step)
                Pe = set((n,parent) for n in pred_step_names)
                eTP = len(Pe & Ge)
                eFP = len(Pe - Ge)
                eFN = len(Ge - Pe)
            else:
                Ge = Counter((g,parent) for g in gold_step)
                Pe = Counter((n,parent) for n in pred_step_names)
                keys = set(Ge)|set(Pe)
                eTP = sum(min(Pe[k], Ge[k]) for k in keys)
                eFP = sum(max(Pe[k]-Ge.get(k,0), 0) for k in keys)
                eFN = sum(max(Ge[k]-Pe.get(k,0), 0) for k in keys)
            edge_tp_by_depth[rd] += eTP
            edge_fp_by_depth[rd] += eFP
            edge_fn_by_depth[rd] += eFN

    # ===== 输出 (I) 面板 =====
    depths_union = (
        set(total_cands_by_depth.keys())   |
        set(gold_type0_by_depth.keys())    |
        set(gold_typepos_by_depth.keys())  |
        set(pred_t0_by_depth.keys())       |
        set(pred_t1_by_depth.keys())       |
        set(pred_t2_by_depth.keys())       |
        set(reward_vals_by_depth.keys())   |
        set(reward_missing_by_depth.keys())|
        set(tp_by_depth.keys())            |
        set(edge_tp_by_depth.keys())
    )
    depths_sorted = sorted(depths_union)

    print("=== (I) 按 depth 聚合的候选/真实/预测条目计数 ===")
    print("{:<12} {:>10}   {:>12} {:>16}   {:>12} {:>12} {:>12}".format(
        "report_depth", "cands",
        "gold_type0", "gold_type(1|2)",
        "pred_type0", "pred_type1", "pred_type2"
    ))

    sum_cands = sum_g0 = sum_gp = sum_p0 = sum_p1 = sum_p2 = 0
    for d in depths_sorted:
        cands = total_cands_by_depth.get(d, 0)
        g0 = gold_type0_by_depth.get(d, 0)
        gp = gold_typepos_by_depth.get(d, 0)
        p0 = pred_t0_by_depth.get(d, 0)
        p1 = pred_t1_by_depth.get(d, 0)
        p2 = pred_t2_by_depth.get(d, 0)
        print("{:<12} {:>10}   {:>12} {:>16}   {:>12} {:>12} {:>12}".format(
            d, cands, g0, gp, p0, p1, p2
        ))
        sum_cands += cands; sum_g0 += g0; sum_gp += gp
        sum_p0 += p0; sum_p1 += p1; sum_p2 += p2
    if depths_sorted:
        print("{:<12} {:>10}   {:>12} {:>16}   {:>12} {:>12} {:>12}".format(
            "OVERALL", sum_cands, sum_g0, sum_gp, sum_p0, sum_p1, sum_p2
        ))
    else:
        print("(无数据)")

    # ===== 输出 (II-A) step_reward F1 =====
    print("\n=== (II-A) F1（仅 gold 非空；以日志 step_reward 为准）按 depth ===")
    print("{:<12} {:>12} {:>16}".format("report_depth", "n_with_F1", "mean_F1"))
    overall_nF1 = 0
    overall_sumF1 = 0.0
    for d in depths_sorted:
        vals = reward_vals_by_depth.get(d, [])
        nF1 = len(vals)
        meanF1 = (sum(vals)/nF1) if nF1 > 0 else 0.0
        overall_nF1 += nF1
        overall_sumF1 += sum(vals)
        print("{:<12} {:>12} {:>16.6f}".format(d, nF1, meanF1))
    if overall_nF1 > 0:
        print("{:<12} {:>12} {:>16.6f}".format("OVERALL", overall_nF1, overall_sumF1 / overall_nF1))
    else:
        print("{:<12} {:>12} {:>16}".format("OVERALL", 0, "—"))

    # (II-B) 缺失
    print("\n=== (II-B) 缺失 step_reward 计数（gold 非空样本） ===")
    print("{:<12} {:>14}".format("report_depth", "reward_missing"))
    total_miss = 0
    for d in depths_sorted:
        miss = reward_missing_by_depth.get(d, 0)
        total_miss += miss
        print("{:<12} {:>14}".format(d, miss))
    print("{:<12} {:>14}".format("OVERALL", total_miss))

    # ===== 输出 (III) 树内合并（bag，不去重）F1 =====
    print("\n=== (III) 树内合并（bag，不去重）F1（按 depth 平均） ===")
    print("{:<12} {:>12} {:>16}".format("report_depth", "n_trees", "mean_bagF1"))
    bag_f1_by_depth = defaultdict(list)
    for gid, depth2gold in tree_depth_gold_bag.items():
        depth2pred = tree_depth_pred_bag.get(gid, {})
        for d in set(depth2gold.keys()) | set(depth2pred.keys()):
            gold_ctr = depth2gold.get(d, Counter())
            pred_ctr = depth2pred.get(d, Counter())
            f1 = bag_f1(pred_ctr, gold_ctr)
            bag_f1_by_depth[d].append(f1)
    overall_list = []
    for d in sorted(bag_f1_by_depth.keys()):
        vals = bag_f1_by_depth[d]
        meanF1 = (sum(vals)/len(vals)) if vals else 0.0
        print("{:<12} {:>12} {:>16.6f}".format(d, len(vals), meanF1))
        overall_list.extend(vals)
    if overall_list:
        print("{:<12} {:>12} {:>16.6f}".format("OVERALL", len(overall_list), sum(overall_list)/len(overall_list)))
    else:
        print("{:<12} {:>12} {:>16}".format("OVERALL", 0, "—"))

    # ===== 输出 (IV) 图级评估 =====
    print("\n=== (IV) 图级评估（树级合并后） ===")
    print("{:<20} {:>10} {:>8} {:>8} {:>8}   {:>7} {:>9} {:>10} {:>9}   {:>12} {:>8} {:>9} {:>10}".format(
        "tree(group_id)", "rel_err", "ePrec", "eRec", "eF1",
        "JSD_out", "Sp_out", "JSD_in", "Sp_in",
        "mean|Δ|", "KS_w", "Gini_g", "Gini_p"
    ))
    rel_list = []; ePrec_list=[]; eRec_list=[]; eF1_list=[]
    jsd_out_list=[]; sp_out_list=[]; jsd_in_list=[]; sp_in_list=[]
    mean_rel_diff_list=[]; ks_list=[]; gini_g_list=[]; gini_p_list=[]
    for gid in sorted(set(tree_edges_gold.keys()) | set(tree_edges_pred.keys())):
        cnt_gold = tree_edges_gold.get(gid, Counter())
        cnt_pred = tree_edges_pred.get(gid, Counter())
        edges_all = sorted(set(cnt_gold.keys()) | set(cnt_pred.keys()))
        if not edges_all:
            continue
        a = [cnt_gold.get(e, 0.0) for e in edges_all]
        b = [cnt_pred.get(e, 0.0) for e in edges_all]
        # rel_error
        num = math.sqrt(sum((ai - bi) * (ai - bi) for ai, bi in zip(a, b)))
        den = math.sqrt(sum(ai * ai for ai in a)) + 1e-8
        rel = num / den
        rel_list.append(rel)

        # 边存在性 PRF（集合）
        edge_set_gold = set(e for e,c in cnt_gold.items() if c>0)
        edge_set_pred = set(e for e,c in cnt_pred.items() if c>0)
        tp = len(edge_set_gold & edge_set_pred)
        fp = len(edge_set_pred - edge_set_gold)
        fn = len(edge_set_gold - edge_set_pred)
        ePrec = tp / (tp+fp+1e-8)
        eRec  = tp / (tp+fn+1e-8)
        eF1   = 0.0 if ePrec+eRec==0 else 2*ePrec*eRec/(ePrec+eRec+1e-8)
        ePrec_list.append(ePrec); eRec_list.append(eRec); eF1_list.append(eF1)

        # 度分布
        out_deg_gold = defaultdict(float); in_deg_gold = defaultdict(float)
        out_deg_pred = defaultdict(float); in_deg_pred = defaultdict(float)
        for (u,v),w in cnt_gold.items(): out_deg_gold[u]+=w; in_deg_gold[v]+=w
        for (u,v),w in cnt_pred.items(): out_deg_pred[u]+=w; in_deg_pred[v]+=w

        Pg = _normalize(out_deg_gold); Qg = _normalize(out_deg_pred)
        JSD_out = _jsd(Pg, Qg); jsd_out_list.append(JSD_out)
        Sp_out  = _spearman(out_deg_gold, out_deg_pred); sp_out_list.append(Sp_out)

        Pg = _normalize(in_deg_gold); Qg = _normalize(in_deg_pred)
        JSD_in  = _jsd(Pg, Qg); jsd_in_list.append(JSD_in)
        Sp_in   = _spearman(in_deg_gold, in_deg_pred); sp_in_list.append(Sp_in)

        # 权重分布
        w_gold = [cnt_gold[e] for e in edges_all]
        w_pred = [cnt_pred[e] for e in edges_all]
        mean_gold = (sum(w_gold)/len(w_gold)) if w_gold else 0.0
        mean_pred = (sum(w_pred)/len(w_pred)) if w_pred else 0.0
        mean_rel_diff = 0.0 if mean_gold==0 else abs(mean_pred-mean_gold)/mean_gold
        ks_w = _ks_dist(w_gold, w_pred)
        gini_gold = _gini(w_gold)
        gini_pred = _gini(w_pred)
        mean_rel_diff_list.append(mean_rel_diff); ks_list.append(ks_w)
        gini_g_list.append(gini_gold); gini_p_list.append(gini_pred)

        print("{:<20} {:>10.3f} {:>8.3f} {:>8.3f} {:>8.3f}   {:>7.3f} {:>9.3f} {:>10.3f} {:>9.3f}   {:>12.3f} {:>8.3f} {:>9.3f} {:>10.3f}".format(
            str(gid)[:20], rel, ePrec, eRec, eF1,
            JSD_out, Sp_out, JSD_in, Sp_in,
            mean_rel_diff, ks_w, gini_gold, gini_pred
        ))
    if rel_list:
        print("{:<20} {:>10.3f} {:>8.3f} {:>8.3f} {:>8.3f}   {:>7.3f} {:>9.3f} {:>10.3f} {:>9.3f}   {:>12.3f} {:>8.3f} {:>9.3f} {:>10.3f}".format(
            "MEAN", sum(rel_list)/len(rel_list),
            sum(ePrec_list)/len(ePrec_list) if ePrec_list else 0.0,
            sum(eRec_list)/len(eRec_list)   if eRec_list  else 0.0,
            sum(eF1_list)/len(eF1_list)     if eF1_list   else 0.0,
            sum(jsd_out_list)/len(jsd_out_list) if jsd_out_list else 0.0,
            sum(sp_out_list)/len(sp_out_list)   if sp_out_list else 0.0,
            sum(jsd_in_list)/len(jsd_in_list)   if jsd_in_list  else 0.0,
            sum(sp_in_list)/len(sp_in_list)     if sp_in_list   else 0.0,
            sum(mean_rel_diff_list)/len(mean_rel_diff_list) if mean_rel_diff_list else 0.0,
            sum(ks_list)/len(ks_list) if ks_list else 0.0,
            sum(gini_g_list)/len(gini_g_list) if gini_g_list else 0.0,
            sum(gini_p_list)/len(gini_p_list) if gini_p_list else 0.0
        ))
    else:
        print("(无可统计树)")

    # ===== 输出 (V) Node-Micro（逐行汇总） =====
    print("\n=== (V) Micro-averaged 节点级 F1 by depth (per-line TP/FP/FN summed; mode={}) ===".format(micro_mode))
    print("{:<12} {:>10} {:>10} {:>10}   {:>8} {:>8} {:>8}".format(
        "report_depth","ΣTP","ΣFP","ΣFN","Prec","Rec","F1"))
    overall_TP = overall_FP = overall_FN = 0
    for d in sorted(set(tp_by_depth)|set(fp_by_depth)|set(fn_by_depth)):
        TP = tp_by_depth.get(d,0); FP = fp_by_depth.get(d,0); FN = fn_by_depth.get(d,0)
        P  = TP / (TP + FP + 1e-8)
        R  = TP / (TP + FN + 1e-8)
        F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R+1e-8)
        print("{:<12} {:>10} {:>10} {:>10}   {:>8.4f} {:>8.4f} {:>8.4f}".format(d,TP,FP,FN,P,R,F1))
        overall_TP += TP; overall_FP += FP; overall_FN += FN
    P  = overall_TP / (overall_TP + overall_FP + 1e-8)
    R  = overall_TP / (overall_TP + overall_FN + 1e-8)
    F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R+1e-8)
    print("{:<12} {:>10} {:>10} {:>10}   {:>8.4f} {:>8.4f} {:>8.4f}".format(
        "OVERALL", overall_TP, overall_FP, overall_FN, P, R, F1))

    # ===== 输出 (VI) Edge-Micro（逐行汇总，严格 child→parent） =====
    print("\n=== (VI) Micro-averaged 边级 F1 by depth (per-line TP/FP/FN summed; mode={}) ===".format(micro_mode))
    print("{:<12} {:>10} {:>10} {:>10}   {:>8} {:>8} {:>8}".format(
        "report_depth","ΣTP","ΣFP","ΣFN","Prec","Rec","F1"))
    oTP = oFP = oFN = 0
    for d in sorted(set(edge_tp_by_depth)|set(edge_fp_by_depth)|set(edge_fn_by_depth)):
        TP = edge_tp_by_depth.get(d,0); FP = edge_fp_by_depth.get(d,0); FN = edge_fn_by_depth.get(d,0)
        P  = TP / (TP + FP + 1e-8)
        R  = TP / (TP + FN + 1e-8)
        F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R+1e-8)
        print("{:<12} {:>10} {:>10} {:>10}   {:>8.4f} {:>8.4f} {:>8.4f}".format(d,TP,FP,FN,P,R,F1))
        oTP += TP; oFP += FP; oFN += FN
    P  = oTP / (oTP + oFP + 1e-8)
    R  = oTP / (oTP + oFN + 1e-8)
    F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R+1e-8)
    print("{:<12} {:>10} {:>10} {:>10}   {:>8.4f} {:>8.4f} {:>8.4f}".format(
        "OVERALL", oTP, oFP, oFN, P, R, F1))

    print("\n[stats] total_rows={}, skipped_bad_depth={}".format(total_rows, skipped_bad_depth))

if __name__ == "__main__":
    main()
