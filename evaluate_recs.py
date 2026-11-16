"""D3 recommender evaluation: leave-one-profile-out, bootstrap CIs, error analysis."""
import csv, json, math, sys
from collections import Counter, defaultdict
import numpy as np
from rankers import *

K = 10
TFIDF_GRID = [dict(ngram=n, min_df=m) for n in [(1, 1), (1, 2)] for m in [1, 2]]
METRICS = ['ndcg@10', 'recall@10', 'precision@10', 'category_coverage', 'organizer_coverage', 'dominant_org_share']

def load_ratings(path):
    r = defaultdict(dict)
    for row in csv.DictReader(open(path, encoding='utf-8')):
        if row['rating'].strip() == '': raise ValueError('incomplete judgments: blank rating')
        v = int(row['rating']); assert v in (0, 1, 2)
        r[row['profile_id']][row['event_id']] = v
    return r

def ndcg(ranked_ids, rel, k=K):
    dcg = sum((2 ** rel.get(i, 0) - 1) / math.log2(n + 2) for n, i in enumerate(ranked_ids[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(n + 2) for n, g in enumerate(ideal))
    return dcg / idcg if idcg else float('nan')

def metrics(ranked, rel):
    top = ranked[:K]; ids = [e['id'] for e in top]; nrel = sum(1 for v in rel.values() if v >= 1)
    hit = sum(1 for i in ids if rel.get(i, 0) >= 1)
    orgs = Counter('|'.join(e.get('organizers') or []) or e.get('source_id', '') for e in top)
    cats = {c for e in top for c in (e.get('source_categories') or [])}
    return {'ndcg@10': ndcg(ids, rel), 'recall@10': hit / nrel if nrel else float('nan'),
            'precision@10': hit / max(1, len(top)), 'category_coverage': len(cats),
            'organizer_coverage': len(orgs), 'dominant_org_share': max(orgs.values()) / len(top) if top else float('nan')}

def make_ctx(events, train_pools_text=None, cfg=None):
    ctx = {'events': events, 'by_id': {e['id']: e for e in events}}
    if train_pools_text is not None: ctx['tfidf'] = make_tfidf(train_pools_text, **cfg)
    return ctx

def past_text(events, profiles, excluded):
    """Vectorizer corpus = events that started before the earliest as_of (known at prediction time); no candidate-pool text, no test split."""
    cut = min(as_of_dt(p) for p in profiles)
    return [event_text(e) for e in events if e['id'] not in excluded and parse_dt(e['start_at']) < cut]

def tune_tfidf(train_profiles, events, pool_by_p, ratings, excluded=frozenset()):
    """Select TF-IDF config by mean nDCG@10 on TRAIN profiles only; vectorizer fit on train pool text only."""
    best, best_s = TFIDF_GRID[0], -1
    for cfg in TFIDF_GRID:
        ctx = make_ctx(events, past_text(events, train_profiles, excluded), cfg)
        s = np.nanmean([ndcg([e['id'] for e in rank_tfidf(pool_by_p[p['profile_id']], p, ctx)], ratings[p['profile_id']]) for p in train_profiles])
        if s > best_s: best, best_s = cfg, s
    return best

def lopo(events, profiles, ratings, excluded, rankers=('baseline', 'current', 'tfidf', 'embed'), tune_fn=tune_tfidf, encode=None):
    pool_by_p = {p['profile_id']: build_pool(events, as_of_dt(p), excluded) for p in profiles}
    res = {r: {} for r in rankers}; ranks = {r: {} for r in rankers}; chosen = {}
    for held in profiles:
        pid = held['profile_id']; train = [p for p in profiles if p['profile_id'] != pid]; pool = pool_by_p[pid]
        base = make_ctx(events)
        if encode: base['encode'] = encode
        for r in rankers:
            ctx = dict(base)
            if r == 'tfidf':
                cfg = tune_fn(train, events, pool_by_p, ratings, excluded=excluded); chosen[pid] = cfg
                ctx = {**make_ctx(events, past_text(events, train, excluded), cfg), **({'encode': encode} if encode else {})}
            try: ranked = RANKERS[r](pool, held, ctx)
            except ImportError: print(f'skip {r}: missing package', file=sys.stderr); res.pop(r, None); ranks.pop(r, None); rankers = tuple(x for x in rankers if x != r); continue
            ranks[r][pid] = ranked; res[r][pid] = metrics(ranked, ratings[pid])
    return res, ranks, chosen, pool_by_p


def grouped(ranked, rel, pool):
    """Treat recurring events (same normalized title) as one item: one slot per group, group relevance = max rating."""
    rep, gmax = {}, {}
    for e in pool:
        g = group_key(e); rep.setdefault(g, e['id'])
        if e['id'] in rel: gmax[g] = max(gmax.get(g, 0), rel[e['id']])
    seen, out = set(), []
    for e in ranked:
        g = group_key(e)
        if g in seen: continue
        seen.add(g); out.append(dict(e, id=rep[g]))
    return metrics(out, {rep[g]: v for g, v in gmax.items()})

def chrono(events, profiles, r1, r2, excluded, encode=None):
    """Select/fit on window 1 (as_of 2026-11-15), evaluate on window 2 (a week later)."""
    from make_time_judgments import t2_profiles
    p2 = t2_profiles(profiles)
    pools1 = {p['profile_id']: build_pool(events, as_of_dt(p), excluded) for p in profiles}
    pools2 = {p['profile_id']: build_pool(events, as_of_dt(p), excluded) for p in p2}
    cfg = tune_tfidf(profiles, events, pools1, r1, excluded=excluded)  # window-1 labels only
    ctx = {**make_ctx(events, past_text(events, p2, excluded), cfg), **({'encode': encode} if encode else {})}
    res = {}
    for name in ('baseline', 'current', 'tfidf', 'embed'):
        try: res[name] = {p['profile_id']: metrics(RANKERS[name](pools2[p['profile_id']], p, ctx), r2[p['profile_id']]) for p in p2}
        except ImportError: print(f'skip {name}', file=sys.stderr)
    return res, cfg

def bootstrap(vals, n=2000, seed=0):
    v = np.array([x for x in vals if not math.isnan(x)]); rng = np.random.default_rng(seed)
    if len(v) == 0: return dict(mean=None, lo=None, hi=None, n=0)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return dict(mean=float(v.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(v))

def error_analysis(ranks, ratings, profiles, res):
    out = {'per_profile': {r: res[r] for r in res}, 'per_topic': {}, 'false_positives': {}, 'false_negatives': {}}
    for r in res:
        topics = defaultdict(list)
        for p in profiles:
            for t in p.get('interests', []): topics[t].append(res[r][p['profile_id']]['ndcg@10'])
        out['per_topic'][r] = {t: {'mean_ndcg@10': float(np.nanmean(v)), 'n_profiles': len(v)} for t, v in sorted(topics.items())}
        fp, fn = [], []
        for pid, ranked in ranks[r].items():
            for n, e in enumerate(ranked):
                g = ratings[pid].get(e['id'], 0)
                row = dict(profile_id=pid, event_id=e['id'], title=e.get('title'), rank=n + 1, rating=g)
                if n < K and g == 0: fp.append(row)
                if n >= K and g >= 1 and e['id'] in ratings[pid]: fn.append(row)
        out['false_positives'][r] = sorted(fp, key=lambda x: x['rank'])[:20]
        out['false_negatives'][r] = sorted(fn, key=lambda x: (-x['rating'], x['rank']))[:20]
    return out

def main():
    events = json.load(open('data/d3_events.json')); profiles = json.load(open('data/profiles.json'))
    ratings = load_ratings('data/judgments.csv'); res, ranks, chosen, pool_by_p = lopo(events, profiles, ratings, test_ids())
    summary = {r: {m: bootstrap([res[r][p][m] for p in res[r]]) for m in METRICS} for r in res}
    gres = {r: {pid: grouped(ranks[r][pid], ratings[pid], pool_by_p[pid]) for pid in ranks[r]} for r in ranks}
    r2 = load_ratings('data/judgments_t2.csv'); cres, ccfg = chrono(events, profiles, ratings, r2, test_ids())
    rep = {'n_profiles': len(profiles), 'ci': '95% bootstrap over profiles, seed 0, 2000 resamples',
           'note': 'recall/precision relative to judged pool; unjudged = 0; ratings 0/1/2, relevant>=1',
           'tfidf_chosen_configs': {k: dict(v, ngram=list(v['ngram'])) for k, v in chosen.items()},
           'summary': summary,
           'grouped_recurring_events': {r: {m: bootstrap([gres[r][p][m] for p in gres[r]]) for m in METRICS} for r in gres},
           'chronological_window2': {'as_of': '2026-11-22', 'tfidf_config': dict(ccfg, ngram=list(ccfg['ngram'])), 'note': 'selected/fit on window 1 only; same 12 profiles, so not independent of window 1',
                                     'summary': {r: {m: bootstrap([cres[r][p][m] for p in cres[r]]) for m in METRICS} for r in cres}}, 'error_analysis': error_analysis(ranks, ratings, profiles, res)}
    json.dump(rep, open('reports/d3_results.json', 'w'), indent=1)
    with open('reports/d3_results.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['ranker', 'metric', 'mean', 'ci_lo', 'ci_hi', 'n_profiles'])
        for r, ms in summary.items():
            for m, s in ms.items(): w.writerow([r, m, s['mean'], s['lo'], s['hi'], s['n']])
    for tag, block in (('grouped', rep['grouped_recurring_events']), ('time', rep['chronological_window2']['summary'])):
        for r, ms in block.items(): print(tag, r, 'ndcg@10', round(ms['ndcg@10']['mean'], 3), 'ci', round(ms['ndcg@10']['lo'], 3), round(ms['ndcg@10']['hi'], 3))
    for r, ms in summary.items(): print(r, {m: round(ms[m]['mean'], 3) for m in ('ndcg@10', 'recall@10', 'precision@10') if ms[m]['mean'] is not None})

if __name__ == '__main__': main()
