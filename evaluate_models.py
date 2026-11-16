"""Model-family comparison on the complexity ladder: leave-one-profile-out + chronological check.
Writes reports/d3_model_comparison.json/.csv. Run after evaluate_recs.py (reuses its per-profile results)."""
import csv, json, math, sys
import numpy as np
import evaluate_recs as ev
from models import *
from rankers import test_ids, as_of_dt
from make_time_judgments import t2_profiles

RUNG = {'baseline': 0, 'current': 1, 'tfidf': 1.5, 'logreg': 2, 'tree': 3, 'forest': 4, 'boosting': 5, 'neural_net': 6, 'embed': 7}
ALL_VARIANTS = [('logreg', False), ('tree', False), ('forest', False), ('boosting', False), ('neural_net', False),
            ('logreg+embed', True), ('boosting+embed', True)]

VARIANTS = [('logreg', False), ('tree', False)]  # simplest two model families; use --full for the whole ladder

def ranked(win, sc): return [e for _, _, e in sorted(((-s, e['start_at'], e) for s, e in zip(sc, win.pool)), key=lambda t: (t[0], t[1], t[2]['id']))]

def build_data(win, profiles, ratings, use_embed):
    return {p['profile_id']: (win.features(p, use_embed), labels(win.pool, ratings[p['profile_id']]), win.pool, ratings[p['profile_id']]) for p in profiles}

def lopo_models(events, profiles, ratings, excluded, encode, tune_fn=tune, variants=None):
    variants = variants or VARIANTS
    win = Window(events, profiles, excluded, encode); out, chosen, ranks = {}, {}, {}
    for vname, use_embed in variants:
        fam = vname.split('+')[0]
        data = build_data(win, profiles, ratings, use_embed); pids = [p['profile_id'] for p in profiles]
        out[vname] = {}; ranks[vname] = {}
        for held in pids:
            sc, kw = fit_predict(fam, [q for q in pids if q != held], data, held, tune_fn)
            ranks[vname][held] = ranked(win, sc); out[vname][held] = ev.metrics(ranks[vname][held], ratings[held]); chosen.setdefault(vname, {})[held] = kw
        print(vname, round(float(np.nanmean([m['ndcg@10'] for m in out[vname].values()])), 3), file=sys.stderr)
    return out, chosen, ranks, win

def chrono_models(events, profiles, r1, r2, excluded, encode, variants=None):
    variants = variants or VARIANTS
    p2 = t2_profiles(profiles); w1 = Window(events, profiles, excluded, encode); w2 = Window(events, p2, excluded, encode); out = {}
    for vname, use_embed in variants:
        fam = vname.split('+')[0]; d1 = build_data(w1, profiles, r1, use_embed); d2 = build_data(w2, p2, r2, use_embed)
        pids = [p['profile_id'] for p in profiles]; kw = tune(fam, pids, d1)  # window-1 labels only
        Xtr = np.vstack([d1[p][0] for p in pids]); ytr = np.concatenate([d1[p][1] for p in pids]); m = ZOO[fam][1](**kw).fit(Xtr, ytr)
        out[vname] = {q['profile_id']: ev.metrics(ranked(w2, scores(m, d2[q['profile_id']][0])), r2[q['profile_id']]) for q in p2}
    return out

def one_se_choice(per_model):
    means = {k: float(np.nanmean([m['ndcg@10'] for m in v.values()])) for k, v in per_model.items()}
    best = max(means, key=means.get); vals = np.array([m['ndcg@10'] for m in per_model[best].values()])
    se = float(np.nanstd(vals, ddof=1) / math.sqrt(len(vals))); thr = means[best] - se
    ok = [k for k in means if means[k] >= thr]
    pick = min(ok, key=lambda k: RUNG.get(k.split('+')[0], 8) + (0.5 if '+' in k else 0))
    return dict(best=best, best_mean=means[best], se=se, threshold=thr, adequate=ok, selected=pick, means=means)

def main():
    events = json.load(open('data/d3_events.json')); profiles = json.load(open('data/profiles.json')); ex = test_ids()
    r1 = ev.load_ratings('data/judgments.csv'); r2 = ev.load_ratings('data/judgments_t2.csv')
    full = '--full' in sys.argv; V = ALL_VARIANTS if full else VARIANTS; tag = '_full' if full else ''
    from rankers import embedder; encode = embedder() if any(u for _, u in V) else None
    base = json.load(open('reports/d3_results.json'))['error_analysis']['per_profile']  # baseline/current/tfidf/embed from evaluate_recs.py
    learned, chosen, ranks, win = lopo_models(events, profiles, r1, ex, encode, variants=V)
    allm = {'baseline': base['baseline'], **learned} if not full else {**base, **learned}
    chrono_learned = chrono_models(events, profiles, r1, r2, ex, encode, variants=V)
    base2 = json.load(open('reports/d3_results.json'))['chronological_window2']
    summary = {k: {m: ev.bootstrap([v[p][m] for p in v]) for m in ev.METRICS} for k, v in allm.items()}
    sel = one_se_choice(allm)
    rep = dict(note='leave-one-profile-out; label rating>=1 (unjudged=0); hyper-parameters tuned by profile-grouped CV inside training profiles only; ratings by Claude (dry run)',
               rung=RUNG, summary=summary, selection_rule='simplest model whose mean nDCG@10 is within 1 SE of the best', selection=sel,
               chosen_hyperparameters={k: list({json.dumps(x, sort_keys=True) for x in v.values()}) for k, v in chosen.items()},
               chronological_window2={k: {m: ev.bootstrap([v[p][m] for p in v]) for m in ('ndcg@10', 'recall@10', 'precision@10')} for k, v in chrono_learned.items()},
               error_analysis=ev.error_analysis(ranks, r1, profiles, learned),
               grouped={k: {m: ev.bootstrap([ev.grouped(ranks[k][p['profile_id']], r1[p['profile_id']], win.pool)[m] for p in profiles]) for m in ev.METRICS} for k in ranks},
               chronological_window2_rankers={k: {m: v[m] for m in ('ndcg@10', 'recall@10', 'precision@10')} for k, v in base2['summary'].items()})
    json.dump(rep, open(f'reports/d3_model_comparison{tag}.json', 'w'), indent=1)
    with open(f'reports/d3_model_comparison{tag}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'rung', 'ndcg@10', 'ci_lo', 'ci_hi', 'recall@10', 'precision@10', 'n_profiles'])
        for k, s in summary.items():
            n = s['ndcg@10']; w.writerow([k, RUNG.get(k.split('+')[0], ''), n['mean'], n['lo'], n['hi'], s['recall@10']['mean'], s['precision@10']['mean'], n['n']])
    print(json.dumps({k: round(v, 3) for k, v in sel['means'].items()}, indent=0)); print('selected:', sel['selected'], 'best:', sel['best'], 'se:', round(sel['se'], 3))

if __name__ == '__main__': main()
