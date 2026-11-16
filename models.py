"""D3 model families on the complexity ladder, trained on human-style relevance ratings.

Row = (profile, candidate event). Label = rating >= 1 (unjudged pool events = 0, same convention as the metrics).
Features use only information available at prediction time. first_seen_at / last_seen_at / updated_at /
content_hash and the ratings are never features. Keyword event tags are used as FEATURES only (never labels).
"""
import math
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from rankers import build_pool, as_of_dt, parse_dt, event_text, profile_text, group_key, make_tfidf

LEXICAL = ['topic_matches', 'any_match', 'saved_sim', 'tfidf_cos', 'days_ahead', 'log_desc_len', 'n_topics',
           'group_size', 'free_food', 'free_admission', 'start_hour', 'weekday']
EMBED = ['embed_cos']
FORBIDDEN = {'first_seen_at', 'last_seen_at', 'updated_at', 'content_hash', 'rating'}

def tags_of(e): return set(((e.get('tags') or {}).get('topics')) or [])

class Window:
    """Per-as_of context: pool, past-only TF-IDF, optional embeddings. No labels inside."""
    def __init__(self, events, profiles, excluded, encode=None):
        self.events, self.by_id = events, {e['id']: e for e in events}
        self.as_of = as_of_dt(profiles[0])
        cut = self.as_of
        past = [event_text(e) for e in events if e['id'] not in excluded and parse_dt(e['start_at']) < cut]
        self.vec = make_tfidf(past, ngram=(1, 1), min_df=1)
        self.pool = build_pool(events, self.as_of, excluded)
        self.gsize = {}
        for e in self.pool: self.gsize[group_key(e)] = self.gsize.get(group_key(e), 0) + 1
        self.encode = encode
        if encode:
            self.emb = dict(zip([e['id'] for e in self.pool], encode([event_text(e) for e in self.pool])))

    def features(self, p, use_embed=False):
        pool, interests = self.pool, set(p.get('interests', []))
        saved = set().union(*(tags_of(self.by_id[i]) for i in p.get('saved_ids', []) if i in self.by_id)) if p.get('saved_ids') else set()
        sims = cosine_similarity(self.vec.transform([profile_text(p, self.by_id)]), self.vec.transform([event_text(e) for e in pool]))[0]
        rows = []
        for e, s in zip(pool, sims):
            t = tags_of(e); st = parse_dt(e['start_at']); tg = e.get('tags') or {}
            rows.append([len(t & interests), float(bool(t & interests)), len(t & saved) / max(1, len(t | saved)), s,
                         (st - self.as_of).total_seconds() / 86400, math.log1p(len(e.get('description') or '')), len(t),
                         self.gsize[group_key(e)], float(tg.get('free_food') == 'yes'), float(tg.get('free_admission') == 'yes'),
                         st.hour, st.weekday()])
        X = np.array(rows, float)
        if use_embed:
            q = self.encode([profile_text(p, self.by_id)])[0]
            X = np.hstack([X, np.array([self.emb[e['id']] @ q for e in pool])[:, None]])
        return X

def labels(pool, rel): return np.array([int(rel.get(e['id'], 0) >= 1) for e in pool])

# ---- model zoo, ordered by complexity: name -> (rung, builder, grid) ----
def _lr(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight='balanced'))
ZOO = {
    'logreg':   (2, lambda C: _lr(C), [dict(C=c) for c in (0.01, 0.1, 1.0)]),
    'tree':     (3, lambda d, m: DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, class_weight='balanced', random_state=0),
                 [dict(d=d, m=m) for d in (2, 3, 5) for m in (5, 20)]),
    'forest':   (4, lambda d, m: RandomForestClassifier(n_estimators=150, max_depth=d, min_samples_leaf=m, class_weight='balanced', random_state=0, n_jobs=-1),
                 [dict(d=d, m=m) for d in (3, 6) for m in (5, 20)]),
    'boosting': (5, lambda n, d, lr: GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=0),
                 [dict(n=n, d=d, lr=lr) for n in (50, 150) for d in (2, 3) for lr in (0.05, 0.1)]),
    'neural_net': (6, lambda a: make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=a, max_iter=800, random_state=0)),
                 [dict(a=a) for a in (1e-3, 1e-1)]),
}

def scores(model, X): return model.predict_proba(X)[:, 1]

def ndcg_from_scores(sc, pool, rel, k=10):
    order = sorted(range(len(pool)), key=lambda i: (-sc[i], pool[i]['start_at'], pool[i]['id']))
    gains = [rel.get(pool[i]['id'], 0) for i in order[:k]]
    dcg = sum((2 ** g - 1) / math.log2(n + 2) for n, g in enumerate(gains))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(n + 2) for n, g in enumerate(ideal))
    return dcg / idcg if idcg else float('nan')

def tune(name, train_pids, data, n_splits=5):
    """Pick hyper-parameters by profile-grouped CV INSIDE the training profiles only.
    data[pid] = (X, y, pool, rel). Returns best kwargs."""
    _, build, grid = ZOO[name]
    pids = list(train_pids); gkf = GroupKFold(n_splits=min(n_splits, len(pids)))
    best, best_s = grid[0], -1
    for kw in grid:
        folds = []
        for tr, te in gkf.split(pids, groups=pids):
            Xtr = np.vstack([data[pids[i]][0] for i in tr]); ytr = np.concatenate([data[pids[i]][1] for i in tr])
            if ytr.min() == ytr.max(): continue
            m = build(**kw).fit(Xtr, ytr)
            folds += [ndcg_from_scores(scores(m, data[pids[i]][0]), data[pids[i]][2], data[pids[i]][3]) for i in te]
        s = float(np.nanmean(folds)) if folds else -1
        if s > best_s: best, best_s = kw, s
    return best

def fit_predict(name, train_pids, data, test_pid, tune_fn=tune):
    kw = tune_fn(name, train_pids, data)
    Xtr = np.vstack([data[p][0] for p in train_pids]); ytr = np.concatenate([data[p][1] for p in train_pids])
    m = ZOO[name][1](**kw).fit(Xtr, ytr)
    return scores(m, data[test_pid][0]), kw
