"""Rankers + shared pool construction for D3. Text features: title + description only."""
import json
from datetime import datetime, timedelta, timezone
from recommend import recommend

TEXT_FIELDS = ('title', 'description')  # allowlist; timestamps/content_hash never used as features
TOPICS = ['technology', 'arts', 'sports', 'wellness', 'career', 'community', 'academic', 'culture']

def parse_dt(s):
    d = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

def as_of_dt(profile):
    return parse_dt(profile['as_of'] + ('T00:00:00+00:00' if len(profile['as_of']) == 10 else ''))

def test_ids(path='data/active_split_manifest.json'):
    return {r['event_id'] for r in json.load(open(path)) if r['split'] == 'test'}

def build_pool(events, as_of, excluded, days=14):
    end, seen, pool = as_of + timedelta(days=days), set(), []
    for e in sorted(events, key=lambda e: (e['start_at'], e['id'])):
        if e['id'] in excluded or e.get('status') != 'active': continue
        if not (as_of <= parse_dt(e['start_at']) < end): continue
        k = e.get('duplicate_key') or e['id']
        if k in seen: continue
        seen.add(k); pool.append(e)
    return pool

import re
def group_key(e):
    """Recurring-event group: normalized title (digits/parentheticals removed)."""
    t = re.sub(r'\([^)]*\)', ' ', (e.get('title') or '').lower())
    return ' '.join(re.sub(r'[^a-z ]+', ' ', t).split())

def event_text(e): return ' '.join(str(e.get(f) or '') for f in TEXT_FIELDS)

def profile_text(p, by_id):
    saved = ' '.join(event_text(by_id[i]) for i in p.get('saved_ids', []) if i in by_id)
    return ' '.join([' '.join(p.get('interests', [])), p.get('free_text') or '', saved])

def rank_baseline(pool, p, ctx): return sorted(pool, key=lambda e: (e['start_at'], e['id']))

def rank_current(pool, p, ctx):
    ids = {e['id'] for e in pool}
    out = recommend(ctx['events'], p.get('interests', ()), p.get('saved_ids', ()), limit=len(ctx['events']), at=as_of_dt(p))
    order = [e for e in out if e['id'] in ids]
    got = {e['id'] for e in order}
    return order + [e for e in pool if e['id'] not in got]  # ineligible-by-recommend go last

def _by_score(pool, scores):
    return [e for _, _, e in sorted(((-s, e['start_at'], e) for s, e in zip(scores, pool)), key=lambda t: (t[0], t[1], t[2]['id']))]

def make_tfidf(texts, ngram=(1, 2), min_df=1):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=ngram, min_df=min_df, stop_words='english').fit(texts)

def rank_tfidf(pool, p, ctx):
    vec = ctx['tfidf']
    from sklearn.metrics.pairwise import cosine_similarity
    s = cosine_similarity(vec.transform([profile_text(p, ctx['by_id'])]), vec.transform([event_text(e) for e in pool]))[0]
    return _by_score(pool, s)

_model = None
def embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return lambda texts: _model.encode(texts, normalize_embeddings=True)

def rank_embed(pool, p, ctx):
    enc = ctx.get('encode') or embedder()
    q = enc([profile_text(p, ctx['by_id'])])[0]
    return _by_score(pool, enc([event_text(e) for e in pool]) @ q)

RANKERS = {'baseline': rank_baseline, 'current': rank_current, 'tfidf': rank_tfidf, 'embed': rank_embed}
