"""Build blinded judgment sheet: pooled top-12 of every ranker, shuffled, no ranker info."""
import csv, json, random, sys
from rankers import *

def pooled(events, profiles, excluded, k=12, seed=0, rankers=RANKERS):
    by_id = {e['id']: e for e in events}; rng = random.Random(seed); rows = []
    for p in profiles:
        pool = build_pool(events, as_of_dt(p), excluded)
        # unsupervised vectorizer fit on pool text only (no labels exist yet)
        ctx = {'events': events, 'by_id': by_id, 'tfidf': make_tfidf([event_text(e) for e in pool])}
        picked = {}
        for name, fn in rankers.items():
            try: top = fn(pool, p, ctx)[:k]
            except ImportError: print(f'skip {name}: missing package', file=sys.stderr); continue
            for e in top: picked[e['id']] = e
        items = list(picked.values()); rng.shuffle(items)
        rows += [dict(profile_id=p['profile_id'], event_id=e['id'], title=e.get('title'),
                      description=(e.get('description') or '')[:300], rating='') for e in items]
    return rows

if __name__ == '__main__':
    events = json.load(open('data/d3_events.json')); profiles = json.load(open('data/profiles.json'))
    rows = pooled(events, profiles, test_ids())
    with open('data/judgments_sheet.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['profile_id', 'event_id', 'title', 'description', 'rating']); w.writeheader(); w.writerows(rows)
    print(len(rows), 'rows ->data/judgments_sheet.csv')
