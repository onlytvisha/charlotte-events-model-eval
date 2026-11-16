"""Chronological check: same profiles, as_of one week later. Ratings of already-judged (profile, event) pairs carry over."""
import csv, json
from rankers import *
from make_judgments import pooled
from evaluate_recs import load_ratings

SHIFT = '2026-11-22'
def t2_profiles(profiles): return [dict(p, profile_id=p['profile_id'] + '_t2', as_of=SHIFT) for p in profiles]

if __name__ == '__main__':
    events = json.load(open('data/d3_events.json')); profiles = json.load(open('data/profiles.json'))
    base = load_ratings('data/judgments.csv')
    rows = pooled(events, t2_profiles(profiles), test_ids(), k=10)
    carry, todo = [], []
    for r in rows:
        prior = base[r['profile_id'][:-3]].get(r['event_id'])
        if prior is None: todo.append(r)
        else: carry.append(dict(r, rating=prior))
    cols = ['profile_id', 'event_id', 'title', 'description', 'rating']
    for name, data in (('data/judgments_t2_carry.csv', carry), ('data/judgments_t2_sheet.csv', todo)):
        with open(name, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(data)
    print(len(carry), 'carried over;', len(todo), 'to rate')
