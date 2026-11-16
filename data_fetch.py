"""Fetch Niner Engage events from Supabase (env: SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)."""
import json
from pathlib import Path
from storage import Supabase

def local(out='data/d3_events.json'):
    """No Supabase: fetch the allowlisted RSS, normalize with pipeline.py, tag with the keyword baseline."""
    from pipeline import collect, prepare
    from tagging import baseline
    _, rows = prepare(collect('ninerengage'), 'ninerengage')
    events = []
    for _, e in rows:
        if e: e['tags'] = baseline(e); events.append(e)
    Path(out).write_text(json.dumps(events, indent=1, default=str))
    print(f'{len(events)} events -> {out} (local)')

def main(out='data/d3_events.json'):
    db = Supabase()
    events = db.rows('events', source_id='eq.ninerengage')
    tags = {t['event_id']: t for t in db.rows('event_tags')}
    for e in events:
        t = tags.get(e['id'])
        e['tags'] = t if t and t['content_hash'] == e['content_hash'] else None
    Path(out).write_text(json.dumps(events, indent=1, default=str))
    print(f'{len(events)} events -> {out}')

if __name__ == '__main__':
    import sys
    local() if '--local' in sys.argv else main()
