"""Fetch Niner Engage events from Supabase (env: SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)."""
import json
from pathlib import Path
from storage import Supabase

CAMPUS = 'https://campusevents.charlotte.edu/calendar/1.xml'

def local(out='data/d3_events.json'):
    """No Supabase: fetch both feeds (Niner Engage RSS + Campus Events calendar), normalize with pipeline.py, tag with the keyword baseline.
    The production allowlist in pipeline.SOURCES is untouched; campusevents is added only inside this process (source id kept as in the frozen split manifest)."""
    import pipeline
    from pipeline import collect, prepare
    from tagging import baseline
    pipeline.SOURCES['campusevents'] = CAMPUS
    events = []
    for src in ('ninerengage', 'campusevents'):
        _, rows = prepare(collect(src), src)
        events += [dict(e, tags=baseline(e)) for _, e in rows if e]
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
