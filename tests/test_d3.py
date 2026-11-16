from datetime import datetime, timedelta, timezone
import evaluate_recs as ev
import rankers as rk
import make_judgments as mj

T0 = datetime(2026, 11, 15, tzinfo=timezone.utc)
SENT = 'ZZZ_SENTINEL'


def mk(i, days):
    return dict(id=f'e{i}', status='active', start_at=(T0 + timedelta(days=days)).isoformat(),
                title=f'event {i} robot music', description='desc', organizers=[f'o{i % 3}'],
                source_categories=[f'c{i % 2}'], duplicate_key=f'k{i}', source_id='ninerengage', tags=None,
                first_seen_at=SENT, last_seen_at=SENT, updated_at=SENT, content_hash=SENT)


EVENTS = [mk(i, 1 + i % 10) for i in range(30)] + [mk(100 + i, -1 - i) for i in range(6)]  # past events feed the vectorizer corpus
TEST = {'e3', 'e7'}
PROFILES = [dict(profile_id=f'p{i}', interests=['technology'], free_text='robot', saved_ids=[], as_of='2026-11-15')
            for i in range(4)]


def ratings():
    return {p['profile_id']: {e['id']: (i + n) % 3 for n, e in enumerate(EVENTS) if e['id'] not in TEST}
            for i, p in enumerate(PROFILES)}


def test_no_test_split_events_used():
    pool = rk.build_pool(EVENTS, T0, TEST)
    assert not TEST & {e['id'] for e in pool}
    rows = mj.pooled(EVENTS, PROFILES, TEST, rankers={k: v for k, v in rk.RANKERS.items() if k != 'embed'})
    assert rows and not TEST & {r['event_id'] for r in rows}
    assert set(rows[0]) == {'profile_id', 'event_id', 'title', 'description', 'rating'}
    _, ranks, _, _ = ev.lopo(EVENTS, PROFILES, ratings(), TEST, rankers=('baseline', 'current', 'tfidf'))
    for r in ranks.values():
        for lst in r.values():
            assert not TEST & {e['id'] for e in lst}


def test_tuning_never_sees_heldout_profile():
    seen = []

    def spy(train, events, pool_by_p, ratings_, **kw):
        seen.append({p['profile_id'] for p in train})
        return ev.TFIDF_GRID[0]

    ev.lopo(EVENTS, PROFILES, ratings(), TEST, rankers=('tfidf',), tune_fn=spy)
    assert len(seen) == len(PROFILES)
    for held, train in zip([p['profile_id'] for p in PROFILES], seen):
        assert held not in train


def test_excluded_columns_not_features():
    assert SENT not in rk.event_text(EVENTS[0])
    assert set(rk.TEXT_FIELDS) == {'title', 'description'}


def test_vectorizer_corpus_is_past_only():
    texts = ev.past_text(EVENTS, PROFILES, TEST)
    assert len(texts) == 6  # only events that started before as_of; no pool or test text
    assert ev.past_text(EVENTS, PROFILES, {'e100'}).__len__() == 5


def test_model_features_exclude_forbidden_and_labels():
    import models
    assert not set(models.LEXICAL + models.EMBED) & models.FORBIDDEN
    win = models.Window(EVENTS, PROFILES, TEST)
    X = win.features(PROFILES[0])
    assert X.shape == (len(win.pool), len(models.LEXICAL))
    assert not TEST & {e['id'] for e in win.pool}


def test_model_tuning_excludes_heldout_profile():
    import evaluate_models as em
    seen = []

    def spy(name, train_pids, data):
        seen.append(set(train_pids))
        return {'C': 1.0}

    em.lopo_models(EVENTS, PROFILES, ratings(), TEST, None, tune_fn=spy, variants=[('logreg', False)])
    assert len(seen) == len(PROFILES)
    for held, train in zip([p['profile_id'] for p in PROFILES], seen):
        assert held not in train
