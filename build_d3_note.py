"""Render reports/d3_note.html (baseline vs logistic regression vs decision tree) and reports/d3_note_data.json
(the same prose, reused by the Word note). All numbers are read from the results files / data, nothing is typed in."""
import csv, html, json, os
from collections import Counter
from rankers import as_of_dt, build_pool, group_key, test_ids

M = json.load(open('reports/d3_model_comparison.json')); S = M['summary']; SEL = M['selection']; EA = M['error_analysis']; G = M['grouped']
R = json.load(open('reports/d3_results.json'))
FULL = json.load(open('reports/d3_model_comparison_full.json')) if os.path.exists('reports/d3_model_comparison_full.json') else None
EV = json.load(open('data/d3_events.json')); TEST = test_ids()
LAB = {'baseline': 'baseline (start time)', 'logreg': 'logistic regression', 'tree': 'decision tree'}
CH = {**{k: v['ndcg@10']['mean'] for k, v in M['chronological_window2'].items()},
      **{k: v['ndcg@10']['mean'] for k, v in M['chronological_window2_rankers'].items() if k == 'baseline'}}
f = lambda x, d=3: f'{x:.{d}f}'
e = html.escape
n = lambda k, m='ndcg@10': S[k][m]['mean']

# ---- facts computed from the data ----
src = Counter(x['source_id'] for x in EV)
pool1 = build_pool(EV, as_of_dt({'as_of': '2026-11-15'}), TEST); pool2 = build_pool(EV, as_of_dt({'as_of': '2026-11-22'}), TEST)
pool1_all = build_pool(EV, as_of_dt({'as_of': '2026-11-15'}), set())
c1 = Counter(x['source_id'] for x in pool1); c2 = Counter(x['source_id'] for x in pool2)
n_rate1 = sum(1 for _ in csv.DictReader(open('data/judgments.csv', encoding='utf-8')))
n_rate2 = sum(1 for _ in csv.DictReader(open('data/judgments_t2.csv', encoding='utf-8')))
no_org = sum(1 for x in EV if x['source_id'] == 'campusevents' and not x['organizers']) / src['campusevents']
assert SEL['selected'] == 'logreg' and SEL['best'] == 'logreg', 'selection changed: rewrite the Selected-model prose'

setup1 = (f"Task: rank upcoming campus events for a student who gives interests, optional free text and saved events. Data: two public feeds, the Niner Engage RSS feed ({src['ninerengage']} events) and the Campus Events calendar feed ({src['campusevents']} events), "
          f"normalized by the repository's pipeline into {len(EV)} records. Candidate pool per profile: active events starting within 14 days of 2026-11-15 ({len(pool1)} events: {c1['campusevents']} Campus Events, {c1['ninerengage']} Niner Engage; {len({group_key(x) for x in pool1})} distinct recurring groups), "
          f"minus the {len(pool1_all) - len(pool1)} events in the frozen two-source test split, with cross-source duplicates collapsed. Judgments: the top 12 of several rankers pooled, shuffled and rated 0/1/2 ({n_rate1} ratings). A row is a (profile, event) pair; label = rating ≥1, unjudged = 0.")
setup2 = ("Features, all known at prediction time: topic overlap with the profile, saved-topic similarity, TF-IDF cosine of profile text vs. event title and description (vocabulary fit on past events only), days ahead, description length, number of topic tags, recurring-group size, free food/admission tags, start hour and weekday. "
          "Ingestion timestamps, content hashes and ratings are never features; keyword tags are features, never labels.")
setup3 = ("Validation: leave-one-profile-out, with hyper-parameters tuned by profile-grouped CV inside the training profiles only (logistic regression C in {0.01, 0.1, 1}; tree max depth in {2, 3, 5}, min leaf in {5, 20}). "
          "Metrics: nDCG@10 (primary), recall@10, precision@10, category and organizer coverage in the top 10; 95% bootstrap CI over profiles (seed 0).")
table_note = (f"Baseline = events sorted by start time. Week-later = trained on the 2026-11-15 window and evaluated on 2026-11-22 ({len(pool2)} events: {c2['campusevents']} Campus Events, {c2['ninerengage']} Niner Engage; {n_rate2} ratings; same profiles, so not independent). "
              f"Recall is relative to the judged pool. Organizer columns are unreliable: {no_org:.0%} of Campus Events items have no organizer field, so they all count as one organizer.")
sel1 = (f"Logistic regression has the higher mean nDCG@10 ({f(n('logreg'))} vs {f(n('tree'))}) and is also the simpler model, so the rule \"choose the simplest model within one standard error of the best\" selects it without a tie-break. "
        f"The gap ({f(n('logreg') - n('tree'))}) is smaller than one standard error ({f(SEL['se'])}) and the intervals overlap heavily, so the two are not clearly separated. Logistic regression also leads on the grouped-event check ({f(G['logreg']['ndcg@10']['mean'])} vs {f(G['tree']['ndcg@10']['mean'])}) "
        f"and on the week-later window ({f(CH['logreg'])} vs {f(CH['tree'])}). Both are far above the baseline ({f(n('baseline'))}).")
RS = R['summary']; rs = lambda k: RS[k]['ndcg@10']['mean']
other = (f"Other approaches were also tried" + (" (reports/d3_model_comparison_full.csv)" if FULL else "") + f". The existing recommend.py heuristic ({f(rs('current'))}) and a zero-shot sentence-embedding similarity ranker ({f(rs('embed'))}) both scored higher than either model here, "
         f"and a TF-IDF similarity ranker fit on past events only scored {f(rs('tfidf'))}")
if FULL:
    FS = FULL['summary']; fm = lambda k: FS[k]['ndcg@10']['mean']
    heavy = {k: fm(k) for k in ('forest', 'boosting', 'neural_net')}
    better = [LAB.get(k, k.replace('_', ' ')) for k, v in heavy.items() if v > n('logreg')]
    other += (f". Random forest ({f(heavy['forest'])}), gradient boosting ({f(heavy['boosting'])}) and a neural net ({f(heavy['neural_net'])}) "
              + (f"scored higher than logistic regression only for: {', '.join(better)}." if better else "did not beat logistic regression."))
else:
    other += '.'
other += " So these are the simplest learned models compared here, not the best-scoring approach found."

def weakest(k):
    t = EA['per_topic'][k]; w = min(t, key=lambda x: t[x]['mean_ndcg@10']); b = max(t, key=lambda x: t[x]['mean_ndcg@10'])
    return w, t[w]['mean_ndcg@10'], b, t[b]['mean_ndcg@10']
wl, wlv, bl, blv = weakest('logreg'); wt, wtv, bt, btv = weakest('tree')
def titles(kind, k, m=4): return list(dict.fromkeys(x['title'].split(' at ')[0] for x in EA[kind][k]))[:m]
if wl == wt: weak_txt = f"{wl} is the weakest interest topic for both models (logistic regression {f(wlv, 2)}, decision tree {f(wtv, 2)})"
else: weak_txt = f"Weakest interest topic: {wl} for logistic regression ({f(wlv, 2)}) and {wt} for the decision tree ({f(wtv, 2)})"
if bl == bt: best_txt = f"the strongest is {bl} ({f(blv, 2)} and {f(btv, 2)})"
else: best_txt = f"strongest: {bl} ({f(blv, 2)}) and {bt} ({f(btv, 2)})"
patterns = (f"{weak_txt}; {best_txt}. False positives include generic notices and unrelated events (logistic regression: {', '.join(titles('false_positives', 'logreg', 3))}; decision tree: {', '.join(titles('false_positives', 'tree', 3))}). "
            "A false positive can also be an event that was never judged and so counts as 0. Per-topic numbers rest on 2–3 profiles and are anecdotal.")
limits = [
    "Placeholder profiles and model-assigned ratings; no human agreement estimate. The earlier Niner-Engage-only ratings were reused for events that stayed in the pool.",
    f"Only 12 profiles; wide CIs, and the logistic regression vs. tree gap is within noise.",
    f"Campus Events items almost never name an organizer ({no_org:.0%} empty), so organizer coverage and top-organizer share are unreliable. Campus Events titles carry an \"at <venue>\" suffix that is stripped only for the recurring-group key.",
    "Unjudged events count as 0 and the judged set was pooled from other rankers, so recall and precision cover judged items only.",
    "Two 14-day windows from one snapshot; no interaction data, so no attendance or satisfaction claim.",
    f"The frozen split was fixed on an earlier snapshot and most of its test IDs are Campus Events ({sum(1 for x in EV if x['id'] in TEST and x['source_id'] == 'campusevents')} of {len(TEST)} present); {len(pool1_all) - len(pool1)} events in the first window are excluded and never rated. Events newer than the split are unassigned.",
]
links = [
    ('Model-evaluation repository (this work)', 'https://github.com/onlytvisha/charlotte-events-model-eval'),
    ('Original project repository (charlotte-events-data)', 'https://github.com/n-a-testing/charlotte-events-data'),
    ('Supabase organization dashboard (login required)', 'https://supabase.com/dashboard/org/mtxbyxjikclwqnvmetqa'),
    ('Niner Engage events feed', 'https://ninerengage.charlotte.edu/events.rss'),
    ('Campus Events calendar feed', 'https://campusevents.charlotte.edu/calendar/1.xml'),
    ('scikit-learn LogisticRegression', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'),
    ('scikit-learn DecisionTreeClassifier', 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html'),
    ('scikit-learn TfidfVectorizer', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html'),
    ('all-MiniLM-L6-v2 embedding model', 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'),
    ('Sentence Transformers documentation', 'https://www.sbert.net/'),
    ('Discounted cumulative gain / nDCG', 'https://en.wikipedia.org/wiki/Discounted_cumulative_gain'),
]
code_files = ['evaluate_models.py', 'models.py', 'evaluate_recs.py', 'rankers.py', 'tests/test_d3.py']
json.dump(dict(setup1=setup1, setup2=setup2, setup3=setup3, table_note=table_note, sel1=sel1, other=other, patterns=patterns, limits=limits), open('reports/d3_note_data.json', 'w'), indent=1)

# ---- HTML ----
rows = ''.join(
    f"<tr{' class=best' if k == SEL['selected'] else ''}><td>{LAB[k]}</td><td>{f(n(k))} <small>({f(S[k]['ndcg@10']['lo'])}–{f(S[k]['ndcg@10']['hi'])})</small></td>"
    f"<td>{f(n(k, 'recall@10'))}</td><td>{f(n(k, 'precision@10'))}</td><td>{f(n(k, 'category_coverage'), 1)}</td>"
    f"<td>{f(n(k, 'organizer_coverage'), 1)}</td><td>{f(n(k, 'dominant_org_share'), 2)}</td><td>{f(CH[k])}</td></tr>" for k in LAB)
topics = sorted(EA['per_topic']['logreg'])
trows = ''.join(f"<tr><td>{t} <small>(n={EA['per_topic']['logreg'][t]['n_profiles']})</small></td><td>{f(EA['per_topic']['logreg'][t]['mean_ndcg@10'], 2)}</td><td>{f(EA['per_topic']['tree'][t]['mean_ndcg@10'], 2)}</td></tr>" for t in topics)
def lst(kind, k, m=4): return ''.join(f"<li>{e(x['title'])} <small>({x['profile_id']}, rank {x['rank']}, rating {x['rating']})</small></li>" for x in EA[kind][k][:m])
base = 'https://github.com/onlytvisha/charlotte-events-model-eval/blob/main/'
code_links = ', '.join(f'<a href="{base}{c}">{c}</a>' for c in code_files)
def link_li(t, u):
    extra = f': {code_links}' if t.startswith('Model-evaluation') else ''
    return f'<li><a href="{u}">{e(t)}</a>{extra}</li>'
group = lambda a, b: ''.join(link_li(t, u) for t, u in links[a:b])
page = f"""<!doctype html><html><head><meta charset=utf-8><title>D3 Modeling Note</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 16px;color:#1a1a1a}}
h1{{font-size:1.6rem}}h2{{font-size:1.15rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0}}td,th{{border:1px solid #ddd;padding:4px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
tr.best{{background:#e8f4ea;font-weight:600}}small{{color:#666;font-weight:400}}.warn{{background:#fff4e0;border-left:4px solid #e0a030;padding:8px 12px}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}td,th{{border-color:#444}}tr.best{{background:#1d3a24}}.warn{{background:#3a2f18}}small{{color:#aaa}}}}
</style></head><body>
<h1>D3 Modeling Note: baseline vs. logistic regression vs. decision tree</h1>
<p class=warn><b>Dry run.</b> The 12 profiles are invented placeholders and all relevance ratings were assigned by Claude, not human reviewers. Treat the numbers as a test of the pipeline, not as the D3 result.</p>
<h2>1. Setup</h2>
<p>{e(setup1)}</p><p>{e(setup2)}</p><p>{e(setup3)}</p>
<h2>2. Model comparison</h2>
<table><tr><th>Model</th><th>nDCG@10 (95% CI)</th><th>recall@10</th><th>precision@10</th><th>categories</th><th>organizers</th><th>top-org share</th><th>week-later nDCG@10</th></tr>{rows}</table>
<p>{e(table_note)}</p>
<h2>3. Selected model</h2>
<p>{e(sel1)}</p><p>{e(other)}</p>
<h2>4. Error analysis</h2>
<p>nDCG@10 by interest topic (n = profiles listing the topic; very small):</p>
<table><tr><th>Topic</th><th>logistic regression</th><th>decision tree</th></tr>{trows}</table>
<p><b>Logistic regression</b> false positives (in top 10, rated 0):</p><ul>{lst('false_positives', 'logreg')}</ul>
<p><b>Logistic regression</b> false negatives (relevant, ranked below 10):</p><ul>{lst('false_negatives', 'logreg')}</ul>
<p><b>Decision tree</b> false positives:</p><ul>{lst('false_positives', 'tree')}</ul>
<p><b>Decision tree</b> false negatives:</p><ul>{lst('false_negatives', 'tree')}</ul>
<p>{e(patterns)} Full top-20 lists are in <code>reports/d3_model_comparison.json</code>.</p>
<h2>5. Limitations</h2>
<ul>{''.join('<li>' + e(x) + '</li>' for x in limits)}</ul>
<h2>6. Links</h2>
<p><b>Repositories and code (the runnable evaluation code lives in the repository files)</b></p>
<ul>{group(0, 3)}</ul>
<p><b>Data sources (both used)</b></p>
<ul>{group(3, 5)}</ul>
<p><b>Methods and libraries</b></p>
<ul>{group(5, len(links))}</ul>
</body></html>"""
open('reports/d3_note.html', 'w', encoding='utf-8').write(page)
