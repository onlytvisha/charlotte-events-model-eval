"""Render reports/d3_note.html (modeling note: baseline vs logistic regression vs decision tree)."""
import html, json
M = json.load(open('reports/d3_model_comparison.json')); S = M['summary']; SEL = M['selection']; EA = M['error_analysis']
LAB = {'baseline': 'baseline (start time)', 'logreg': 'logistic regression', 'tree': 'decision tree'}
CH = {**{k: v['ndcg@10']['mean'] for k, v in M['chronological_window2'].items()},
      **{k: v['ndcg@10']['mean'] for k, v in M['chronological_window2_rankers'].items() if k == 'baseline'}}
f = lambda x, d=3: f'{x:.{d}f}'
e = html.escape
rows = ''.join(
    f"<tr{' class=best' if k == SEL['selected'] else ''}><td>{LAB[k]}</td><td>{f(S[k]['ndcg@10']['mean'])} <small>({f(S[k]['ndcg@10']['lo'])}–{f(S[k]['ndcg@10']['hi'])})</small></td>"
    f"<td>{f(S[k]['recall@10']['mean'])}</td><td>{f(S[k]['precision@10']['mean'])}</td><td>{f(S[k]['category_coverage']['mean'], 1)}</td>"
    f"<td>{f(S[k]['organizer_coverage']['mean'], 1)}</td><td>{f(S[k]['dominant_org_share']['mean'], 2)}</td><td>{f(CH[k])}</td></tr>" for k in LAB)
topics = sorted(EA['per_topic']['logreg'])
trows = ''.join(f"<tr><td>{t} <small>(n={EA['per_topic']['logreg'][t]['n_profiles']})</small></td><td>{f(EA['per_topic']['logreg'][t]['mean_ndcg@10'], 2)}</td><td>{f(EA['per_topic']['tree'][t]['mean_ndcg@10'], 2)}</td></tr>" for t in topics)
def lst(kind, k, n=4): return ''.join(f"<li>{e(x['title'])} <small>({x['profile_id']}, rank {x['rank']}, rating {x['rating']})</small></li>" for x in EA[kind][k][:n])
G = M['grouped']
page = f"""<!doctype html><html><head><meta charset=utf-8><title>D3 Modeling Note</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 16px;color:#1a1a1a}}
h1{{font-size:1.6rem}}h2{{font-size:1.15rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0}}td,th{{border:1px solid #ddd;padding:4px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
tr.best{{background:#e8f4ea;font-weight:600}}pre{{background:#f5f5f5;padding:10px;overflow-x:auto;font-size:12px;line-height:1.35}}small{{color:#666;font-weight:400}}.warn{{background:#fff4e0;border-left:4px solid #e0a030;padding:8px 12px}}
@media(prefers-color-scheme:dark){{pre{{background:#222}}body{{background:#161616;color:#eee}}td,th{{border-color:#444}}tr.best{{background:#1d3a24}}.warn{{background:#3a2f18}}small{{color:#aaa}}}}
</style></head><body>
<h1>D3 Modeling Note: event recommender, baseline vs. logistic regression vs. decision tree</h1>
<p class=warn><b>Dry run.</b> The 12 profiles are invented placeholders and all relevance ratings were assigned by Claude, not human reviewers. Treat the numbers as a test of the pipeline, not as the D3 result.</p>
<h2>1. Setup</h2>
<p>Task: rank upcoming Niner Engage events for a student who gives interests, optional free text and saved events. Candidate pool per profile: active events starting within 14 days of 2026-11-15, minus the 8 frozen test-split events, deduplicated (278 events). Judgments: top 12 of several rankers, pooled, shuffled and rated 0/1/2 (405 ratings). A row is a (profile, event) pair; label = rating ≥1, unjudged = 0.</p>
<p>Features (all known at prediction time): topic overlap with the profile, saved-topic similarity, TF-IDF cosine between profile text and event title+description (vocabulary fit on past events only), days ahead, description length, number of topic tags, recurring-group size, free food/admission tags, start hour and weekday. Ingestion timestamps, content hashes and ratings are never features; keyword tags are features, never labels.</p>
<p>Validation: leave-one-profile-out. Hyper-parameters are tuned by profile-grouped CV inside the training profiles only (logistic regression: C ∈ {{0.01, 0.1, 1}}; tree: max depth ∈ {{2, 3, 5}}, min leaf ∈ {{5, 20}}). Metric: nDCG@10 (primary), recall@10 and precision@10 (rating ≥1 relevant), category and organizer coverage in the top 10; 95% bootstrap CI over profiles (seed 0).</p>
<h2>2. Model comparison</h2>
<table><tr><th>Model</th><th>nDCG@10 (95% CI)</th><th>recall@10</th><th>precision@10</th><th>categories</th><th>organizers</th><th>top-org share</th><th>week-later nDCG@10</th></tr>{rows}</table>
<p>Baseline = events sorted by start time. "Week-later" = trained on the 2026-11-15 window, evaluated on 2026-11-22 (same profiles, so not independent). Recall is relative to the judged pool.</p>
<h2>3. Selected model</h2>
<p>The decision tree has the higher mean nDCG@10 ({f(S['tree']['ndcg@10']['mean'])} vs {f(S['logreg']['ndcg@10']['mean'])}), but the two intervals overlap almost completely and the gap ({f(S['tree']['ndcg@10']['mean'] - S['logreg']['ndcg@10']['mean'])}) is about one standard error ({f(SEL['se'])}). The rule "choose the simplest model within one standard error of the best" therefore selects <b>logistic regression</b>. It is easier to explain, has the lowest top-organizer share ({f(S['logreg']['dominant_org_share']['mean'], 2)} vs {f(S['tree']['dominant_org_share']['mean'], 2)}), while the tree stays slightly ahead in the grouped-event check too ({f(G['tree']['ndcg@10']['mean'])} vs {f(G['logreg']['ndcg@10']['mean'])}). The tree is slightly stronger on the week-later window ({f(CH['tree'])} vs {f(CH['logreg'])}) and on precision. Both are far above the baseline ({f(S['baseline']['ndcg@10']['mean'])}). Other approaches were also tried (<code>reports/d3_model_comparison_full.csv</code>). On the main validation the random forest (0.487), gradient boosting (0.530) and neural net (0.478) did not beat these two. But the existing <code>recommend.py</code> heuristic (0.568) and a zero-shot sentence-embedding similarity ranker (0.724) both scored higher than either, so these are the simplest learned models compared here, not the best-scoring approach found.</p>
<h2>4. Error analysis</h2>
<p>nDCG@10 by interest topic (n = profiles listing the topic; very small):</p>
<table><tr><th>Topic</th><th>logistic regression</th><th>decision tree</th></tr>{trows}</table>
<p><b>Logistic regression</b> false positives (in top 10, rated 0):</p><ul>{lst('false_positives', 'logreg')}</ul>
<p><b>Logistic regression</b> false negatives (relevant, ranked below 10):</p><ul>{lst('false_negatives', 'logreg')}</ul>
<p><b>Decision tree</b> false positives:</p><ul>{lst('false_positives', 'tree')}</ul>
<p><b>Decision tree</b> false negatives:</p><ul>{lst('false_negatives', 'tree')}</ul>
<p>Patterns: both models lean on the keyword tag features, so movie-review events ("Project Hail Mary") surface for technology/career profiles. Academic is the weakest topic for logistic regression (0.29): tutoring events are numerous and look alike, and relevant ones are pushed below rank 10 by near-duplicates. The tree handles academic better (0.52) but gives more top slots to a single organizer. Per-topic numbers rest on 2–3 profiles each and are anecdotal. Full top-20 lists are in <code>reports/d3_model_comparison.json</code>.</p>
<h2>5. Limitations</h2>
<ul><li>Placeholder profiles and model-assigned ratings; no human agreement estimate.</li>
<li>Only 12 profiles; wide CIs, and the logistic-regression vs. tree gap is within noise.</li>
<li>Unjudged events count as 0 and the judged set was pooled from other rankers, so recall and precision cover judged items only.</li>
<li>Two 14-day windows from one snapshot; no interaction data, so no attendance or satisfaction claim.</li>
<li>Recurring events (e.g. drop-in tutoring) are many near-duplicates; grouped scoring (167 groups) gives the same ordering.</li>
<li>The 8 test-split events are excluded and unrated.</li></ul>
<h2>6. Links</h2>
<p><b>Repositories and code (runnable evaluation code lives in the repository files)</b></p>
<ul><li><a href="https://github.com/onlytvisha/charlotte-events-model-eval">Model-evaluation repository (this work)</a>:
<a href="https://github.com/onlytvisha/charlotte-events-model-eval/blob/main/evaluate_models.py">evaluate_models.py</a>,
<a href="https://github.com/onlytvisha/charlotte-events-model-eval/blob/main/models.py">models.py</a>,
<a href="https://github.com/onlytvisha/charlotte-events-model-eval/blob/main/evaluate_recs.py">evaluate_recs.py</a>,
<a href="https://github.com/onlytvisha/charlotte-events-model-eval/blob/main/rankers.py">rankers.py</a>,
<a href="https://github.com/onlytvisha/charlotte-events-model-eval/blob/main/tests/test_d3.py">tests/test_d3.py</a></li>
<li><a href="https://github.com/n-a-testing/charlotte-events-data">Original project repository (charlotte-events-data)</a></li>
<li><a href="https://supabase.com/dashboard/org/mtxbyxjikclwqnvmetqa">Supabase organization dashboard</a> (login required)</li></ul>
<p><b>Data sources</b></p>
<ul><li><a href="https://ninerengage.charlotte.edu/events.rss">Niner Engage events feed</a> (the only source used)</li>
<li><a href="https://campusevents.charlotte.edu/calendar/1.xml">Campus Events calendar feed</a> (legacy source, not used in D3)</li></ul>
<p><b>Methods and libraries</b></p>
<ul><li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">scikit-learn LogisticRegression</a></li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">scikit-learn DecisionTreeClassifier</a></li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">scikit-learn TfidfVectorizer</a></li>
<li><a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2 embedding model</a> and <a href="https://www.sbert.net/">Sentence Transformers documentation</a> (used in the extra comparison)</li>
<li><a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain">Discounted cumulative gain / nDCG</a></li></ul>
</body></html>"""
open('reports/d3_note.html', 'w', encoding='utf-8').write(page)
