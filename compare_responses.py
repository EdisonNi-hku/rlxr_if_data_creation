#!/usr/bin/env python3
"""Generate a self-contained HTML comparison viewer for two model responses.

Merges rollout files (containing response text) with graded files (containing
scores) by row index, and produces an interactive HTML viewer for qualitative
comparison.
"""

import argparse
import json
import html
import sys


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_data(rollout_rows, graded_rows, model_name):
    """Merge rollout (response text) and graded (scores) by row index."""
    assert len(rollout_rows) == len(graded_rows), (
        f"Row count mismatch for {model_name}: "
        f"{len(rollout_rows)} rollout vs {len(graded_rows)} graded"
    )
    merged = []
    for i, (roll, grad) in enumerate(zip(rollout_rows, graded_rows)):
        response = roll["responses"][0] if roll.get("responses") else ""
        entry = {
            "response": response,
            "avg_score": grad.get("avg_score", 0),
            "pass_rate": grad.get("pass_rate", 0),
            "scores": grad.get("scores", []),
            "word_count": len(response.split()),
            "char_count": len(response),
        }
        # Extract reward scores
        for key in grad:
            if key.startswith("avg_reward_"):
                short_name = key.replace("avg_reward_root_models_", "")
                entry[f"reward_{short_name}"] = grad[key]
        merged.append(entry)
    return merged


def build_comparison_data(m1_rollout, m1_graded, m2_rollout, m2_graded,
                          m1_name, m2_name):
    """Build the full comparison dataset."""
    m1_data = merge_data(m1_rollout, m1_graded, m1_name)
    m2_data = merge_data(m2_rollout, m2_graded, m2_name)

    items = []
    for i in range(len(m1_data)):
        # Instruction from rollout
        instruction = m1_rollout[i].get("instruction", "")
        # Ground truth from graded
        ground_truth_raw = m1_graded[i].get("ground_truth", "[]")
        try:
            ground_truth = json.loads(ground_truth_raw)
        except (json.JSONDecodeError, TypeError):
            ground_truth = []

        # Parse constraint types
        constraint_types = []
        constraints_display = []
        for ct in ground_truth:
            ids = ct.get("instruction_id", [])
            kwargs = ct.get("kwargs", [{}])
            for j, iid in enumerate(ids):
                category = iid.split(":")[0]
                detail = iid.split(":", 1)[1] if ":" in iid else iid
                kw = kwargs[j] if j < len(kwargs) and kwargs[j] else {}
                constraint_types.append(category)
                constraints_display.append({
                    "id": iid,
                    "category": category,
                    "detail": detail,
                    "kwargs": kw,
                })

        # Determine winner
        s1 = m1_data[i]["avg_score"]
        s2 = m2_data[i]["avg_score"]
        if s1 > s2:
            winner = m1_name
        elif s2 > s1:
            winner = m2_name
        else:
            winner = "tie"

        # Average reward across RMs
        m1_rewards = {k: v for k, v in m1_data[i].items() if k.startswith("reward_")}
        m2_rewards = {k: v for k, v in m2_data[i].items() if k.startswith("reward_")}
        m1_avg_reward = sum(m1_rewards.values()) / len(m1_rewards) if m1_rewards else 0
        m2_avg_reward = sum(m2_rewards.values()) / len(m2_rewards) if m2_rewards else 0

        items.append({
            "idx": i,
            "instruction": instruction,
            "constraints": constraints_display,
            "constraint_categories": list(set(constraint_types)),
            "winner": winner,
            "m1": m1_data[i],
            "m2": m2_data[i],
            "score_delta": round(s1 - s2, 4),
            "reward_delta": round(m1_avg_reward - m2_avg_reward, 4),
            "length_delta": m1_data[i]["char_count"] - m2_data[i]["char_count"],
            "m1_avg_reward": round(m1_avg_reward, 4),
            "m2_avg_reward": round(m2_avg_reward, 4),
        })

    return items


QUALITY_DIMS = [
    "incoherent_expression",
    "logical_inconsistency",
    "inappropriate_word_choice",
    "repetitive_expression",
    "language_inconsistency",
]


def _has_quality_issues(quality_dict):
    """Check if a quality analysis dict has any issues flagged."""
    return any(quality_dict.get(dim, 0) == 1 for dim in QUALITY_DIMS)


def compute_summary(items, m1_name, m2_name, has_analysis=False):
    """Compute aggregate summary statistics."""
    n = len(items)
    m1_wins = sum(1 for it in items if it["winner"] == m1_name)
    m2_wins = sum(1 for it in items if it["winner"] == m2_name)
    ties = sum(1 for it in items if it["winner"] == "tie")

    m1_avg_score = sum(it["m1"]["avg_score"] for it in items) / n
    m2_avg_score = sum(it["m2"]["avg_score"] for it in items) / n
    m1_pass_rate = sum(it["m1"]["pass_rate"] for it in items) / n
    m2_pass_rate = sum(it["m2"]["pass_rate"] for it in items) / n
    m1_avg_reward = sum(it["m1_avg_reward"] for it in items) / n
    m2_avg_reward = sum(it["m2_avg_reward"] for it in items) / n

    result = {
        "total": n,
        "m1_wins": m1_wins,
        "m2_wins": m2_wins,
        "ties": ties,
        "m1_win_rate": round(m1_wins / n * 100, 1),
        "m2_win_rate": round(m2_wins / n * 100, 1),
        "tie_rate": round(ties / n * 100, 1),
        "m1_avg_score": round(m1_avg_score, 4),
        "m2_avg_score": round(m2_avg_score, 4),
        "m1_pass_rate": round(m1_pass_rate, 4),
        "m2_pass_rate": round(m2_pass_rate, 4),
        "m1_avg_reward": round(m1_avg_reward, 4),
        "m2_avg_reward": round(m2_avg_reward, 4),
        "has_analysis": has_analysis,
    }

    if has_analysis:
        m1_issues = sum(1 for it in items if _has_quality_issues(it.get("m1_quality", {})))
        m2_issues = sum(1 for it in items if _has_quality_issues(it.get("m2_quality", {})))
        result["m1_issue_count"] = m1_issues
        result["m2_issue_count"] = m2_issues
        result["m1_issue_pct"] = round(m1_issues / n * 100, 1)
        result["m2_issue_pct"] = round(m2_issues / n * 100, 1)

    return result


def generate_html(items, summary, m1_name, m2_name):
    """Generate the self-contained HTML viewer."""

    # Collect unique constraint categories for filter dropdown
    all_categories = set()
    for it in items:
        for c in it["constraint_categories"]:
            all_categories.add(c)
    all_categories = sorted(all_categories)

    data_json = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    categories_json = json.dumps(all_categories)

    m1_esc = html.escape(m1_name)
    m2_esc = html.escape(m2_name)
    m1_js = json.dumps(m1_name)
    m2_js = json.dumps(m2_name)

    # Build JS as a plain string (no f-string) to avoid {{}} escaping issues
    js_code = r"""
var M1_NAME = __M1_NAME__;
var M2_NAME = __M2_NAME__;
var ALL_DATA = __ALL_DATA__;
var SUMMARY = __SUMMARY__;
var ALL_CATEGORIES = __ALL_CATEGORIES__;
var HAS_ANALYSIS = __HAS_ANALYSIS__;
var QUALITY_DIMS = ['incoherent_expression','logical_inconsistency','inappropriate_word_choice','repetitive_expression','language_inconsistency'];
var QUALITY_LABELS = {'incoherent_expression':'Incoherent','logical_inconsistency':'Logic','inappropriate_word_choice':'Word Choice','repetitive_expression':'Repetitive','language_inconsistency':'Language'};

function hasQualityIssues(q) {
  if (!q) return false;
  for (var i = 0; i < QUALITY_DIMS.length; i++) {
    if (q[QUALITY_DIMS[i]] === 1) return true;
  }
  return false;
}

var filtered = ALL_DATA.slice();
var currentIdx = 0;
var searchTimer = null;

function initSummary() {
  var g = document.getElementById('summaryGrid');
  var cards = [
    {l: 'Total Samples', v: SUMMARY.total, s: ''},
    {l: M1_NAME + ' Win Rate', v: SUMMARY.m1_win_rate + '%', s: SUMMARY.m1_wins + ' wins'},
    {l: M2_NAME + ' Win Rate', v: SUMMARY.m2_win_rate + '%', s: SUMMARY.m2_wins + ' wins'},
    {l: 'Ties', v: SUMMARY.tie_rate + '%', s: SUMMARY.ties + ' ties'},
    {l: M1_NAME + ' Avg Score', v: SUMMARY.m1_avg_score.toFixed(3), s: 'Pass rate: ' + (SUMMARY.m1_pass_rate * 100).toFixed(1) + '%'},
    {l: M2_NAME + ' Avg Score', v: SUMMARY.m2_avg_score.toFixed(3), s: 'Pass rate: ' + (SUMMARY.m2_pass_rate * 100).toFixed(1) + '%'},
    {l: M1_NAME + ' Avg Reward', v: SUMMARY.m1_avg_reward.toFixed(2), s: ''},
    {l: M2_NAME + ' Avg Reward', v: SUMMARY.m2_avg_reward.toFixed(2), s: ''},
  ];
  if (HAS_ANALYSIS) {
    cards.push({l: M1_NAME + ' Quality Issues', v: SUMMARY.m1_issue_pct + '%', s: SUMMARY.m1_issue_count + ' of ' + SUMMARY.total + ' responses'});
    cards.push({l: M2_NAME + ' Quality Issues', v: SUMMARY.m2_issue_pct + '%', s: SUMMARY.m2_issue_count + ' of ' + SUMMARY.total + ' responses'});
  }
  var h = '';
  for (var i = 0; i < cards.length; i++) {
    var c = cards[i];
    h += '<div class="summary-card"><div class="label">' + c.l + '</div><div class="value">' + c.v + '</div>';
    if (c.s) h += '<div class="sub">' + c.s + '</div>';
    h += '</div>';
  }
  g.innerHTML = h;
}

function initFilters() {
  var sel = document.getElementById('filterCategory');
  for (var i = 0; i < ALL_CATEGORIES.length; i++) {
    var opt = document.createElement('option');
    opt.value = ALL_CATEGORIES[i];
    opt.textContent = ALL_CATEGORIES[i];
    sel.appendChild(opt);
  }
}

function scoreBadgeClass(score) {
  if (score >= 0.8) return 'badge-green';
  if (score >= 0.3) return 'badge-yellow';
  return 'badge-red';
}

function esc(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function highlight(text, query) {
  if (!query) return esc(text);
  var escaped = esc(text);
  var qEsc = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  var re = new RegExp('(' + qEsc + ')', 'gi');
  return escaped.replace(re, '<mark>$1</mark>');
}

function fmtKwargs(kw) {
  var parts = [];
  var keys = Object.keys(kw);
  for (var i = 0; i < keys.length; i++) {
    var k = keys[i], v = kw[k];
    if (v === null || v === undefined) continue;
    var val = typeof v === 'object' ? JSON.stringify(v) : String(v);
    if (val.length > 40) val = val.slice(0, 37) + '...';
    parts.push(k + '=' + val);
  }
  return parts.join(', ');
}

function qualityBadges(qa) {
  if (!HAS_ANALYSIS || !qa) return '';
  var badges = '';
  for (var i = 0; i < QUALITY_DIMS.length; i++) {
    var dim = QUALITY_DIMS[i];
    if (qa[dim] === 1) {
      badges += '<span class="badge badge-quality" title="' + dim.replace(/_/g, ' ') + '">' + QUALITY_LABELS[dim] + '</span> ';
    }
  }
  return badges;
}

function qualityNotesHtml(qa) {
  if (!HAS_ANALYSIS || !qa || !qa.notes) return '';
  return '<div class="quality-notes">' + esc(qa.notes) + '</div>';
}

function panelHtml(name, d, isWinner, q, qa) {
  var winnerClass = isWinner ? ' winner' : '';
  var rewardKeys = Object.keys(d).filter(function(k) { return k.indexOf('reward_') === 0; });
  var rewardBadges = '';
  for (var i = 0; i < rewardKeys.length; i++) {
    var rk = rewardKeys[i];
    var shortName = rk.replace('reward_', '');
    rewardBadges += '<span class="badge badge-gray">' + shortName + ': ' + d[rk].toFixed(2) + '</span> ';
  }
  return '<div class="panel">' +
    '<div class="panel-header">' +
      '<span class="model-name' + winnerClass + '">' + esc(name) + (isWinner ? ' &#9733;' : '') + '</span>' +
      '<span class="badge ' + scoreBadgeClass(d.avg_score) + '">Score: ' + d.avg_score.toFixed(2) + '</span>' +
      '<span class="badge ' + scoreBadgeClass(d.pass_rate) + '">Pass: ' + (d.pass_rate * 100).toFixed(0) + '%</span>' +
      rewardBadges +
      qualityBadges(qa) +
    '</div>' +
    '<div class="panel-body">' +
      qualityNotesHtml(qa) +
      '<div class="response-text">' + highlight(d.response, q) + '</div>' +
    '</div>' +
    '<div class="panel-footer">' +
      '<span>' + d.word_count + ' words</span>' +
      '<span>' + d.char_count.toLocaleString() + ' chars</span>' +
    '</div>' +
  '</div>';
}

function render() {
  var viewer = document.getElementById('viewer');
  if (filtered.length === 0) {
    viewer.innerHTML = '<div class="empty-state">No items match the current filters.</div>';
    document.getElementById('counter').textContent = '0 / 0';
    return;
  }
  if (currentIdx >= filtered.length) currentIdx = filtered.length - 1;
  if (currentIdx < 0) currentIdx = 0;

  var it = filtered[currentIdx];
  var q = document.getElementById('searchBox').value.trim();
  document.getElementById('counter').textContent = (currentIdx + 1) + ' / ' + filtered.length;

  var constraintHtml = '';
  for (var ci = 0; ci < it.constraints.length; ci++) {
    var c = it.constraints[ci];
    var kwStr = fmtKwargs(c.kwargs);
    constraintHtml += '<span class="constraint-badge">' + esc(c.id);
    if (kwStr) constraintHtml += '<span class="kw">(' + esc(kwStr) + ')</span>';
    constraintHtml += '</span> ';
  }

  var m1Win = it.winner === M1_NAME;
  var m2Win = it.winner === M2_NAME;

  var h = '<div class="prompt-section">' +
    '<h3>Prompt #' + it.idx + ' &mdash; Winner: ' + esc(it.winner) +
    ' &mdash; Score &Delta;: ' + it.score_delta.toFixed(2) +
    ' &mdash; Reward &Delta;: ' + it.reward_delta.toFixed(2) + '</h3>' +
    '<div class="prompt-text">' + highlight(it.instruction, q) + '</div>';
  if (constraintHtml) {
    h += '<div class="constraints-bar">' + constraintHtml + '</div>';
  }
  h += '</div>';
  h += '<div class="panels">' +
    panelHtml(M1_NAME, it.m1, m1Win, q, it.m1_quality) +
    panelHtml(M2_NAME, it.m2, m2Win, q, it.m2_quality) +
  '</div>';

  viewer.innerHTML = h;
  window.scrollTo(0, 0);
}

function applyFilters() {
  var q = document.getElementById('searchBox').value.trim().toLowerCase();
  var cat = document.getElementById('filterCategory').value;
  var winner = document.getElementById('filterWinner').value;
  var passStatus = document.getElementById('filterPass').value;
  var qualityFilter = document.getElementById('filterQuality').value;
  var sort = document.getElementById('sortBy').value;

  filtered = [];
  for (var i = 0; i < ALL_DATA.length; i++) {
    var it = ALL_DATA[i];
    if (q) {
      var haystack = (it.instruction + ' ' + it.m1.response + ' ' + it.m2.response).toLowerCase();
      if (haystack.indexOf(q) === -1) continue;
    }
    if (cat && it.constraint_categories.indexOf(cat) === -1) continue;
    if (winner && it.winner !== winner) continue;
    if (passStatus) {
      var m1Pass = it.m1.pass_rate >= 1.0;
      var m2Pass = it.m2.pass_rate >= 1.0;
      if (passStatus === 'both_pass' && !(m1Pass && m2Pass)) continue;
      if (passStatus === 'both_fail' && !(!m1Pass && !m2Pass)) continue;
      if (passStatus === 'm1_only' && !(m1Pass && !m2Pass)) continue;
      if (passStatus === 'm2_only' && !(!m1Pass && m2Pass)) continue;
    }
    if (qualityFilter && HAS_ANALYSIS) {
      var m1Has = hasQualityIssues(it.m1_quality);
      var m2Has = hasQualityIssues(it.m2_quality);
      if (qualityFilter === 'any_issues' && !m1Has && !m2Has) continue;
      if (qualityFilter === 'no_issues' && (m1Has || m2Has)) continue;
      if (QUALITY_DIMS.indexOf(qualityFilter) !== -1) {
        var m1Dim = it.m1_quality && it.m1_quality[qualityFilter] === 1;
        var m2Dim = it.m2_quality && it.m2_quality[qualityFilter] === 1;
        if (!m1Dim && !m2Dim) continue;
      }
    }
    filtered.push(it);
  }

  if (sort === 'score_delta_desc') filtered.sort(function(a, b) { return b.score_delta - a.score_delta; });
  else if (sort === 'score_delta_asc') filtered.sort(function(a, b) { return a.score_delta - b.score_delta; });
  else if (sort === 'reward_delta_desc') filtered.sort(function(a, b) { return b.reward_delta - a.reward_delta; });
  else if (sort === 'reward_delta_asc') filtered.sort(function(a, b) { return a.reward_delta - b.reward_delta; });
  else if (sort === 'length_delta_desc') filtered.sort(function(a, b) { return b.length_delta - a.length_delta; });
  else if (sort === 'length_delta_asc') filtered.sort(function(a, b) { return a.length_delta - b.length_delta; });
  else filtered.sort(function(a, b) { return a.idx - b.idx; });

  currentIdx = 0;
  render();
}

function goNext() { if (currentIdx < filtered.length - 1) { currentIdx++; render(); } }
function goPrev() { if (currentIdx > 0) { currentIdx--; render(); } }
function goFirst() { currentIdx = 0; render(); }
function goLast() { currentIdx = Math.max(0, filtered.length - 1); render(); }

document.getElementById('btnNext').onclick = goNext;
document.getElementById('btnPrev').onclick = goPrev;
document.getElementById('btnFirst').onclick = goFirst;
document.getElementById('btnLast').onclick = goLast;

document.getElementById('searchBox').addEventListener('input', function() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(applyFilters, 300);
});
document.getElementById('filterCategory').onchange = applyFilters;
document.getElementById('filterWinner').onchange = applyFilters;
document.getElementById('filterPass').onchange = applyFilters;
document.getElementById('filterQuality').onchange = applyFilters;
document.getElementById('sortBy').onchange = applyFilters;

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); goNext(); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); goPrev(); }
  else if (e.key === 'Home') { e.preventDefault(); goFirst(); }
  else if (e.key === 'End') { e.preventDefault(); goLast(); }
});

initSummary();
initFilters();
render();
"""
    # Replace placeholders with actual data
    js_code = js_code.replace("__M1_NAME__", m1_js)
    js_code = js_code.replace("__M2_NAME__", m2_js)
    js_code = js_code.replace("__ALL_DATA__", data_json)
    js_code = js_code.replace("__SUMMARY__", summary_json)
    js_code = js_code.replace("__ALL_CATEGORIES__", categories_json)
    js_code = js_code.replace("__HAS_ANALYSIS__", json.dumps(summary.get("has_analysis", False)))

    # Build HTML (only simple substitutions via f-string)
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{m1_esc} vs {m2_esc} — Response Comparison</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #1a1a2e; color: white; padding: 16px 24px; }}
.header h1 {{ font-size: 20px; margin-bottom: 12px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin-top: 8px; }}
.summary-card {{ background: rgba(255,255,255,0.1); border-radius: 6px; padding: 10px 14px; }}
.summary-card .label {{ font-size: 11px; text-transform: uppercase; opacity: 0.7; letter-spacing: 0.5px; }}
.summary-card .value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
.summary-card .sub {{ font-size: 12px; opacity: 0.8; }}
.controls {{ background: white; border-bottom: 1px solid #ddd; padding: 12px 24px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; position: sticky; top: 0; z-index: 100; }}
.controls input, .controls select {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }}
.controls input[type="text"] {{ width: 250px; }}
.controls select {{ min-width: 140px; }}
.nav-group {{ display: flex; align-items: center; gap: 6px; margin-left: auto; }}
.nav-group button {{ padding: 8px 16px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; font-size: 16px; }}
.nav-group button:hover {{ background: #e0e0e0; }}
.nav-group .counter {{ font-size: 14px; font-weight: 600; min-width: 100px; text-align: center; }}
.item-container {{ max-width: 1400px; margin: 0 auto; padding: 16px 24px; }}
.prompt-section {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px; border: 1px solid #ddd; }}
.prompt-section h3 {{ font-size: 14px; color: #666; margin-bottom: 8px; }}
.prompt-text {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
.constraints-bar {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; }}
.constraint-badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; background: #e8eaf6; color: #3949ab; }}
.constraint-badge .kw {{ font-weight: 400; color: #666; margin-left: 4px; }}
.panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.panel {{ background: white; border-radius: 8px; border: 1px solid #ddd; overflow: hidden; display: flex; flex-direction: column; }}
.panel-header {{ padding: 12px 16px; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.panel-header .model-name {{ font-weight: 700; font-size: 15px; }}
.panel-header .model-name.winner {{ color: #1565c0; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; color: white; }}
.badge-green {{ background: #43a047; }}
.badge-yellow {{ background: #f9a825; color: #333; }}
.badge-red {{ background: #e53935; }}
.badge-blue {{ background: #1565c0; }}
.badge-gray {{ background: #888; }}
.badge-quality {{ background: #e53935; font-size: 10px; cursor: help; }}
.quality-notes {{ background: #fff3f3; border: 1px solid #ffcdd2; border-radius: 6px; padding: 8px 12px; margin-top: 8px; font-size: 12px; color: #c62828; line-height: 1.5; }}
.panel-body {{ padding: 16px; flex: 1; max-height: 600px; overflow-y: auto; }}
.response-text {{ font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono', Consolas, 'Liberation Mono', monospace; }}
.panel-footer {{ padding: 8px 16px; border-top: 1px solid #eee; font-size: 11px; color: #888; display: flex; gap: 12px; }}
mark {{ background: #fff176; padding: 0 1px; border-radius: 2px; }}
.empty-state {{ text-align: center; padding: 60px 20px; color: #888; font-size: 16px; }}
#errorBanner {{ display: none; background: #e53935; color: white; padding: 10px 24px; font-size: 13px; font-family: monospace; }}
@media (max-width: 900px) {{ .panels {{ grid-template-columns: 1fr; }} .controls input[type="text"] {{ width: 180px; }} }}
</style>
</head>
<body>
<div id="errorBanner"></div>
<div class="header">
  <h1>{m1_esc} vs {m2_esc}</h1>
  <div class="summary-grid" id="summaryGrid"></div>
</div>
<div class="controls">
  <input type="text" id="searchBox" placeholder="Search prompts & responses...">
  <select id="filterCategory"><option value="">All constraints</option></select>
  <select id="filterWinner">
    <option value="">All outcomes</option>
    <option value="{m1_esc}">{m1_esc} wins</option>
    <option value="{m2_esc}">{m2_esc} wins</option>
    <option value="tie">Ties</option>
  </select>
  <select id="filterPass">
    <option value="">All pass status</option>
    <option value="both_pass">Both pass</option>
    <option value="both_fail">Both fail</option>
    <option value="m1_only">Only {m1_esc} passes</option>
    <option value="m2_only">Only {m2_esc} passes</option>
  </select>
  <select id="filterQuality">
    <option value="">All quality</option>
    <option value="any_issues">Any issues</option>
    <option value="no_issues">No issues</option>
    <option value="incoherent_expression">Incoherent expression</option>
    <option value="logical_inconsistency">Logical inconsistency</option>
    <option value="inappropriate_word_choice">Inappropriate word choice</option>
    <option value="repetitive_expression">Repetitive expression</option>
    <option value="language_inconsistency">Language inconsistency</option>
  </select>
  <select id="sortBy">
    <option value="index">Sort: Index</option>
    <option value="score_delta_desc">Sort: Score delta ↓</option>
    <option value="score_delta_asc">Sort: Score delta ↑</option>
    <option value="reward_delta_desc">Sort: Reward delta ↓</option>
    <option value="reward_delta_asc">Sort: Reward delta ↑</option>
    <option value="length_delta_desc">Sort: Length delta ↓</option>
    <option value="length_delta_asc">Sort: Length delta ↑</option>
  </select>
  <div class="nav-group">
    <button id="btnFirst" title="Home">⏮</button>
    <button id="btnPrev" title="Previous">◀ Prev</button>
    <span class="counter" id="counter">0 / 0</span>
    <button id="btnNext" title="Next">Next ▶</button>
    <button id="btnLast" title="End">⏭</button>
  </div>
</div>
<div class="item-container" id="viewer"></div>
<script>
window.onerror = function(msg, url, line) {{
  var b = document.getElementById('errorBanner');
  b.style.display = 'block';
  b.textContent = 'JS Error: ' + msg + ' (line ' + line + ')';
}};
</script>
<script>
"""
    html_out += js_code
    html_out += "\n</script>\n</body>\n</html>"
    return html_out


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML comparison viewer for two model responses."
    )
    parser.add_argument(
        "--model1-rollout",
        default="checkpoint_eval/eval_Qwen3-30B-GDPO-150.jsonl",
        help="Path to model 1 rollout JSONL (with responses)",
    )
    parser.add_argument(
        "--model1-graded",
        default="checkpoint_eval/eval_Qwen3-30B-GDPO-150_graded.jsonl",
        help="Path to model 1 graded JSONL (with scores)",
    )
    parser.add_argument(
        "--model2-rollout",
        default="checkpoint_eval/eval_Qwen3-30B-PPO-Norm-150.jsonl",
        help="Path to model 2 rollout JSONL (with responses)",
    )
    parser.add_argument(
        "--model2-graded",
        default="checkpoint_eval/eval_Qwen3-30B-PPO-Norm-150_graded.jsonl",
        help="Path to model 2 graded JSONL (with scores)",
    )
    parser.add_argument("--model1-name", default="GDPO-150")
    parser.add_argument("--model2-name", default="PPO-Norm-150")
    parser.add_argument("--output", "-o", default="comparison.html")
    parser.add_argument(
        "--analysis",
        default=None,
        help="Optional quality analysis JSONL from analyze_quality_vllm.py.",
    )
    args = parser.parse_args()

    print(f"Loading {args.model1_name} rollout: {args.model1_rollout}")
    m1_rollout = load_jsonl(args.model1_rollout)
    print(f"Loading {args.model1_name} graded: {args.model1_graded}")
    m1_graded = load_jsonl(args.model1_graded)
    print(f"Loading {args.model2_name} rollout: {args.model2_rollout}")
    m2_rollout = load_jsonl(args.model2_rollout)
    print(f"Loading {args.model2_name} graded: {args.model2_graded}")
    m2_graded = load_jsonl(args.model2_graded)

    print(f"Merging data ({len(m1_rollout)} rows)...")
    items = build_comparison_data(
        m1_rollout, m1_graded, m2_rollout, m2_graded,
        args.model1_name, args.model2_name
    )

    # Merge quality analysis if provided
    has_analysis = False
    if args.analysis:
        analysis_rows = load_jsonl(args.analysis)
        analysis_by_idx = {row["idx"]: row for row in analysis_rows}
        merged_count = 0
        for it in items:
            if it["idx"] in analysis_by_idx:
                a = analysis_by_idx[it["idx"]]
                it["m1_quality"] = a.get("m1_analysis", {})
                it["m2_quality"] = a.get("m2_analysis", {})
                merged_count += 1
            else:
                it["m1_quality"] = {}
                it["m2_quality"] = {}
        has_analysis = merged_count > 0
        print(f"Merged quality analysis for {merged_count}/{len(items)} items")

    summary = compute_summary(items, args.model1_name, args.model2_name,
                              has_analysis=has_analysis)

    print(f"Summary: {args.model1_name} wins {summary['m1_win_rate']}%, "
          f"{args.model2_name} wins {summary['m2_win_rate']}%, "
          f"ties {summary['tie_rate']}%")

    print(f"Generating HTML...")
    html_content = generate_html(items, summary, args.model1_name, args.model2_name)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)

    size_mb = len(html_content.encode("utf-8")) / (1024 * 1024)
    print(f"Written to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
