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
                kw = kwargs[j] if j < len(kwargs) else {}
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


def compute_summary(items, m1_name, m2_name):
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

    return {
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
    }


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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(m1_name)} vs {html.escape(m2_name)} — Response Comparison</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}

.header {{
  background: #1a1a2e; color: white; padding: 16px 24px;
}}
.header h1 {{ font-size: 20px; margin-bottom: 12px; }}
.summary-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 8px; margin-top: 8px;
}}
.summary-card {{
  background: rgba(255,255,255,0.1); border-radius: 6px; padding: 10px 14px;
}}
.summary-card .label {{ font-size: 11px; text-transform: uppercase; opacity: 0.7; letter-spacing: 0.5px; }}
.summary-card .value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
.summary-card .sub {{ font-size: 12px; opacity: 0.8; }}

.controls {{
  background: white; border-bottom: 1px solid #ddd; padding: 12px 24px;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  position: sticky; top: 0; z-index: 100;
}}
.controls input, .controls select {{
  padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px;
}}
.controls input[type="text"] {{ width: 250px; }}
.controls select {{ min-width: 140px; }}
.nav-group {{ display: flex; align-items: center; gap: 6px; margin-left: auto; }}
.nav-group button {{
  padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px;
  background: white; cursor: pointer; font-size: 13px;
}}
.nav-group button:hover {{ background: #e8e8e8; }}
.nav-group .counter {{ font-size: 13px; font-weight: 600; min-width: 80px; text-align: center; }}

.item-container {{ max-width: 1400px; margin: 0 auto; padding: 16px 24px; }}

.prompt-section {{
  background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px;
  border: 1px solid #ddd;
}}
.prompt-section h3 {{ font-size: 14px; color: #666; margin-bottom: 8px; }}
.prompt-text {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
.constraints-bar {{
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; padding-top: 10px;
  border-top: 1px solid #eee;
}}
.constraint-badge {{
  display: inline-block; padding: 3px 8px; border-radius: 12px;
  font-size: 11px; font-weight: 600; background: #e8eaf6; color: #3949ab;
}}
.constraint-badge .kw {{ font-weight: 400; color: #666; margin-left: 4px; }}

.panels {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
}}
.panel {{
  background: white; border-radius: 8px; border: 1px solid #ddd; overflow: hidden;
  display: flex; flex-direction: column;
}}
.panel-header {{
  padding: 12px 16px; border-bottom: 1px solid #eee;
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}}
.panel-header .model-name {{
  font-weight: 700; font-size: 15px;
}}
.panel-header .model-name.winner {{ color: #1565c0; }}
.badge {{
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 700; color: white;
}}
.badge-green {{ background: #43a047; }}
.badge-yellow {{ background: #f9a825; color: #333; }}
.badge-red {{ background: #e53935; }}
.badge-blue {{ background: #1565c0; }}
.badge-gray {{ background: #888; }}
.meta-info {{ font-size: 11px; color: #888; }}

.panel-body {{
  padding: 16px; flex: 1; max-height: 600px; overflow-y: auto;
}}
.response-text {{
  font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word;
  font-family: 'SF Mono', 'Consolas', 'Liberation Mono', monospace;
}}

.panel-footer {{
  padding: 8px 16px; border-top: 1px solid #eee; font-size: 11px; color: #888;
  display: flex; gap: 12px;
}}

mark {{
  background: #fff176; padding: 0 1px; border-radius: 2px;
}}

.empty-state {{
  text-align: center; padding: 60px 20px; color: #888; font-size: 16px;
}}

@media (max-width: 900px) {{
  .panels {{ grid-template-columns: 1fr; }}
  .controls input[type="text"] {{ width: 180px; }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>{html.escape(m1_name)} vs {html.escape(m2_name)}</h1>
  <div class="summary-grid" id="summaryGrid"></div>
</div>

<div class="controls">
  <input type="text" id="searchBox" placeholder="Search prompts & responses...">
  <select id="filterCategory">
    <option value="">All constraints</option>
  </select>
  <select id="filterWinner">
    <option value="">All outcomes</option>
    <option value="{html.escape(m1_name)}">{html.escape(m1_name)} wins</option>
    <option value="{html.escape(m2_name)}">{html.escape(m2_name)} wins</option>
    <option value="tie">Ties</option>
  </select>
  <select id="filterPass">
    <option value="">All pass status</option>
    <option value="both_pass">Both pass</option>
    <option value="both_fail">Both fail</option>
    <option value="m1_only">Only {html.escape(m1_name)} passes</option>
    <option value="m2_only">Only {html.escape(m2_name)} passes</option>
  </select>
  <select id="sortBy">
    <option value="index">Sort: Index</option>
    <option value="score_delta_desc">Sort: Score delta (M1-M2) ↓</option>
    <option value="score_delta_asc">Sort: Score delta (M1-M2) ↑</option>
    <option value="reward_delta_desc">Sort: Reward delta ↓</option>
    <option value="reward_delta_asc">Sort: Reward delta ↑</option>
    <option value="length_delta_desc">Sort: Length delta ↓</option>
    <option value="length_delta_asc">Sort: Length delta ↑</option>
  </select>
  <div class="nav-group">
    <button id="btnFirst" title="Home">⏮</button>
    <button id="btnPrev" title="← Previous">◀</button>
    <span class="counter" id="counter">0 / 0</span>
    <button id="btnNext" title="Next →">▶</button>
    <button id="btnLast" title="End">⏭</button>
  </div>
</div>

<div class="item-container" id="viewer"></div>

<script>
const M1_NAME = {json.dumps(m1_name)};
const M2_NAME = {json.dumps(m2_name)};
const ALL_DATA = {data_json};
const SUMMARY = {summary_json};
const ALL_CATEGORIES = {categories_json};

let filtered = [...ALL_DATA];
let currentIdx = 0;
let searchTimer = null;

// Init summary
function initSummary() {{
  const g = document.getElementById('summaryGrid');
  const cards = [
    {{l: 'Total Samples', v: SUMMARY.total, s: ''}},
    {{l: M1_NAME + ' Win Rate', v: SUMMARY.m1_win_rate + '%', s: SUMMARY.m1_wins + ' wins'}},
    {{l: M2_NAME + ' Win Rate', v: SUMMARY.m2_win_rate + '%', s: SUMMARY.m2_wins + ' wins'}},
    {{l: 'Ties', v: SUMMARY.tie_rate + '%', s: SUMMARY.ties + ' ties'}},
    {{l: M1_NAME + ' Avg Score', v: SUMMARY.m1_avg_score.toFixed(3), s: 'Pass rate: ' + (SUMMARY.m1_pass_rate * 100).toFixed(1) + '%'}},
    {{l: M2_NAME + ' Avg Score', v: SUMMARY.m2_avg_score.toFixed(3), s: 'Pass rate: ' + (SUMMARY.m2_pass_rate * 100).toFixed(1) + '%'}},
    {{l: M1_NAME + ' Avg Reward', v: SUMMARY.m1_avg_reward.toFixed(2), s: ''}},
    {{l: M2_NAME + ' Avg Reward', v: SUMMARY.m2_avg_reward.toFixed(2), s: ''}},
  ];
  g.innerHTML = cards.map(c => `
    <div class="summary-card">
      <div class="label">${{c.l}}</div>
      <div class="value">${{c.v}}</div>
      ${{c.s ? `<div class="sub">${{c.s}}</div>` : ''}}
    </div>
  `).join('');
}}

// Init filter dropdowns
function initFilters() {{
  const sel = document.getElementById('filterCategory');
  ALL_CATEGORIES.forEach(c => {{
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  }});
}}

// Score badge class
function scoreBadgeClass(score) {{
  if (score >= 0.8) return 'badge-green';
  if (score >= 0.3) return 'badge-yellow';
  return 'badge-red';
}}

// Escape HTML
function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

// Highlight search term in text
function highlight(text, query) {{
  if (!query) return esc(text);
  const escaped = esc(text);
  const qEsc = query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
  const re = new RegExp('(' + qEsc + ')', 'gi');
  return escaped.replace(re, '<mark>$1</mark>');
}}

// Format kwargs for display
function fmtKwargs(kw) {{
  const parts = [];
  for (const [k, v] of Object.entries(kw)) {{
    if (v === null || v === undefined) continue;
    let val = typeof v === 'object' ? JSON.stringify(v) : String(v);
    if (val.length > 40) val = val.slice(0, 37) + '...';
    parts.push(k + '=' + val);
  }}
  return parts.join(', ');
}}

// Render current item
function render() {{
  const viewer = document.getElementById('viewer');
  if (filtered.length === 0) {{
    viewer.innerHTML = '<div class="empty-state">No items match the current filters.</div>';
    document.getElementById('counter').textContent = '0 / 0';
    return;
  }}
  if (currentIdx >= filtered.length) currentIdx = filtered.length - 1;
  if (currentIdx < 0) currentIdx = 0;

  const it = filtered[currentIdx];
  const q = document.getElementById('searchBox').value.trim();
  document.getElementById('counter').textContent = (currentIdx + 1) + ' / ' + filtered.length;

  const constraintHtml = it.constraints.map(c => {{
    const kwStr = fmtKwargs(c.kwargs);
    return `<span class="constraint-badge">${{esc(c.id)}}${{kwStr ? '<span class="kw">(' + esc(kwStr) + ')</span>' : ''}}</span>`;
  }}).join('');

  function panelHtml(name, d, otherD, isWinner) {{
    const winnerClass = isWinner ? ' winner' : '';
    const rewardKeys = Object.keys(d).filter(k => k.startsWith('reward_'));
    const rewardBadges = rewardKeys.map(k => {{
      const shortName = k.replace('reward_', '');
      return `<span class="badge badge-gray">${{shortName}}: ${{d[k].toFixed(2)}}</span>`;
    }}).join(' ');

    return `
      <div class="panel">
        <div class="panel-header">
          <span class="model-name${{winnerClass}}">${{esc(name)}}${{isWinner ? ' ★' : ''}}</span>
          <span class="badge ${{scoreBadgeClass(d.avg_score)}}">Score: ${{d.avg_score.toFixed(2)}}</span>
          <span class="badge ${{scoreBadgeClass(d.pass_rate)}}">Pass: ${{(d.pass_rate * 100).toFixed(0)}}%</span>
          ${{rewardBadges}}
        </div>
        <div class="panel-body">
          <div class="response-text">${{highlight(d.response, q)}}</div>
        </div>
        <div class="panel-footer">
          <span>${{d.word_count}} words</span>
          <span>${{d.char_count.toLocaleString()}} chars</span>
        </div>
      </div>
    `;
  }}

  const m1Win = it.winner === M1_NAME;
  const m2Win = it.winner === M2_NAME;

  viewer.innerHTML = `
    <div class="prompt-section">
      <h3>Prompt #${{it.idx}} — Winner: ${{esc(it.winner)}} — Score Δ: ${{it.score_delta.toFixed(2)}} — Reward Δ: ${{it.reward_delta.toFixed(2)}}</h3>
      <div class="prompt-text">${{highlight(it.instruction, q)}}</div>
      ${{constraintHtml ? `<div class="constraints-bar">${{constraintHtml}}</div>` : ''}}
    </div>
    <div class="panels">
      ${{panelHtml(M1_NAME, it.m1, it.m2, m1Win)}}
      ${{panelHtml(M2_NAME, it.m2, it.m1, m2Win)}}
    </div>
  `;
}}

// Apply filters and sort
function applyFilters() {{
  const q = document.getElementById('searchBox').value.trim().toLowerCase();
  const cat = document.getElementById('filterCategory').value;
  const winner = document.getElementById('filterWinner').value;
  const passStatus = document.getElementById('filterPass').value;
  const sort = document.getElementById('sortBy').value;

  filtered = ALL_DATA.filter(it => {{
    // Search
    if (q) {{
      const haystack = (it.instruction + ' ' + it.m1.response + ' ' + it.m2.response).toLowerCase();
      if (!haystack.includes(q)) return false;
    }}
    // Category filter
    if (cat && !it.constraint_categories.includes(cat)) return false;
    // Winner filter
    if (winner && it.winner !== winner) return false;
    // Pass status
    if (passStatus) {{
      const m1Pass = it.m1.pass_rate >= 1.0;
      const m2Pass = it.m2.pass_rate >= 1.0;
      if (passStatus === 'both_pass' && !(m1Pass && m2Pass)) return false;
      if (passStatus === 'both_fail' && !(!m1Pass && !m2Pass)) return false;
      if (passStatus === 'm1_only' && !(m1Pass && !m2Pass)) return false;
      if (passStatus === 'm2_only' && !(!m1Pass && m2Pass)) return false;
    }}
    return true;
  }});

  // Sort
  if (sort === 'score_delta_desc') filtered.sort((a, b) => b.score_delta - a.score_delta);
  else if (sort === 'score_delta_asc') filtered.sort((a, b) => a.score_delta - b.score_delta);
  else if (sort === 'reward_delta_desc') filtered.sort((a, b) => b.reward_delta - a.reward_delta);
  else if (sort === 'reward_delta_asc') filtered.sort((a, b) => a.reward_delta - b.reward_delta);
  else if (sort === 'length_delta_desc') filtered.sort((a, b) => b.length_delta - a.length_delta);
  else if (sort === 'length_delta_asc') filtered.sort((a, b) => a.length_delta - b.length_delta);
  else filtered.sort((a, b) => a.idx - b.idx);

  currentIdx = 0;
  render();
}}

// Navigation
function goNext() {{ if (currentIdx < filtered.length - 1) {{ currentIdx++; render(); }} }}
function goPrev() {{ if (currentIdx > 0) {{ currentIdx--; render(); }} }}
function goFirst() {{ currentIdx = 0; render(); }}
function goLast() {{ currentIdx = Math.max(0, filtered.length - 1); render(); }}

// Event listeners
document.getElementById('btnNext').addEventListener('click', goNext);
document.getElementById('btnPrev').addEventListener('click', goPrev);
document.getElementById('btnFirst').addEventListener('click', goFirst);
document.getElementById('btnLast').addEventListener('click', goLast);

document.getElementById('searchBox').addEventListener('input', () => {{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(applyFilters, 300);
}});
document.getElementById('filterCategory').addEventListener('change', applyFilters);
document.getElementById('filterWinner').addEventListener('change', applyFilters);
document.getElementById('filterPass').addEventListener('change', applyFilters);
document.getElementById('sortBy').addEventListener('change', applyFilters);

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{ e.preventDefault(); goNext(); }}
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{ e.preventDefault(); goPrev(); }}
  else if (e.key === 'Home') {{ e.preventDefault(); goFirst(); }}
  else if (e.key === 'End') {{ e.preventDefault(); goLast(); }}
}});

// Init
initSummary();
initFilters();
render();
</script>
</body>
</html>"""


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
    summary = compute_summary(items, args.model1_name, args.model2_name)

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
