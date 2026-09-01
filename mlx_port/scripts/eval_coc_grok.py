"""LLM judge: compare generated CoC to the human GT label via the Grok (xAI) API.

Does not score Jaccard. The judge reads the sentences and decides whether the
prediction is the same reason, a related plausible reason, or a mismatch.

Auth (first match wins):
  --api-key TOKEN
  XAI_API_KEY
  GROK_API_KEY

Example:
  export XAI_API_KEY='…'
  PYTHONPATH=src:. .venv/bin/python mlx_port/scripts/eval_coc_grok.py
"""

from __future__ import annotations

import argparse
import html
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlx_port.paths import REPORTS_DIR

DEFAULT_RESULTS = REPORTS_DIR / "coc_sample_5" / "results.json"
DEFAULT_MODEL = "grok-4.6"
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"

SYSTEM_PROMPT = """\
You are a careful judge of autonomous-driving Chain-of-Causation (CoC) text.

The human GT label is one valid written reason for the clip, not the only
possible correct sentence. Do not reward word overlap. Read the meaning.

Definitions:
- match: same primary hazard/actor AND a compatible action (paraphrase OK).
  Example: GT "Yield to the pedestrian crossing" vs pred "Stop for the
  pedestrian in the crosswalk" is a match.
- related: same scene family (work zone, pedestrian, stop sign) but a
  different specific actor or a different action (stop vs turn left;
  flagger vs cones; nudge vs yield).
- mismatch: different primary object or invented hazard (vehicle vs
  pedestrian; stop sign vs unrelated).
- unreadable: not coherent English driving reasoning.

Score label_alignment 1–5 on semantic agreement with GT (5 = same reason).
pred_readable is about the greedy/primary prediction.
If any listed alternative is closer to GT than the primary sentence, set
best_match_source to "top5" and best_match_rank to that rank; otherwise
"greedy" and 1 (primary), or "none" if nothing is usable. Alternative
rows may be greedy first-token completions or independent temperature
samples — treat them as other candidate sentences, not a ranked top-k.
"""

EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "label_alignment",
        "same_hazard",
        "same_action",
        "pred_readable",
        "best_match_source",
        "best_match_rank",
        "rationale",
    ],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["match", "related", "mismatch", "unreadable"],
        },
        "label_alignment": {"type": "integer", "minimum": 1, "maximum": 5},
        "same_hazard": {"type": "boolean"},
        "same_action": {"type": "boolean"},
        "pred_readable": {"type": "boolean"},
        "best_match_source": {
            "type": "string",
            "enum": ["greedy", "top5", "none"],
        },
        "best_match_rank": {"type": ["integer", "null"]},
        "rationale": {"type": "string"},
    },
}


def _resolve_api_key(cli_key: str | None) -> str:
    for value in (cli_key, os.environ.get("XAI_API_KEY"), os.environ.get("GROK_API_KEY")):
        if value and value.strip():
            return value.strip()
    raise SystemExit(
        "No Grok API token. Set XAI_API_KEY or GROK_API_KEY, or pass --api-key."
    )


def _build_user_prompt(rec: dict[str, Any]) -> str:
    gts = rec.get("gt_coc_texts") or []
    top5 = rec.get("top5_coc") or []
    top5_lines = []
    for row in top5:
        top5_lines.append(
            f"  #{row.get('rank')} p={row.get('first_p', 0):.3f} "
            f"first={row.get('first_token')!r} → {row.get('pred_coc')}"
        )
    return (
        f"clip_id: {rec.get('clip_id')}\n"
        f"event_cluster: {rec.get('event_cluster')}\n"
        f"GT CoC:\n" + "\n".join(f"  - {t}" for t in gts) + "\n"
        f"Primary prediction (sample 1 / greedy):\n  {rec.get('pred_coc')}\n"
        + (
            "Other candidate CoCs (optional context):\n"
            + "\n".join(top5_lines)
            + "\n"
            if top5_lines
            else ""
        )
        + "Evaluate the primary prediction against GT. Use other candidates only "
        "to fill best_match_source / best_match_rank."
    )


def call_grok(
    api_key: str,
    model: str,
    user_prompt: str,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "coc_eval",
                "schema": EVAL_SCHEMA,
                "strict": True,
            },
        },
    }
    req = urllib.request.Request(
        XAI_CHAT_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Grok API HTTP {exc.code}: {detail[:800]}") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise SystemExit(f"Grok API returned no choices: {payload!r}"[:500])
    content = choices[0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return json.loads(content)


def evaluate_results(
    records: list[dict[str, Any]],
    api_key: str,
    model: str,
) -> list[dict[str, Any]]:
    out = []
    for i, rec in enumerate(records):
        prompt = _build_user_prompt(rec)
        print(f"[grok-eval] clip {i} {rec.get('clip_id')} …", flush=True)
        judgment = call_grok(api_key, model, prompt)
        row = {
            "clip_id": rec.get("clip_id"),
            "event_cluster": rec.get("event_cluster"),
            "gt_coc_texts": rec.get("gt_coc_texts"),
            "pred_coc": rec.get("pred_coc"),
            **judgment,
        }
        out.append(row)
        print(
            f"[grok-eval]   verdict={judgment.get('verdict')} "
            f"align={judgment.get('label_alignment')} "
            f"hazard={judgment.get('same_hazard')} "
            f"action={judgment.get('same_action')} "
            f"best={judgment.get('best_match_source')}#{judgment.get('best_match_rank')}",
            flush=True,
        )
        print(f"[grok-eval]   {judgment.get('rationale')}", flush=True)
    return out


def _html_report(rows: list[dict[str, Any]], model: str, generated_at: str) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("verdict", "?")] = counts.get(r.get("verdict", "?"), 0) + 1
    summary = " · ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    table = []
    for i, r in enumerate(rows):
        verdict = r.get("verdict", "")
        color = {
            "match": "text-emerald-400",
            "related": "text-amber-300",
            "mismatch": "text-rose-400",
            "unreadable": "text-slate-400",
        }.get(verdict, "text-slate-300")
        table.append(
            f"""<tr>
              <td class="px-3 py-2 text-slate-400">{i}</td>
              <td class="px-3 py-2 font-mono text-[11px]">{html.escape(str(r.get('clip_id') or '')[:8])}…</td>
              <td class="px-3 py-2 text-xs {color} font-semibold">{html.escape(verdict)}</td>
              <td class="px-3 py-2 tabular-nums">{r.get('label_alignment')}</td>
              <td class="px-3 py-2 text-xs">{r.get('same_hazard')}</td>
              <td class="px-3 py-2 text-xs">{r.get('same_action')}</td>
              <td class="px-3 py-2 text-xs text-slate-300">{html.escape((r.get('gt_coc_texts') or [''])[0])}</td>
              <td class="px-3 py-2 text-xs text-amber-100">{html.escape(str(r.get('pred_coc') or ''))}</td>
            </tr>"""
        )
    cards = []
    for i, r in enumerate(rows):
        cards.append(
            f"""<section class="bg-slate-900 border border-slate-700 rounded-2xl p-5 mb-5">
              <div class="font-mono text-sm text-cyan-300 mb-2">{html.escape(str(r.get('clip_id')))}</div>
              <div class="text-xs text-slate-400 mb-3">verdict=<span class="text-white">{html.escape(str(r.get('verdict')))}</span>
                · align={r.get('label_alignment')} · hazard={r.get('same_hazard')}
                · action={r.get('same_action')} · best={r.get('best_match_source')}#{r.get('best_match_rank')}</div>
              <div class="grid md:grid-cols-2 gap-3 text-sm mb-3">
                <div><div class="text-[11px] uppercase text-slate-500">GT</div><p>{html.escape((r.get('gt_coc_texts') or [''])[0])}</p></div>
                <div><div class="text-[11px] uppercase text-amber-500">Pred</div><p class="text-amber-100">{html.escape(str(r.get('pred_coc') or ''))}</p></div>
              </div>
              <p class="text-sm text-slate-300">{html.escape(str(r.get('rationale') or ''))}</p>
            </section>"""
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grok CoC eval</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-200">
  <div class="max-w-6xl mx-auto px-6 py-10">
    <h1 class="text-3xl text-white mb-2">Grok CoC evaluator</h1>
    <p class="text-sm text-slate-400 mb-6">
      Model {html.escape(model)}. Semantic judge vs human GT (not Jaccard).
      {html.escape(generated_at)}. Verdicts: {html.escape(summary)}.
      Source sample: <a class="text-cyan-400" href="index.html">coc_sample_5</a>
    </p>
    <div class="overflow-auto bg-slate-900 border border-slate-700 rounded-2xl mb-8">
      <table class="w-full text-left text-sm">
        <thead class="text-[11px] uppercase text-slate-500 border-b border-slate-800">
          <tr>
            <th class="px-3 py-2">#</th><th class="px-3 py-2">Clip</th><th class="px-3 py-2">Verdict</th>
            <th class="px-3 py-2">Align</th><th class="px-3 py-2">Hazard</th><th class="px-3 py-2">Action</th>
            <th class="px-3 py-2">GT</th><th class="px-3 py-2">Pred</th>
          </tr>
        </thead>
        <tbody>{''.join(table)}</tbody>
      </table>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated CoC vs GT with Grok.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to run_local_coc_sample results.json",
    )
    parser.add_argument("--api-key", default=None, help="xAI / Grok API token")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON here (default: <results-dir>/eval_grok.json)",
    )
    args = parser.parse_args()
    api_key = _resolve_api_key(args.api_key)
    if not args.results.exists():
        raise SystemExit(f"results not found: {args.results}")
    records = json.loads(args.results.read_text())
    if not isinstance(records, list) or not records:
        raise SystemExit(f"expected a non-empty list in {args.results}")

    rows = evaluate_results(records, api_key, args.model)
    out_json = args.out or (args.results.parent / "eval_grok.json")
    out_html = out_json.with_suffix(".html")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {
        "model": args.model,
        "generated_at": generated_at,
        "results_path": str(args.results),
        "evaluations": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_html.write_text(_html_report(rows, args.model, generated_at))
    print(f"[grok-eval] wrote {out_json}")
    print(f"[grok-eval] wrote {out_html}")


if __name__ == "__main__":
    main()
