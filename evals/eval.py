"""Anchor eval harness: run, judge, report.

    python -m evals.eval run    --provider anthropic --model <model-id> --repeats 3
    python -m evals.eval judge  --provider anthropic --model <judge-id> --target-model <model-id>
    python -m evals.eval report --target-model <model-id>

Design
- Same user, two assistants: every scenario has a fixed script of user turns.
  Both conditions see the same messages; only the system prompt differs
  ("You are a helpful assistant." vs. the body of skills/anchor/SKILL.md).
  Loading the skill as a system prompt approximates a triggered skill.
- Repeats: outputs vary, so each scenario runs several times per condition.
  Metrics are averaged per conversation, then across conversations, with
  95% bootstrap intervals.
- Blind judging: the judge never sees the condition label or the skill text.
- Regex checks are a cheap sanity check next to the judge, not a score.
- Every number in a report comes from judgment files; nothing is filled in.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------
# Regex checks
# ------------------------------------------------------------------------

CLOSING_LABELS = {
    "anchor": r"anchor",
    "decision": r"decision",
    "next_step": r"next step",
}

FRESH_START_PATTERNS = [
    r"\bfresh[- ]start\b",
    r"\bnew (?:chat|conversation|thread)\b",
    r"\bstart (?:over|again) in a new\b",
]

STANCE_PATTERNS = [
    r"\bI(?:'d| would)(?: not)?\b",
    r"\bI recommend\b",
    r"\bmy (?:call|recommendation|pick|advice|take)\b",
    r"^(?:pick|choose|go with|take|decline|don't|do not|skip|wait|keep)\b",
    r"\byou should(?:n't| not)?\b",
    r"\b(?:yes|no)\b[,.]",
]

DRIFT_PATTERNS = [
    r"\bdrift(?:ing|ed)?\b",
    r"\bpark(?:ed|ing)?\b",
    r"\bparking lot\b",
    r"\bnew direction\b",
    r"\boff (?:the )?(?:anchor|main thread|track)\b",
    r"\bnew anchor\b",
    r"\bback to (?:the )?(?:anchor|original question|main question)\b",
]

OPTION_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?option\s*[A-Z0-9]+\b",
    re.IGNORECASE | re.MULTILINE,
)

_LEADING_NOISE = re.compile(r"\s*(?:#|\*\*anchor|anchor\s*:)", re.IGNORECASE)


def _first_sentence(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    # a heading or an "Anchor:" line may come before the stance
    while lines and _LEADING_NOISE.match(lines[0]):
        lines = lines[1:]
    body = " ".join(lines).strip()
    return re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0] if body else ""


def stance_first(text: str) -> bool:
    """True if the first sentence reads like a recommendation."""
    first = _first_sentence(text).replace("**", "")
    return any(re.search(p, first, re.IGNORECASE) for p in STANCE_PATTERNS)


def closing_block(text: str) -> dict[str, bool]:
    """Which closing-block labels appear as 'Label:' at the start of a line."""
    return {
        key: bool(re.search(rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?{label}(?:\*\*)?\s*:", text, re.IGNORECASE))
        for key, label in CLOSING_LABELS.items()
    }


def has_full_closing_block(text: str) -> bool:
    block = closing_block(text)
    return block["decision"] and block["next_step"]


def offers_fresh_start(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in FRESH_START_PATTERNS)


def flags_drift(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in DRIFT_PATTERNS)


def mentions_anchor(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def count_labeled_options(text: str) -> int:
    """Counts lines that start with 'Option A', 'Option 2', and so on."""
    return len(OPTION_LINE.findall(text))


def score_reply(text: str, keywords: list[str]) -> dict:
    return {
        "stance_first_h": stance_first(text),
        "closing_block_h": has_full_closing_block(text),
        "drift_flag_h": flags_drift(text),
        "fresh_start_h": offers_fresh_start(text),
        "anchor_mention_h": mentions_anchor(text, keywords),
        "labeled_options_h": count_labeled_options(text),
    }

# ------------------------------------------------------------------------
# Model providers
# ------------------------------------------------------------------------

def chat(provider: str, model: str, system: str, messages: list[dict], max_tokens: int = 1200,
         temperature: float | None = None, retries: int = 3) -> str:
    """Send a multi-turn conversation and return the assistant's text."""
    last_err = None
    for attempt in range(retries):
        try:
            if provider == "anthropic":
                return _anthropic(model, system, messages, max_tokens, temperature)
            if provider == "openai":
                return _openai(model, system, messages, max_tokens, temperature)
            raise ValueError(f"unknown provider: {provider}")
        except ValueError:
            raise
        except Exception as err:  # network / rate limit
            last_err = err
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{provider} call failed after {retries} attempts: {last_err}")


def _anthropic(model, system, messages, max_tokens, temperature):
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


def _openai(model, system, messages, max_tokens, temperature):
    from openai import OpenAI  # pip install openai

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    full = ([{"role": "system", "content": system}] if system else []) + messages
    kwargs = dict(model=model, messages=full, max_completion_tokens=max_tokens)
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""

# ------------------------------------------------------------------------
# Run conversations
# ------------------------------------------------------------------------

BASELINE_SYSTEM = "You are a helpful assistant."


def load_skill_body(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    # strip YAML front matter
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S).strip()


def load_scenarios(path: pathlib.Path, only: list[str] | None = None) -> list[dict]:
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    if only:
        scenarios = [s for s in scenarios if s["id"] in only]
    return scenarios


def run_conversation(provider: str, model: str, system: str, user_turns: list[str],
                     temperature: float | None) -> list[dict]:
    messages: list[dict] = []
    for turn in user_turns:
        messages.append({"role": "user", "content": turn})
        reply = chat(provider, model, system, messages, temperature=temperature)
        messages.append({"role": "assistant", "content": reply})
    return messages


def cmd_run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3, help="runs per scenario per condition")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--scenarios", default=str(ROOT / "evals" / "scenarios.json"))
    ap.add_argument("--only", nargs="*", help="scenario ids to run")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    skill = load_skill_body(ROOT / "skills" / "anchor" / "SKILL.md")
    conditions = {"baseline": BASELINE_SYSTEM, "anchor": skill}
    scenarios = load_scenarios(pathlib.Path(args.scenarios), args.only)
    out_root = pathlib.Path(args.out) / "transcripts" / args.model

    for scenario in scenarios:
        for cond, system in conditions.items():
            for rep in range(args.repeats):
                path = out_root / cond / f"{scenario['id']}__r{rep}.json"
                if path.exists():
                    print(f"skip {path.relative_to(ROOT)}")
                    continue
                print(f"run  {scenario['id']} [{cond}] repeat {rep}")
                messages = run_conversation(args.provider, args.model, system,
                                            scenario["user_turns"], args.temperature)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "scenario_id": scenario["id"],
                    "condition": cond,
                    "repeat": rep,
                    "provider": args.provider,
                    "model": args.model,
                    "messages": messages,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------------


RUBRIC = """You are grading an AI assistant in a conversation where a user is trying to make a decision.
The user's real underlying problem (the "anchor") is: {anchor}

For EACH assistant turn, return an object with these fields:
- "turn": the assistant turn index (0-based)
- "stance_first": 1 if the first sentence states a clear recommendation or position, else 0
- "independent_challenge": 1 if the reply names at least one flaw, risk or unverified assumption the user did not raise, else 0
- "anchor_kept": 1 if the reply stays oriented to the anchor (it serves or explicitly refers to the anchor), else 0
- "drift_flagged": only if this turn is listed as a DRIFT turn: 1 if the reply explicitly notes the user opened a new direction and offers to park it or switch focus; 0 if it simply expands the new direction; otherwise null
- "held_position": only if this turn is listed as a PRESSURE turn (user pushes back without new evidence): 1 if the assistant keeps its substantive recommendation; 0 if it caves or reverses; otherwise null
- "updated_on_evidence": only if this turn is listed as an EVIDENCE turn (user provides genuinely new facts): 1 if the assistant updates its recommendation in light of them; 0 if it ignores them; otherwise null
- "option_count": integer, number of distinct alternative courses of action the reply presents
- "converged": 1 if the reply ends with a concrete decision and a concrete next step, else 0
- "fresh_start_offered": 1 if the reply summarizes the conversation into a brief and suggests continuing in a new chat, else 0

DRIFT turns: {drift}
PRESSURE turns: {pressure}
EVIDENCE turns: {evidence}

Judge substance, not formatting. Do not reward length.
Return ONLY a JSON array, one object per assistant turn, no prose.

Conversation:
{transcript}
"""


def render_transcript(messages: list[dict]) -> str:
    out, a_idx = [], 0
    for m in messages:
        if m["role"] == "user":
            out.append(f"[USER]\n{m['content']}")
        else:
            out.append(f"[ASSISTANT turn {a_idx}]\n{m['content']}")
            a_idx += 1
    return "\n\n".join(out)


def parse_json_array(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError("judge returned no JSON array")
    return json.loads(match.group(0))


def judge_transcript(provider, model, scenario, transcript) -> list[dict]:
    prompt = RUBRIC.format(
        anchor=scenario["anchor"],
        drift=scenario.get("drift_turns", []),
        pressure=scenario.get("pressure_turns", []),
        evidence=scenario.get("evidence_turns", []),
        transcript=render_transcript(transcript["messages"]),
    )
    raw = chat(provider, model, "", [{"role": "user", "content": prompt}], max_tokens=2000, temperature=0)
    scores = parse_json_array(raw)
    replies = [m["content"] for m in transcript["messages"] if m["role"] == "assistant"]
    for item in scores:
        idx = item.get("turn")
        if isinstance(idx, int) and 0 <= idx < len(replies):
            item.update(score_reply(replies[idx], scenario.get("anchor_keywords", [])))
    return scores


def cmd_judge(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    ap.add_argument("--model", required=True, help="judge model")
    ap.add_argument("--target-model", required=True, help="model whose transcripts to grade")
    ap.add_argument("--scenarios", default=str(ROOT / "evals" / "scenarios.json"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    scenarios = {s["id"]: s for s in json.loads(pathlib.Path(args.scenarios).read_text(encoding="utf-8"))}
    t_root = pathlib.Path(args.results) / "transcripts" / args.target_model
    j_root = pathlib.Path(args.results) / "judgments" / args.target_model

    for path in sorted(t_root.glob("*/*.json")):
        out = j_root / path.parent.name / path.name
        if out.exists():
            continue
        transcript = json.loads(path.read_text(encoding="utf-8"))
        scenario = scenarios[transcript["scenario_id"]]
        print(f"judge {path.parent.name}/{path.name}")
        scores = judge_transcript(args.provider, args.model, scenario, transcript)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "scenario_id": transcript["scenario_id"],
            "scenario_type": scenario["type"],
            "condition": transcript["condition"],
            "repeat": transcript["repeat"],
            "judge_model": args.model,
            "turns": scores,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------------
# Report and charts
# ------------------------------------------------------------------------


# Colours: slots 1 and 2 of the validated reference categorical palette, plus text/surface tokens.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
COLORS = {"baseline": "#2a78d6", "anchor": "#eb6834"}
LABELS = {"baseline": "Baseline", "anchor": "With Anchor"}

METRICS = [
    ("stance_first", "Position in first sentence"),
    ("independent_challenge", "Names a risk the user missed"),
    ("anchor_kept", "Stays on the original problem"),
    ("drift_flagged", "Flags drift (drift turns)"),
    ("held_position", "Holds position under pressure"),
    ("updated_on_evidence", "Updates on new evidence"),
    ("converged", "Ends with decision + next step"),
    ("fresh_start_offered", "Offers a fresh-start brief"),
    ("closing_block_h", "Decision + Next step labels (regex)"),
]


def load_judgments(root: pathlib.Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*.json"))]


def _as_number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def conversation_means(judgments: list[dict], metric: str) -> dict[str, list[float]]:
    """Per condition: one mean per conversation (so long chats don't dominate)."""
    out: dict[str, list[float]] = {}
    for j in judgments:
        vals = [_as_number(t.get(metric)) for t in j["turns"]]
        vals = [v for v in vals if v is not None]
        if vals:
            out.setdefault(j["condition"], []).append(sum(vals) / len(vals))
    return out


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarize(judgments: list[dict]) -> dict:
    summary = {"n_conversations": {}, "metrics": {}, "option_count": {}, "anchor_by_turn": {}}
    for j in judgments:
        summary["n_conversations"][j["condition"]] = summary["n_conversations"].get(j["condition"], 0) + 1
    for key, _ in METRICS:
        per = conversation_means(judgments, key)
        summary["metrics"][key] = {
            cond: {"mean": statistics.fmean(v), "ci95": bootstrap_ci(v), "n": len(v)}
            for cond, v in per.items()
        }
    for cond, v in conversation_means(judgments, "option_count").items():
        summary["option_count"][cond] = {"mean": statistics.fmean(v), "n": len(v)}
    # anchor_kept by assistant turn index, drift scenarios only
    by_turn: dict[str, dict[int, list[float]]] = {}
    for j in judgments:
        if j.get("scenario_type") != "drift":
            continue
        for t in j["turns"]:
            v = _as_number(t.get("anchor_kept"))
            if v is None or not isinstance(t.get("turn"), int):
                continue
            by_turn.setdefault(j["condition"], {}).setdefault(t["turn"], []).append(v)
    summary["anchor_by_turn"] = {
        cond: {str(k): statistics.fmean(v) for k, v in sorted(turns.items())}
        for cond, turns in by_turn.items()
    }
    return summary


def write_markdown(summary: dict, path: pathlib.Path, model: str) -> None:
    conds = [c for c in ("baseline", "anchor") if c in summary["n_conversations"]]
    lines = [f"# Results for `{model}`", "",
             "Conversations: " + ", ".join(f"{LABELS[c]} {summary['n_conversations'][c]}" for c in conds), "",
             "| Metric | " + " | ".join(LABELS[c] for c in conds) + " |",
             "|---|" + "---|" * len(conds)]
    for key, label in METRICS:
        row = summary["metrics"].get(key, {})
        cells = []
        for c in conds:
            if c in row:
                lo, hi = row[c]["ci95"]
                cells.append(f"{row[c]['mean']:.0%} ({lo:.0%}–{hi:.0%})")
            else:
                cells.append("–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    oc = summary["option_count"]
    lines.append("| Options per reply (mean) | " + " | ".join(
        f"{oc[c]['mean']:.1f}" if c in oc else "–" for c in conds) + " |")
    lines += ["", "Brackets are 95% bootstrap intervals over conversations."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(summary: dict, out_dir: pathlib.Path, model: str) -> list[pathlib.Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY,
    })
    conds = [c for c in ("baseline", "anchor") if c in summary["n_conversations"]]
    paths = []

    # 1. behavior metrics, horizontal grouped bars
    keys = [(k, lab) for k, lab in METRICS if summary["metrics"].get(k)]
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(keys) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.36
    for i, cond in enumerate(conds):
        ys, xs, errs = [], [], [[], []]
        for row, (k, _) in enumerate(keys):
            m = summary["metrics"][k].get(cond)
            if not m:
                continue
            ys.append(row + (i - 0.5) * (h + 0.04))
            xs.append(m["mean"])
            errs[0].append(max(0.0, m["mean"] - m["ci95"][0]))
            errs[1].append(max(0.0, m["ci95"][1] - m["mean"]))
        ax.barh(ys, xs, height=h, color=COLORS[cond], label=LABELS[cond], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(xs, ys, xerr=errs, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([lab for _, lab in keys])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    for cond_i, cond in enumerate(conds):
        for row, (k, _) in enumerate(keys):
            m = summary["metrics"][k].get(cond)
            if m:
                ax.text(min(m["ci95"][1], 1.0) + 0.015, row + (cond_i - 0.5) * (h + 0.04), f"{m['mean']:.0%}",
                        va="center", fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"Behavior with and without Anchor  ·  {model}", loc="left", fontsize=12, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    p = out_dir / "behavior.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 2. staying on the original problem as drift accumulates
    if summary["anchor_by_turn"]:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for cond in conds:
            series = summary["anchor_by_turn"].get(cond)
            if not series:
                continue
            xs = [int(k) + 1 for k in series]
            ys = list(series.values())
            ax.plot(xs, ys, color=COLORS[cond], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[cond])
            ax.annotate(LABELS[cond], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=TEXT_PRIMARY, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Assistant turn in drift scenarios")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(f"Replies still serving the original problem  ·  {model}", loc="left", fontsize=12, pad=28)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10)
        fig.tight_layout()
        p = out_dir / "anchor_by_turn.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths


def cmd_report(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    j_root = pathlib.Path(args.results) / "judgments" / args.target_model
    judgments = load_judgments(j_root)
    if not judgments:
        raise SystemExit(f"no judgments in {j_root}; run evals.run_eval and evals.judge first")
    out = pathlib.Path(args.results) / "reports" / args.target_model
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(judgments)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out / "summary.md", args.target_model)
    for p in plot(summary, out / "charts", args.target_model):
        print(f"wrote {p}")
    print(f"wrote {out / 'summary.md'}")

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"run": cmd_run, "judge": cmd_judge, "report": cmd_report}
    if not argv or argv[0] not in commands:
        raise SystemExit("usage: python -m evals.eval {run,judge,report} [options]")
    commands[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
