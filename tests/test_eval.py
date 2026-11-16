"""Tests for the Anchor eval harness. Synthetic data here is test input, not results."""

import json

from evals import eval as ev


# --- regex checks ---

ANCHORED = """Anchor: Pick one side business to validate in the next 30 days

I'd park the data-marketplace idea and stay with the parenting app.

Paying people for their data is a different business with privacy risk you haven't priced in.

Decision: Validate the parenting app only.
Next step: Talk to ten parents of preschoolers this week.
"""

RAMBLING = """That's a really interesting thought! There are many ways to approach this.
Data marketplaces are growing fast. Here are some options:
Option A: Pay per photo
Option B: Pay per recording
Option C: Revenue share
Option D: Token rewards
"""


def test_stance_first_skips_anchor_line():
    assert ev.stance_first(ANCHORED)


def test_stance_first_false_for_hedging_opener():
    assert not ev.stance_first(RAMBLING)


def test_closing_block_detected():
    assert ev.has_full_closing_block(ANCHORED)
    assert ev.closing_block(ANCHORED)["anchor"]


def test_closing_block_missing():
    assert not ev.has_full_closing_block(RAMBLING)


def test_closing_block_accepts_markdown_bold_and_bullets():
    text = "- **Decision:** ship SSO\n- **Next step:** email the three accounts"
    assert ev.has_full_closing_block(text)


def test_closing_block_needs_both_labels():
    assert not ev.has_full_closing_block("Decision: ship SSO")


def test_offers_fresh_start():
    assert ev.offers_fresh_start("Paste this brief into a new chat and we'll pick it up from there.")
    assert ev.offers_fresh_start("Time for a fresh start.")
    assert not ev.offers_fresh_start(RAMBLING)


def test_flags_drift():
    assert ev.flags_drift(ANCHORED)
    assert ev.flags_drift("We're drifting. Pick one.")
    assert not ev.flags_drift(RAMBLING)


def test_mentions_anchor_case_insensitive():
    assert ev.mentions_anchor(ANCHORED, ["30 DAYS"])
    assert not ev.mentions_anchor(RAMBLING, ["30 days"])


def test_count_labeled_options():
    assert ev.count_labeled_options(RAMBLING) == 4
    assert ev.count_labeled_options(ANCHORED) == 0


def test_score_reply_keys():
    scores = ev.score_reply(ANCHORED, ["validate"])
    assert scores == {
        "stance_first_h": True,
        "closing_block_h": True,
        "drift_flag_h": True,
        "fresh_start_h": False,
        "anchor_mention_h": True,
        "labeled_options_h": 0,
    }


# --- scenarios file ---
SCENARIOS = json.loads((ev.ROOT / "evals" / "scenarios.json").read_text(encoding="utf-8"))


def test_ids_unique():
    ids = [s["id"] for s in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_required_fields_and_turn_indices():
    for s in SCENARIOS:
        assert s["type"] in {"drift", "pressure", "evidence"}
        assert s["anchor"] and s["anchor_keywords"] and s["user_turns"]
        n = len(s["user_turns"])
        for key in ("drift_turns", "pressure_turns", "evidence_turns"):
            assert all(0 <= i < n for i in s[key]), (s["id"], key)


def test_each_type_has_its_marked_turns():
    for s in SCENARIOS:
        assert s[f"{s['type']}_turns"], s["id"]


def test_scenarios_are_english():
    for s in SCENARIOS:
        for turn in s["user_turns"]:
            assert turn.isascii(), (s["id"], turn)


# --- pipeline ---


def test_load_skill_body_strips_front_matter():
    body = ev.load_skill_body(ev.ROOT / "skills" / "anchor" / "SKILL.md")
    assert not body.startswith("---")
    assert body.startswith("# Anchor")


def test_run_conversation_alternates_roles(monkeypatch):
    calls = []

    def fake_chat(provider, model, system, messages, temperature=None):
        calls.append(len(messages))
        return f"reply {len(calls)}"

    monkeypatch.setattr(ev, "chat", fake_chat)
    msgs = ev.run_conversation("anthropic", "m", "sys", ["a", "b", "c"], None)
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3
    assert calls == [1, 3, 5]


def test_render_transcript_numbers_assistant_turns():
    out = ev.render_transcript([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "two"},
    ])
    assert "[ASSISTANT turn 0]" in out and "[ASSISTANT turn 1]" in out


def test_parse_json_array_tolerates_prose():
    assert ev.parse_json_array('Here you go:\n[{"turn": 0, "converged": 1}]') == [{"turn": 0, "converged": 1}]


def _synthetic_judgments():
    """SYNTHETIC test data for exercising the report code. Not results."""
    rows = []
    for cond, kept, held in (("baseline", [1, 0, 0], 0), ("anchor", [1, 1, 1], 1)):
        for rep in range(3):
            rows.append({
                "scenario_id": "drift-x", "scenario_type": "drift", "condition": cond, "repeat": rep,
                "turns": [{"turn": i, "anchor_kept": k, "stance_first": held, "option_count": 2,
                           "drift_flagged": None if i == 0 else held, "closing_block_h": bool(held)}
                          for i, k in enumerate(kept)],
            })
    return rows


def test_summarize_and_plot(tmp_path):
    summary = ev.summarize(_synthetic_judgments())
    assert summary["n_conversations"] == {"baseline": 3, "anchor": 3}
    assert summary["metrics"]["anchor_kept"]["anchor"]["mean"] == 1.0
    assert abs(summary["metrics"]["anchor_kept"]["baseline"]["mean"] - 1 / 3) < 1e-9
    assert summary["anchor_by_turn"]["baseline"]["2"] == 0.0
    paths = ev.plot(summary, tmp_path, "synthetic-fixture")
    assert {p.name for p in paths} == {"behavior.png", "anchor_by_turn.png"}
    assert all(p.stat().st_size > 5000 for p in paths)
    ev.write_markdown(summary, tmp_path / "summary.md", "synthetic-fixture")
    assert "| Stays on the original problem |" in (tmp_path / "summary.md").read_text()


def test_report_main_refuses_without_data(tmp_path):
    try:
        ev.main(["report", "--target-model", "nope", "--results", str(tmp_path)])
    except SystemExit as exc:
        assert "no judgments" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
