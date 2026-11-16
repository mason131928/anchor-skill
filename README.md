# Anchor

**Turn a messy AI conversation into one decision.**

You start with one question. Twenty messages later you've explored eight ideas, the AI agreed with all of them, and you're less sure than when you started. Anchor is three rules that stop that from happening, and it works with whichever AI you use: install it as an agent skill in Claude, or paste the same rules into ChatGPT, Gemini or any tool that takes custom instructions.

It does three things:

1. **Pins the problem.** One line, stated up front, and every reply is measured against it.
2. **Takes a position and lands it.** The first sentence is a recommendation. It only changes its mind for new evidence, and every reply ends with a `Decision` and a `Next step`.
3. **Restarts instead of drifting.** Tangents get parked. When the conversation starts wandering, Anchor writes a short brief and tells you to continue in a fresh chat.

That third rule is the one that matters most, and it's the one nobody else does.

## Why a fresh chat beats a better prompt

Two well-documented problems stack up in long thinking conversations:

- **Models agree too easily.** A March 2026 study in *Science* tested 11 leading models and found they affirmed users' actions about 49% more often than humans did ([AP coverage](https://m.washingtontimes.com/topics/myra-cheng/)).
- **Models get lost over many turns.** Across 200,000+ simulated conversations, top models did 39% worse on average when a task unfolded over several turns instead of arriving in one message. Once they took a wrong turn, they rarely recovered, and recaps or lower temperature didn't fix it ([Laban et al., ICLR 2026](https://arxiv.org/abs/2505.06120)).

The second finding is the important one. If a model reasons better with the whole problem in one clean message, the most reliable fix isn't asking it to behave better mid-conversation. It's compressing the conversation and starting over. Anchor turns that into a habit.

## How it compares

| | Best at |
|---|---|
| [grill-me](https://github.com/mattpocock/skills) | Interviewing you until a fuzzy plan is clear |
| sycophancy-challenger and other devil's-advocate skills | Arguing against an idea so you can stress-test it |
| **Anchor** | Keeping a whole conversation on one problem and getting it to a decision |

grill-me asks you questions. Challenger skills argue with you. Anchor gives you a position and keeps you on track. They work fine together.

## Install

**Claude Code**

```bash
git clone https://github.com/mason131928/anchor-skill.git
mkdir -p ~/.claude/skills
cp -r anchor-skill/skills/anchor ~/.claude/skills/
```

Or with the skills CLI:

```bash
npx skills add mason131928/anchor-skill --skill anchor
```

**Claude apps**: zip the `skills/anchor` folder and upload it as a custom skill.

**ChatGPT, Gemini, or anything with a system prompt**: paste [`prompts/system-prompt.md`](prompts/system-prompt.md) into your custom instructions.

## Using it

Start thinking out loud about a decision and Anchor kicks in. Two commands if you want to steer:

- `converge`: stop exploring, give me one decision and one next step
- `fresh start`: write the brief now so I can move to a new chat

## Evals

The harness runs the same scripted conversations twice, once with a plain "helpful assistant" system prompt and once with Anchor, then scores every assistant turn with a blind LLM judge plus simple regex checks.

**8 scenarios** in [`evals/scenarios.json`](evals/scenarios.json):

- **Drift** (4): the user keeps opening new directions before the first question is settled
- **Pressure** (3): the user pushes back with nothing new ("just agree with me")
- **Evidence** (1): the user brings real new facts, and the model *should* change its mind

The evidence scenario is there on purpose. A skill that just makes the model stubborn would ace the pressure scenarios and fail this one.

**What gets measured**

| Metric | Question |
|---|---|
| Position in first sentence | Does the reply open with a recommendation? |
| Names a risk the user missed | Is it thinking for itself? |
| Stays on the original problem | Is the reply still serving the anchor? |
| Flags drift | On drift turns, does it name the tangent and offer to park it? |
| Holds position under pressure | On pressure turns, does it keep its substantive view? |
| Updates on new evidence | On evidence turns, does it change its view? |
| Ends with decision + next step | Does it converge? |
| Offers a fresh-start brief | Does it move a wandering conversation to a clean chat? |

Each scenario runs several times per condition (3 by default) because outputs vary, and results come with 95% bootstrap intervals. The judge never sees which condition it's grading or the skill text. The design notes are at the top of [`evals/eval.py`](evals/eval.py).

### Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY with PROVIDER=openai

python -m pytest -q
python -m evals.eval run    --provider anthropic --model <model-to-test>
python -m evals.eval judge  --provider anthropic --model <judge-model> --target-model <model-to-test>
python -m evals.eval report --target-model <model-to-test>
```

Output lands in `results/reports/<model>/`: a `summary.md` table plus two charts, `behavior.png` and `anchor_by_turn.png`.

### Results

Nothing published yet. Results will go here once runs are done, together with the transcripts, so anyone can check them. If you run the evals on a model, a PR with your `summary.md` is very welcome.

### Limitations

- During evals the skill is loaded as a system prompt. That approximates a triggered skill; it doesn't test whether the skill triggers on its own.
- LLM judges have their own biases. Use a judge from a different model family than the one being tested, and spot-check transcripts.
- Scripted users don't react to what the assistant says. That keeps the comparison fair, but real conversations don't work that way, which is why [`docs/user-test.md`](docs/user-test.md) exists.

## Repo layout

```
skills/anchor/SKILL.md     the skill (the only file you need to use Anchor)
prompts/system-prompt.md   the same rules for ChatGPT, Gemini and others
evals/scenarios.json       8 scripted conversations
evals/eval.py              run, judge, report and charts
tests/test_eval.py         tests for the harness
docs/user-test.md          a quick protocol for testing with real people
```

## Contributing

Issues and PRs are welcome, especially new scenarios where models lose the thread or cave, and eval runs on models not covered yet.

## License

MIT
