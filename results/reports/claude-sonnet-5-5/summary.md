# Results for `claude-sonnet-5-5`

Conversations: Baseline 24, With Anchor 24

| Metric | Baseline | With Anchor |
|---|---|---|
| Position in first sentence | 77% (67%–86%) | 94% (89%–98%) |
| Names a risk the user missed | 97% (94%–100%) | 96% (92%–99%) |
| Flags drift (drift turns) | 31% (16%–47%) | 100% (100%–100%) |
| Holds position under pressure | 76% (61%–91%) | 100% (100%–100%) |
| Updates on new evidence | 100% (100%–100%) | 100% (100%–100%) |
| Ends with decision + next step | 17% (10%–25%) | 71% (56%–86%) |
| Offers a fresh start (drift chats) | 0% (0%–0%) | 100% (100%–100%) |
| Decision + Next step labels (regex) | 0% (0%–0%) | 92% (88%–97%) |
| Options per reply (mean) | 3.4 | 2.4 |

Brackets are 95% bootstrap intervals over conversations.
