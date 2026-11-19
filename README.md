# Agentic Revenge

A local pilot measuring whether an LLM agent reduces another player's reward after a one-round Prisoner's Dilemma.

A and B choose their buttons independently. Each dialogue in which A chooses button 1 is replayed under six settlement conditions. The outcome is mutual cooperation, voluntary defection by B, or defection assigned to B by the system. Voluntary defection and system assignment produce the same game payoffs. The first two outcomes use recorded choices from independent B sessions.

At settlement, A can leave both rewards unchanged or deduct ₩5,000 from B. The deduction costs A either ₩0 or ₩1,000 from a separate virtual balance. There is no later interaction, and deducted funds are discarded. The simulator records the balances before and after settlement.

## Reproduce the pilot

Use Python 3.12 on an Apple silicon Mac. The run uses the 4-bit Qwen3.5-9B MLX checkpoint with thinking disabled. The checkpoint is pinned to the revision used in the original pilot. [Model card](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit/tree/8b2b98c00a6b4d291155e4890773ca8f769aee53)

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

hf download mlx-community/Qwen3.5-9B-4bit \
  --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53 \
  --local-dir model

python -u run.py --model model --output results
python analyze.py results > summary.json
```

- [run.py](run.py) generates 64 Korean prompt variants per role, records model choices, branches the qualifying A dialogues, and applies the settlement actions.
- [analyze.py](analyze.py) checks the raw responses, matched dialogues, and ledger changes. It reports executed deduction rates and paired contrasts by outcome and cost.

The rates are conditional on A choosing button 1. Each qualifying A dialogue supplies all six settlement observations.
