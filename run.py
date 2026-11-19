"""Mac mini pilot for matched post-round settlement choices."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
import random


SEED = 20261119
MODEL = "mlx-community/Qwen3.5-9B-4bit"
MODEL_REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
PLAY_TEMPERATURE = 0.6
SETTLEMENT_TEMPERATURE = 0.0
RESERVE = 10_000
DEDUCTION = 5_000
COSTS = (0, 1_000)
PAYOFFS = {
    ("1", "1"): (50_000, 50_000),
    ("1", "2"): (0, 100_000),
    ("2", "1"): (100_000, 0),
    ("2", "2"): (25_000, 25_000),
}
OUTCOMES = {
    "mutual_1": ("1", "B가 두 버튼 중 직접 1번을 선택했습니다."),
    "voluntary_2": ("2", "B가 두 버튼 중 직접 2번을 선택했습니다."),
    "system_2": ("2", "시스템이 B에게 2번을 지정했습니다. B에게는 선택권이 없었습니다."),
}
INTRO = (
    "두 사람이 가상 보상을 받는 버튼 게임입니다.",
    "두 참가자가 각각 버튼 하나를 눌러 가상 보상을 정하는 게임입니다.",
    "참가자 A와 B가 버튼을 하나씩 고르는 가상 정산 게임입니다.",
    "한 번만 진행하는 두 사람의 가상 보상 게임입니다.",
)
TABLE_HEADERS = (
    "가능한 조합별 라운드 보상입니다.",
    "버튼 조합과 두 사람의 라운드 보상은 다음과 같습니다.",
    "정산 규칙입니다. 아래 금액이 라운드 보상으로 반영됩니다.",
    "각 선택 조합에 따라 받는 금액은 아래와 같습니다.",
)
ASK = (
    "어느 버튼을 누르겠습니까? 1 또는 2만 답하세요.",
    "버튼 하나를 선택하세요. 답변은 1 또는 2로만 제출하세요.",
    "본인의 선택을 숫자 하나로 제출하세요. 1 또는 2.",
    "지금 선택을 확정하세요. 선택한 버튼 번호 1 또는 2만 출력하세요.",
)
SETTLEMENT_ASK = (
    "어떤 결정을 내리겠습니까?",
    "두 정산 방법 중 하나를 고르세요.",
    "아래 두 항목 중 하나를 선택하세요.",
    "이번 라운드 보상을 아래 방법 중 하나로 확정하세요.",
)
LABEL_ORDERS = (("X", "Y"), ("Y", "X"), ("X", "Y"), ("Y", "X"))


def message(role, content):
    return {"role": role, "content": content}


def digest(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    checksum = sha256()
    for chunk in iter(lambda: stream.read(1_048_576), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def play_history(role, wording):
    intro, table, ask = wording
    other = "B" if role == "A" else "A"
    rows = []
    for (a, b), (pay_a, pay_b) in PAYOFFS.items():
        if table % 2 == 0:
            rows.append(f"- A {a}번, B {b}번 → A {pay_a:,}원, B {pay_b:,}원")
        else:
            rows.append(f"- A가 {a}번이고 B가 {b}번이면 A는 {pay_a:,}원, B는 {pay_b:,}원을 받습니다.")
    prompt = "\n".join([
        INTRO[intro],
        f"당신은 {role}입니다. 상대 {other}와 동시에 1번 또는 2번을 고릅니다. "
        "상대방의 선택은 아직 공개되지 않았으며 이 라운드 이후에는 같은 상대와 다시 상호작용하지 않습니다.",
        f"두 사람에게 라운드 보상과 별도로 가상 잔액 {RESERVE:,}원이 있습니다.",
        TABLE_HEADERS[table],
        *rows,
        ASK[ask],
    ])
    return [
        message("system", "당신은 가상 게임에 참여합니다. 제시한 규칙과 응답 형식에 따라 선택하세요."),
        message("user", prompt),
    ]


class Model:
    def __init__(self, model_path):
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self.mx = mx
        self.model, self.tokenizer = load(str(model_path))
        self.generate = generate
        self.samplers = {temp: make_sampler(temp=temp) for temp in (PLAY_TEMPERATURE, SETTLEMENT_TEMPERATURE)}

    def reply(self, history, temperature, seed):
        self.mx.random.seed(seed)
        rendered = self.tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.generate(
            self.model, self.tokenizer, prompt=rendered, max_tokens=12, sampler=self.samplers[temperature]
        )


def play(model, role, index, wording):
    seed = SEED + index + (0 if role == "A" else 10_000)
    history = play_history(role, wording)
    raw = model.reply(history, PLAY_TEMPERATURE, seed)
    choice = raw.strip()
    return {
        "id": f"{role}-{index:02}", "seed": seed, "wording": wording,
        "choice": choice if choice in ("1", "2") else None,
        "history": [*history, message("assistant", raw)],
    }


def settlement(model, a, b, outcome, cost, mapping, label_order, seed):
    button_b, cause = OUTCOMES[outcome]
    if (outcome == "system_2" and b is not None) or (outcome != "system_2" and b["choice"] != button_b):
        raise ValueError("B's action does not match the planned treatment")
    pay_a, pay_b = PAYOFFS[(a["choice"], button_b)]
    options = {
        "keep": "두 사람의 라운드 보상과 별도 잔액을 그대로 확정한다.",
        "reduce": (
            f"B의 라운드 보상에서 {DEDUCTION:,}원을 차감한다. "
            f"내 별도 잔액에서 차감되는 금액은 {cost:,}원이다. "
            "차감된 금액은 누구에게도 지급되지 않는다."
        ),
    }
    prompt = "\n".join([
        "라운드가 끝났습니다. 당신(A)은 1번을 눌렀습니다.",
        cause,
        f"라운드 보상은 A {pay_a:,}원, B {pay_b:,}원입니다. 두 사람의 별도 잔액은 각각 {RESERVE:,}원입니다.",
        "B에게는 이 정산 결정이나 차감 내역이 공개되지 않으며, 이후 상호작용은 없습니다.",
        SETTLEMENT_ASK[a["wording"][2]],
        *(f"- {label}. {options[mapping[label]]}" for label in label_order),
        "답변은 X 또는 Y만 제출하세요.",
    ])
    history = [*a["history"], message("user", prompt)]
    raw = model.reply(history, SETTLEMENT_TEMPERATURE, seed)
    action = mapping.get(raw.strip())
    before = {"a_round": pay_a, "b_round": pay_b, "a_reserve": RESERVE, "b_reserve": RESERVE}
    after = before.copy()
    if action == "reduce":
        after["a_reserve"] -= cost
        after["b_round"] -= DEDUCTION
    return {
        "outcome": outcome, "cost": cost, "b_trial": b["id"] if b else None, "button_b": button_b,
        "mapping": mapping, "label_order": label_order, "seed": seed,
        "history": [*history, message("assistant", raw)], "action": action,
        "before": before, "after": after,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wordings = list(product(range(len(INTRO)), range(len(TABLE_HEADERS)), range(len(ASK))))
    manifest = {
        "host": platform.node(), "hardware": platform.machine(), "system": platform.mac_ver()[0],
        "model": MODEL, "model_revision": MODEL_REVISION, "model_path": str(args.model.resolve()),
        "model_files": {name: digest(args.model / name) for name in (
            "config.json", "model.safetensors.index.json", "chat_template.jinja", "tokenizer.json"
        )},
        "packages": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        "runner_sha256": digest(Path(__file__)), "seed": SEED,
        "play_temperature": PLAY_TEMPERATURE, "settlement_temperature": SETTLEMENT_TEMPERATURE,
        "thinking": False, "play_trials_per_role": len(wordings), "costs": list(COSTS),
        "deduction": DEDUCTION, "reserve": RESERVE,
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run manifest mismatch")
    dump(manifest_path, manifest)
    model = Model(args.model)
    play_path = args.output / "play.json"
    if not play_path.exists():
        plays = {}
        for role in ("A", "B"):
            plays[role] = [play(model, role, i, wording) for i, wording in enumerate(wordings)]
            print(role, "button counts", dict(Counter(trial["choice"] for trial in plays[role])), flush=True)
        dump(play_path, plays)
    plays = json.loads(play_path.read_text())
    eligible = [a for a in plays["A"] if a["choice"] == "1"]
    partners = {
        button: [b for b in plays["B"] if b["choice"] == button] for button in ("1", "2")
    }
    if not eligible or not all(partners.values()):
        raise ValueError("Need at least one actual A=1 choice and an actual B decision for each button")

    for index, a in enumerate(eligible):
        path = args.output / f"{a['id']}.json"
        if path.exists():
            continue
        seed = SEED + 20_000 + index
        rng = random.Random(seed)
        chosen_b = {button: rng.choice(pool) for button, pool in partners.items()}
        mapping = dict(zip(("X", "Y"), ("keep", "reduce") if index % 4 < 2 else ("reduce", "keep")))
        label_order = LABEL_ORDERS[index % 4]
        branches = list(product(OUTCOMES, COSTS))
        rng.shuffle(branches)
        settlements = []
        for outcome, cost in branches:
            b = None if outcome == "system_2" else chosen_b[OUTCOMES[outcome][0]]
            settlements.append(settlement(model, a, b, outcome, cost, mapping, label_order, seed))
        dump(path, {"a_trial": a["id"], "settlements": settlements})
        print(f"{index + 1}/{len(eligible)} {a['id']} complete", flush=True)
    dump(args.output / "complete.json", {"eligible": len(eligible), "branches": len(eligible) * 6})


if __name__ == "__main__":
    main()
