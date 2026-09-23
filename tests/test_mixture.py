import random

from decision_model.data.mixture import pick_question, prepare_example, subsample_question
from decision_model.data.schema import DecisionExample, Option, Question


def make_large(n_options=50):
    options = [Option(id=f"o{i}", text=f"option {i}") for i in range(n_options)]
    return Question(
        text="pick one",
        options=options,
        answer_ids=["o7"],
        type="choice",
    )


def test_subsample_keeps_gold_and_caps():
    rng = random.Random(0)
    for _ in range(50):
        q = subsample_question(make_large(50), rng, max_options=10)
        assert len(q.options) <= 10
        assert "o7" in q.option_ids()
        assert q.answer_ids == ["o7"]
        assert len(set(q.option_ids())) == len(q.options)


def test_subsample_noop_when_small():
    rng = random.Random(0)
    options = [Option(id=f"o{i}", text=f"option {i}") for i in range(4)]
    q = Question(options=options, answer_ids=["o1"], type="choice")
    out = subsample_question(q, rng, max_options=10)
    assert out is q


def test_subsample_respects_min_options():
    rng = random.Random(0)
    options = [Option(id=f"o{i}", text=f"option {i}") for i in range(30)]
    q = Question(options=options, answer_ids=["o3"], type="choice")
    out = subsample_question(q, rng, min_options=6, max_options=4)
    assert len(out.options) >= 6


def test_pick_question_single():
    rng = random.Random(0)
    q1 = make_large(5)
    q2 = Question(
        options=[Option(id="a", text="a"), Option(id="b", text="b")],
        answer_ids=["a"],
        type="choice",
    )
    ex = DecisionExample(task="t", state="s", questions=[q1, q2])
    out = pick_question(ex, rng)
    assert len(out.questions) == 1
    assert out.questions[0] in (q1, q2)


def test_batched_length_bucketing():
    from decision_model.data.batching import batched

    def mk(n: int, length: int, task: str):
        return DecisionExample(
            task=task,
            state="x" * length,
            questions=[
                Question(
                    options=[Option(id="a", text="a"), Option(id="b", text="b")],
                    answer_ids=["a"],
                    type="choice",
                )
            ],
        )

    examples = [mk(i, 10 if i % 2 else 500, f"t{i%3}") for i in range(24)]
    batches = list(batched(iter(examples), size=4, bucket=True, buffer_mult=2))
    seen = [ex for b in batches for ex in b]
    assert len(seen) == len(examples)
    assert sorted(id(e) for e in seen) == sorted(id(e) for e in examples)
    for b in batches:
        lens = [len(e.state) for e in b]
        assert lens == sorted(lens), lens


def test_prepare_example():
    rng = random.Random(0)
    ex = DecisionExample(task="t", state="s", questions=[make_large(100)])
    out = prepare_example(ex, rng, min_options=2, max_options=8)
    assert len(out.questions) == 1
    assert len(out.questions[0].options) <= 8
    assert "o7" in out.questions[0].option_ids()
