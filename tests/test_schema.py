import pytest

from decision_model.data.schema import (
    DecisionExample,
    Option,
    Question,
    validate,
)


def make_example():
    return DecisionExample(
        task="sst2",
        state="a wonderful movie",
        questions=[
            Question(
                text="What is the sentiment?",
                options=[Option(id="negative", text="negative"), Option(id="positive", text="positive")],
                answer_ids=["positive"],
                type="choice",
            )
        ],
    )


def test_roundtrip_json():
    ex = make_example()
    ex2 = DecisionExample.from_json(ex.to_json())
    assert ex2 == ex


def test_validate_ok():
    assert validate(make_example()) == []


def test_validate_missing_answer_option():
    ex = make_example()
    ex.questions[0].answer_ids = ["nope"]
    errors = validate(ex)
    assert any("not in options" in e for e in errors)


def test_validate_choice_needs_single_answer():
    ex = make_example()
    ex.questions[0].answer_ids = ["negative", "positive"]
    errors = validate(ex)
    assert any("exactly one answer" in e for e in errors)


def test_validate_duplicate_option_ids():
    ex = make_example()
    ex.questions[0].options.append(Option(id="positive", text="dup"))
    errors = validate(ex)
    assert any("duplicate" in e for e in errors)


def test_option_render():
    o = Option(id="billing", text="billing", description="payments, invoices")
    assert o.render() == "billing: payments, invoices"
    o2 = Option(id="billing", text="billing")
    assert o2.render() == "billing"
