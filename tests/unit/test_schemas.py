import math

import pytest
from pydantic import ValidationError

from fraudguard.api.schemas import BatchRequest, Transaction


def test_valid_transaction(tx):
    t = Transaction(**tx)
    assert t.Amount == 42.5 and t.transaction_id == "tx-001"


def test_int_is_accepted_as_float(tx):
    assert Transaction(**{**tx, "Amount": 10}).Amount == 10.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("Amount", -0.01),
        ("Amount", 2_000_000),
        ("Amount", "10.0"),
        ("V1", 999.0),
        ("V2", math.nan),
        ("V3", math.inf),
        ("Time", -1),
        ("transaction_id", "x" * 65),
        ("transaction_id", "tem espaço"),
        ("V4", None),
        ("Amount", True),
    ],
)
def test_invalid_values_rejected(tx, field, value):
    with pytest.raises(ValidationError):
        Transaction(**{**tx, field: value})


def test_missing_and_extra_fields_rejected(tx):
    missing = dict(tx)
    missing.pop("V28")
    with pytest.raises(ValidationError):
        Transaction(**missing)
    with pytest.raises(ValidationError):
        Transaction(**tx, customer_name="Fulano")


def test_batch_requires_items():
    with pytest.raises(ValidationError):
        BatchRequest(transactions=[])
