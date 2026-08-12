"""Shared training configuration."""

DEFAULT_DROP_COLUMNS: list[str] = [
    "previous-cards",
    "client_id",
    "fact-region",
    "region",
    "registration-region",
    "tp-foreign",
    "reg-and-fact-equality",
    "post-and-fact-equality",
    "reg-and-post-equality",
    "reg-fact-post-and-last-credit-equality",
    "total-of-delinquencies",
    "max-delinquency-no",
    "mean-delinquency-amount",
    "driving-license",
    "cottage",
    "garage",
    "land",
    "reg-phone",
]
