"""M9a feature schema hash tracks YAML feature list."""

from stock_transformer.features.universe_tensor import DEFAULT_UNIVERSE_FEATURE_NAMES, feature_schema


def test_feature_order_changes_hash():
    a = feature_schema(["a", "b"])
    b = feature_schema(["b", "a"])
    assert a["hash"] != b["hash"]


def test_default_features_stable():
    s = feature_schema(list(DEFAULT_UNIVERSE_FEATURE_NAMES))
    assert s["n"] == len(DEFAULT_UNIVERSE_FEATURE_NAMES)
    assert len(s["hash"]) == 16
