from emotions.suite.config_merge import deep_merge_with_null, merge_many


def test_deep_merge_with_null_removes_keys_and_merges_nested_dicts() -> None:
    base = {
        "a": 1,
        "nested": {
            "x": 10,
            "y": 20,
            "z": {"k": "keep", "remove": "yes"},
        },
        "remove_me": 5,
    }
    override = {
        "nested": {
            "y": 99,
            "z": {"remove": None, "added": "ok"},
        },
        "remove_me": None,
    }

    merged = deep_merge_with_null(base, override)

    assert merged["a"] == 1
    assert merged["nested"]["x"] == 10
    assert merged["nested"]["y"] == 99
    assert merged["nested"]["z"]["k"] == "keep"
    assert "remove" not in merged["nested"]["z"]
    assert merged["nested"]["z"]["added"] == "ok"
    assert "remove_me" not in merged


def test_merge_many_applies_overrides_in_order() -> None:
    base = {"v": 1, "nested": {"a": 1, "b": 2}}
    o1 = {"nested": {"a": 10}}
    o2 = {"nested": {"b": None, "c": 30}, "v": 2}

    merged = merge_many(base, o1, o2)

    assert merged == {"v": 2, "nested": {"a": 10, "c": 30}}
