from prompt import build_user_message, PURPOSE_HINTS


def test_build_contains_purpose():
    msg = build_user_message("Patient has", "Note text.", {}, "diagnosis")
    assert "Purpose: diagnosis" in msg


def test_build_contains_prefix():
    msg = build_user_message("Patient has", "Note text.", {}, "diagnosis")
    assert '"Patient has"' in msg


def test_build_contains_full_note():
    msg = build_user_message("p", "Note text here.", {}, "medications")
    assert "Note text here." in msg


def test_build_contains_context_json():
    ctx = {"chief_complaint": "headache"}
    msg = build_user_message("p", "n", ctx, "diagnosis")
    assert '"chief_complaint": "headache"' in msg


def test_build_contains_purpose_hint_for_each_purpose():
    for purpose, hint in PURPOSE_HINTS.items():
        msg = build_user_message("p", "n", {}, purpose)
        assert hint in msg, f"Hint missing for purpose: {purpose}"


def test_build_unknown_purpose_empty_guidance():
    msg = build_user_message("p", "n", {}, "unknown_section")
    assert "Guidance: \n" in msg


def test_build_completion_label_present():
    msg = build_user_message("p", "n", {}, "procedures")
    assert "Completion:" in msg
