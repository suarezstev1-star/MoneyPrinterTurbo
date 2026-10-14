from app.utils.utils import redact_secrets_in_text


def test_redacts_urls_keys_and_extra_secrets():
    text = (
        "401 from https://user:hunter2@api.example.com/v1?api_key=abc123def456&x=1 "
        "Authorization: Bearer eyJabcdefgh.ijklmnop.qrstuvw sk-ABCDEFGHIJKLMNOP1234 "
        "custom-secret-value"
    )
    out = redact_secrets_in_text(text, extra_secrets=["custom-secret-value"])
    for leaked in (
        "hunter2",
        "abc123def456",
        "ijklmnop",
        "ABCDEFGHIJKLMNOP",
        "custom-secret-value",
    ):
        assert leaked not in out
    assert "x=1" in out
    assert "api.example.com" in out


def test_plain_text_and_none_are_safe():
    assert (
        redact_secrets_in_text("connection reset by peer")
        == "connection reset by peer"
    )
    assert redact_secrets_in_text(None) == ""
