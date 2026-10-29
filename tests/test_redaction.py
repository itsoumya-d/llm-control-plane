from llm_control_plane.redaction import Redactor


def test_redacts_email_and_phone():
    result = Redactor().redact("Email a@b.com or call +91 70316 48617")
    assert "a@b.com" not in result.text
    assert "[REDACTED_EMAIL]" in result.text
    assert "[REDACTED_PHONE]" in result.text
    assert result.counts["email"] == 1
    assert result.counts["phone"] == 1


def test_redacts_card_iban_and_ip():
    text = "card 4111 1111 1111 1111 iban NL91ABNA0417164300 ip 192.168.1.1"
    result = Redactor().redact(text)
    assert "4111" not in result.text
    assert "NL91ABNA0417164300" not in result.text
    assert "192.168.1.1" not in result.text
    assert result.counts["credit_card"] == 1
    assert result.counts["iban"] == 1
    assert result.counts["ipv4"] == 1


def test_clean_text_is_unchanged():
    result = Redactor().redact("The audit evidence is attached.")
    assert result.text == "The audit evidence is attached."
    assert result.total == 0


def test_redact_many_aggregates_counts():
    texts, counts = Redactor().redact_many(["a@b.com", "c@d.com a@b.com"])
    assert all("@b.com" not in text and "@d.com" not in text for text in texts)
    assert counts["email"] == 3
