from sloth.factcheck import check_entry


def test_factcheck_supported_sentence_passes() -> None:
    facts = "time: 09:00; place: Lisbon; caption: coffee by the river"
    text = "In the morning I had coffee by the river in Lisbon."
    result = check_entry(text, facts)
    assert result.ok is True
    assert result.unsupported_words == []


def test_factcheck_invented_object_and_number_fail() -> None:
    facts = "time: 09:00; place: Lisbon; caption: coffee by the river"
    text = "I bought 3 bicycles in Lisbon."
    result = check_entry(text, facts)
    assert result.ok is False
    assert "bicycles" in result.unsupported_words
    assert "3" in result.unsupported_words
