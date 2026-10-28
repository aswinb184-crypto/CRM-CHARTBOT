def consume_voice_question(
    event: object,
    last_event_id: str | None,
) -> tuple[str | None, str | None]:
    if not isinstance(event, dict):
        return last_event_id, None

    event_id = event.get("id")
    if (
        not isinstance(event_id, str)
        or not event_id
        or len(event_id) > 128
        or event_id == last_event_id
    ):
        return last_event_id, None

    transcript = event.get("text")
    question = transcript.strip() if isinstance(transcript, str) else ""
    return event_id, question or None
