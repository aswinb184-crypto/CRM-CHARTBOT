import unittest

from voice_input import consume_voice_question


class ConsumeVoiceQuestionTests(unittest.TestCase):
    def test_trims_a_new_transcript(self):
        self.assertEqual(
            consume_voice_question(
                {"id": "event-1", "text": "  What are the clinic hours?  "},
                None,
            ),
            ("event-1", "What are the clinic hours?"),
        )

    def test_ignores_a_repeated_event_id(self):
        self.assertEqual(
            consume_voice_question(
                {"id": "event-1", "text": "Send this once"},
                "event-1",
            ),
            ("event-1", None),
        )

    def test_consumes_empty_transcript_without_submitting_it(self):
        self.assertEqual(
            consume_voice_question(
                {"id": "event-2", "text": "   "},
                "event-1",
            ),
            ("event-2", None),
        )

    def test_ignores_malformed_events(self):
        self.assertEqual(consume_voice_question(None, "event-1"), ("event-1", None))
        self.assertEqual(
            consume_voice_question({"id": "x" * 129, "text": "Question"}, None),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
