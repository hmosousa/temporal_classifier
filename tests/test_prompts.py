from src.prompts import GenerationPrompter


class TestPrompts:
    def test_call(self):
        prompter = GenerationPrompter()
        example = {
            "source_text": "built",
            "source_type": "start",
            "target_text": "opened",
            "target_type": "end",
            "relation": "<",
        }
        examples = [
            {
                "text": "The building was built in 2024 and opened in 2025.",
                "source_text": "built",
                "source_type": "start",
                "target_text": "opened",
                "target_type": "end",
                "relation": "<",
            },
            {
                "text": "The men run after he stopped.",
                "source_text": "stopped",
                "source_type": "end",
                "target_text": "run",
                "target_type": "start",
                "relation": ">",
            },
        ]
        prompt = prompter(example, examples)
        print(prompt)
