NO_CONTEXT_PROMPT = """Context:
{context}

Question:
What is the temporal relation between the {source} and the {target}?

Options:
<, in case the {source} happens before the {target}
>, in case the {source} happens after the {target}
=, in case the {source} happens the same time as the {target}
-, in case the {source} happens not related to the {target}

Answer:
"""


RELATIONS2STR = {
    "<": "occurs before",
    ">": "occurs after",
    "=": "occurs at the same time",
    "-": "is not related",
}

_GENERATION_PROMPT_INSTRUCTION = """I will provide a pair of temporal entities and their relation.
Your task is to generate a context where the relation between the two entities is the one provided.
{examples}
{new_example}
"""


_EXAMPLE_TEMPLATE = """
Relation:
{source_type} of {source_text} {relation_text} {target_type} of {target_text}

Text:
{text}
"""


_OUTPUT_TEMPLATE = """Relation:
{source_type} of {source_text} {relation_text} {target_type} of {target_text}

Text:
"""


class GenerationPrompter:
    def __call__(self, new_example: dict, examples: list[dict]) -> str:
        examples_str = ""
        for example in examples:
            examples_str += _EXAMPLE_TEMPLATE.format(
                source_type=example["source_type"],
                source_text=example["source_text"],
                relation_text=RELATIONS2STR[example["label"]],
                target_type=example["target_type"],
                target_text=example["target_text"],
                text=example["text"],
            )

        new_example_str = _OUTPUT_TEMPLATE.format(
            source_type=new_example["source_type"],
            source_text=new_example["source_text"],
            relation_text=RELATIONS2STR[new_example["label"]],
            target_type=new_example["target_type"],
            target_text=new_example["target_text"],
        )

        prompt = _GENERATION_PROMPT_INSTRUCTION.format(
            examples=examples_str,
            new_example=new_example_str,
        )

        return prompt
