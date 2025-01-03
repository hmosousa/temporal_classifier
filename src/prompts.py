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


_GENERATION_PROMPT_INSTRUCTION = """You will receive a pair of temporal entities and their temporal relation. 
Your task is to generate a coherent and natural-sounding text that accurately reflects the provided temporal relation and explicitly has the two entities. 

The possible relations are:
- "occurs before": The start/end of the first entity happens before the start/end of the second entity.
- "occurs after": The start/end of the first entity happens after the start/end of the second entity.
- "occurs at the same time": The start/end of the first entity coincides with the start/end of the second entity.
- "is not related": There is no temporal connection between the start/end of the first entity and the start/end of the second entity.

When generating the text:
- Use language that sounds natural and fluent, adapting temporal verbs as needed.
- Do not ask for more context, just generate the text.
{examples}
{new_example}
"""


_EXAMPLE_TEMPLATE = """
Input:
<{source_tag}>{source_text}</{source_tag}> {relation_text} <{target_tag}>{target_text}</{target_tag}>

Output:
{text}
"""


_OUTPUT_TEMPLATE = """Input:
<{source_tag}>{source_text}</{source_tag}> {relation_text} <{target_tag}>{target_text}</{target_tag}>

Output:
"""


class GenerationPrompter:
    def __call__(self, new_example: dict, examples: list[dict]) -> str:
        examples_str = ""
        for example in examples:
            examples_str += _EXAMPLE_TEMPLATE.format(
                source_tag=f"{example['source_type']}_source",
                source_text=example["source_text"],
                relation_text=RELATIONS2STR[example["label"]],
                target_tag=f"{example['target_type']}_target",
                target_text=example["target_text"],
                text=example["text"],
            )

        new_example_str = _OUTPUT_TEMPLATE.format(
            source_tag=f"{new_example['source_type']}_source",
            source_text=new_example["source_text"],
            relation_text=RELATIONS2STR[new_example["label"]],
            target_tag=f"{new_example['target_type']}_target",
            target_text=new_example["target_text"],
        )

        prompt = _GENERATION_PROMPT_INSTRUCTION.format(
            examples=examples_str,
            new_example=new_example_str,
        )

        return prompt
