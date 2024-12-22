"""Run a language model over the temporal games to simplify the relations."""

import asyncio
import json
import logging
import os
from pathlib import Path

import google.generativeai as genai
from tqdm.asyncio import tqdm_asyncio

from src.constants import GOOGLE_API_KEY

# Suppress google api logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


class GeminiAPI:
    QUOTA_LIMIT_PER_MINUTE = 2_000
    SLEEP_SECONDS = 60
    MODEL_NAME = "gemini-1.5-flash"
    INPUT_PRICE_M_TOKENS = 0.075
    OUTPUT_PRICE_M_TOKENS = 0.30

    def __init__(self):
        logging.info(f"Initializing model: {self.MODEL_NAME}")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.MODEL_NAME)

    def generate_content(self, prompt):
        return self.model.generate_content(prompt)

    async def _process_multiple_prompts(self, prompts):
        logging.info(f"Processing batch of {len(prompts)} prompts")
        tasks = [self.model.generate_content_async(prompt) for prompt in prompts]
        results = await tqdm_asyncio.gather(*tasks)
        return results

    async def __call__(self, prompts: list[str], cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using cache directory: {cache_dir}")

        n_input_tkns = 0
        n_output_tkns = 0
        for i in range(0, len(prompts), self.QUOTA_LIMIT_PER_MINUTE):
            cache_filepath = cache_dir / f"{i}.json"
            if cache_filepath.exists():
                logging.info(f"Skipping batch {i}, cache file exists")
                continue

            logging.info(f"Processing batch starting at index {i}")
            responses = await self._process_multiple_prompts(
                prompts[i : i + self.QUOTA_LIMIT_PER_MINUTE]
            )
            texts = [response.text if response.parts else "" for response in responses]
            n_input_tkns += sum(
                response.usage_metadata.prompt_token_count for response in responses
            )
            n_output_tkns += sum(
                response.usage_metadata.candidates_token_count for response in responses
            )

            json.dump(texts, cache_filepath.open("w"), indent=4)
            logging.info(f"Saved batch results to {cache_filepath}")

            self.log_cost(n_input_tkns, n_output_tkns)

            logging.info(f"Sleeping for {self.SLEEP_SECONDS} seconds")
            if len(texts) < self.QUOTA_LIMIT_PER_MINUTE:
                break
            else:
                await asyncio.sleep(self.SLEEP_SECONDS)

        self.log_cost(n_input_tkns, n_output_tkns)
        logging.info("Combining all cached results")
        texts = []
        for i in range(0, len(prompts), self.QUOTA_LIMIT_PER_MINUTE):
            cache_filepath = cache_dir / f"{i}.json"
            texts.extend(json.load(cache_filepath.open("r")))
        return texts

    def estimate_cost(self, prompts: list[str], as_input: bool = True):
        n_tokens = sum(
            self.model.count_tokens(prompt).total_tokens for prompt in prompts
        )
        if as_input:
            return n_tokens / 1_000_000 * self.INPUT_PRICE_M_TOKENS
        else:
            return n_tokens / 1_000_000 * self.OUTPUT_PRICE_M_TOKENS

    def log_cost(self, n_input_tkns, n_output_tkns):
        input_tkns_cost = round(n_input_tkns / 1_000_000 * self.INPUT_PRICE_M_TOKENS, 2)
        output_tkns_cost = round(
            n_output_tkns / 1_000_000 * self.OUTPUT_PRICE_M_TOKENS, 2
        )
        logging.info(f"Total input tokens: {n_input_tkns} Cost: ${input_tkns_cost}")
        logging.info(f"Total output tokens: {n_output_tkns} Cost: ${output_tkns_cost}")
        logging.info(f"Total cost: ${input_tkns_cost + output_tkns_cost}")

    def generate(self, prompt: str):
        return self.model.generate_content(prompt)
