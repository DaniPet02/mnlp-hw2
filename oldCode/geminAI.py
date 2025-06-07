from io import StringIO
import json
import os
from pathlib import Path
from time import time
import pandas as pd
from google import genai
from google.genai import types
from oldCode.globals import BASELINE
from utils import EvaluationGride


class GeminiJudge:
    def __init__(
        self,
        TOKEN: str,
        model: str = "gemini-2.0-flash",
        contents: str = "",
        log_dir: str = "",
    ):
        self.client = genai.Client(api_key=TOKEN)
        self.model = model
        self.log = Path(log_dir)
        base = BASELINE
        self.config = types.GenerateContentConfig(
            system_instruction=base,
            max_output_tokens=500,
            response_mime_type="application/json",
            response_schema=list[EvaluationGride],
        )

        response = self.client.models.generate_content(
            model=self.model, config=self.config, contents=contents
        )

        if self.log.joinpath("gemini_log.json").exists():
            os.remove(self.log.joinpath("gemini_log.json"))
        try:
            pd.read_json(StringIO(response.text)).to_json(self.log.joinpath("gemini_log.json"))
        except (json.JSONDecodeError, ValueError) as e:
            error_log_path = self.log.joinpath(f"panic_gen{time.time():.f2}.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write("--- Parsing JSON Error ---\n")
                f.write(f"Error type: {type(e).__name__}\n")
                f.write("\n-----------------------------------------------------\n")

    def judge(self, query: str):
        response = self.client.models.generate_content(
            model=self.model, config=self.config, contents=query
        )

        try:
            pd.read_json(StringIO(response.text)).to_json(self.log.joinpath("gemini_log.json"))
        except (json.JSONDecodeError, ValueError) as e:
            error_log_path = self.log.joinpath(f"panic_gen{time.time():.f2}.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write("--- Parsing JSON Error ---\n")
                f.write(f"Error type: {type(e).__name__}\n")
                f.write("\n-----------------------------------------------------\n")

        return response.text