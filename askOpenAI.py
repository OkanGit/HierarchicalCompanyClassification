
"""Simple script to run LLM inference on a DataFrame column using a specified prompt defined in the OpenAI Playground."""
import asyncio
import nest_asyncio
import pandas as pd
import time

from openai import AsyncOpenAI
client = AsyncOpenAI()

# Patch event loop for Jupyter
nest_asyncio.apply()

# -----------------------------------
# Simple string grader
# -----------------------------------
def grade_string_match(prediction: str, truth: str) -> float:
    if prediction is None or truth is None:
        return 0.0
    return float(prediction.strip().lower() == truth.strip().lower())


# -----------------------------------
# Single API call
# -----------------------------------
async def call_prompt_id(row, prompt_id: str):
    # print(row["text"])
    # print(prompt_id)
    try:
        response = await client.responses.create(
            model="gpt-5",
            prompt={
                "id": prompt_id,
                "variables":{
                    "text": row["cleaned_text"],
                },
            },
            # input={"text": row["text"]}
        )
        return response.output_text
    except Exception as e:
        return f"ERROR: {e}"


# -----------------------------------
# Batch processor
# -----------------------------------
async def _process_dataframe_async(df, prompt_id, batch_size, batch_pause):
    results = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]

        tasks = [
            call_prompt_id(row, prompt_id)
            for _, row in batch.iterrows()
        ]

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        time.sleep(batch_pause)

    return results


# -----------------------------------
# Jupyter-safe user function
# -----------------------------------
def run_inference(df: pd.DataFrame, prompt_id: str,
                  batch_size=5, batch_pause=0.3) -> pd.DataFrame:
    """
    Notebook-safe inference function.
    Works even when an event loop is already running.
    """
    loop = asyncio.get_event_loop()

    preds = loop.run_until_complete(
        _process_dataframe_async(df, prompt_id, batch_size, batch_pause)
    )

    df_out = df.copy()
    df_out["prediction"] = preds
    df_out["score"] = [
        grade_string_match(pred, truth)
        for pred, truth in zip(df_out["prediction"], df_out["truth"])
    ]

    return df_out
