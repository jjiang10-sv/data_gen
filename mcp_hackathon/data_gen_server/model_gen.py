from starfish import data_factory
from starfish.common.env_loader import load_env_file
from datasets import load_dataset
import json
import asyncio
import os
import random
from agents import Agent, Runner, function_tool, ModelSettings
from agents.tool import WebSearchTool
from pydantic import BaseModel, Field

load_env_file()


class DiagnosisSuggestion(BaseModel):
    code: str = Field(..., description="The suggested diagnosis code (e.g., ICD-10)")
    confidence: float = Field(..., description="Model confidence in the suggestion, between 0 and 1")
    reason: str = Field(..., description="Explanation or rationale for the suggested diagnosis")


async def run_model_gen(num_datapoints, model_name="openai/gpt-4o-mini"):
    # Get HF token from environment
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    # Load the dataset
    dataset = load_dataset("starfishdata/playground_endocronology_notes_1500", split="train", token=hf_token)

    # Get total number of samples
    total_samples = len(dataset)

    # Generate random indices
    random_indices = random.sample(range(total_samples), num_datapoints)

    # Create list of dictionaries with only transcript key
    transcript_list = [{"transcript": dataset[idx]["transcript"]} for idx in random_indices]

    # Create the Agent
    diagnosis_code_agent = Agent(
        name="Diagnosis Code Agent",
        tools=[WebSearchTool()],
        model=model_name,
        output_type=DiagnosisSuggestion,
        model_settings=ModelSettings(tool_choice="required"),
        tool_use_behavior="stop_on_first_tool",
        instructions="""
        You are an Endocrinology Medical Coding Specialist.
        You will be provided with a medical transcript describing a patient encounter.
        Your task is to analyze the medical transcript and assign the most appropriate diagnosis code(s).
        You will have access to a web search tool and only use it to search endocrinology related code and verification.
        Use it only to verify the accuracy or current validity of the diagnosis codes.
        """,
    )

    web_search_prompt = """Please select top 3 likely code from given list for this doctor and patient conversation transcript.
        Transcript: {transcript}
    """

    @data_factory(max_concurrency=100, task_runner_timeout=300)
    async def generate_data(transcript):
        diagnosis_code_result = await Runner.run(diagnosis_code_agent, input=web_search_prompt.format(transcript=transcript))

        code_result = diagnosis_code_result.final_output.model_dump()

        return [{"transcript": transcript, "icd_10_code": code_result["code"]}]

    return generate_data.run(transcript_list)


if __name__ == "__main__":
    # Run the async function
    results = asyncio.run(run_model_gen())
    print(len(results))
    print(results[0].keys())
