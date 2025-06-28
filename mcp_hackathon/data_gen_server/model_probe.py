from starfish import StructuredLLM, data_factory
from starfish.common.env_loader import load_env_file
from datasets import load_dataset
import json
import asyncio

load_env_file()


def run_model_probe(model_name="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", num_datapoints=10):
    # Load the dataset
    dataset = load_dataset("starfishdata/endocrinology_transcription_and_notes_and_icd_codes", split="train")
    top_n_data = dataset.select(range(num_datapoints))

    # Create a list to store the parsed data
    parsed_data = []

    # Process each entry
    for idx, entry in enumerate(top_n_data):
        # Extract transcript - get the value directly from the transcript key
        transcript = entry["transcript"] if isinstance(entry["transcript"], str) else entry["transcript"].get("transcript", "")

        # Extract ICD-10 code (top_1 code)
        icd_codes_str = entry.get("icd_10_code", "{}")
        try:
            icd_codes = json.loads(icd_codes_str)
            top_1_code = icd_codes.get("top_1", {}).get("code", "")
        except json.JSONDecodeError:
            top_1_code = ""

        # Add to parsed data
        parsed_data.append({"id": idx, "transcript": transcript, "icd_10_code": top_1_code})

    model_probe_prompt = """
    Given a transcript of a patient's medical history, determine the ICD-10 code that is most relevant to the patient's condition.
    Transcript: {{transcript}}

    Please do not return anything other than the ICD-10 code in json format.
    like this: {"icd_10_code": "A00.0"}
    """

    response_gen_llm = StructuredLLM(model_name=model_name, prompt=model_probe_prompt, output_schema=[{"name": "icd_10_code", "type": "str"}])

    @data_factory()
    async def model_probe_batch(input_data):
        response = await response_gen_llm.run(transcript=input_data["transcript"])
        return [{"id": input_data["id"], "generated_icd_10_code": response.data[0]["icd_10_code"], "actual_icd_10_code": input_data["icd_10_code"]}]

    def evaluate_model():
        data = model_probe_batch.run(input_data=parsed_data[:num_datapoints])

        # Calculate exact match accuracy
        exact_matches = sum(1 for item in data if item["generated_icd_10_code"] == item["actual_icd_10_code"])
        total_samples = len(data)
        accuracy = (exact_matches / total_samples) * 100

        return {"total_samples": total_samples, "exact_matches": exact_matches, "accuracy": accuracy}

    return evaluate_model()


if __name__ == "__main__":
    # Example usage when running this file directly
    results = run_model_probe(model_name="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", num_datapoints=5)
    print(results)
