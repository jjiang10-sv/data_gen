"""Data Generation Template with Embedding-based Deduplication

This template generates diverse synthetic data with real-time similarity checking
and deduplication using FAISS and SentenceTransformers embeddings.
"""

from starfish import data_gen_template
from starfish.embedding import EmbeddingManager, SimilarityChecker, DataDeduplicator
from starfish.components.prepare_topic import generate_topics
from pydantic import BaseModel

from typing import Optional, Dict, Any, List
import random

from starfish.common.logger import get_logger

logger = get_logger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding-based processing."""

    model_name: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85
    index_type: str = "flat"
    enable_deduplication: bool = True
    enable_diversity_check: bool = True
    min_diversity_score: float = 0.3
    device: Optional[str] = None


## Pydantic Input Schema
class GenerateWithEmbeddings(BaseModel):
    """
    Input schema for the generate_with_embeddings template.

    This template generates diverse synthetic data with embedding-based
    similarity checking and deduplication.
    """

    num_records: Optional[int] = 10
    user_instruction: str
    topics: Optional[List[str]] = None

    # Model Configuration
    generation_model_name: str = "openai/gpt-4o-mini"
    generation_model_kwargs: Optional[Dict[str, Any]] = None
    topic_model_name: str = "openai/gpt-4o-mini"
    topic_model_kwargs: Optional[Dict[str, Any]] = None

    # Embedding Configuration
    embedding_config: EmbeddingConfig = EmbeddingConfig()

    # Data Factory Configuration
    data_factory_config: Optional[Dict[str, Any]] = {}


@data_gen_template.register(
    name="starfish/generate_with_embeddings",
    input_schema=GenerateWithEmbeddings,
    output_schema=None,
    description="""Generates diverse synthetic data with embedding-based similarity checking.
                   Features real-time deduplication, diversity metrics, and quality assurance
                   using FAISS and SentenceTransformers embeddings.""",
    author="Starfish AI",
    starfish_version="0.1.3",
    dependencies=["faiss-cpu>=1.7.4", "sentence-transformers>=2.2.2"],
    input_example="""{
        "num_records": 20,
        "user_instruction": "Generate questions and answers about machine learning concepts",
        "topics": ["supervised learning", "neural networks", "feature engineering"],
        "generation_model_name": "openai/gpt-4o-mini",
        "generation_model_kwargs": {"temperature": 0.7, "max_tokens": 300},
        "embedding_config": {
            "model_name": "all-MiniLM-L6-v2",
            "similarity_threshold": 0.85,
            "enable_deduplication": true,
            "enable_diversity_check": true,
            "min_diversity_score": 0.3
        },
        "data_factory_config": {"max_concurrency": 8, "task_runner_timeout": 120}
    }""",
)
async def generate_with_embeddings_workflow(input_data: GenerateWithEmbeddings):
    """
    Main workflow for generating data with embedding-based quality control.
    """
    print("🔮 Embedding-Enhanced Data Generation Pipeline")
    print("=" * 60)
    print("📋 Process Overview:")
    print("   1. Initialize embedding infrastructure")
    print("   2. Generate or validate topics")
    print("   3. Generate synthetic data with diversity checking")
    print("   4. Perform embedding-based deduplication")
    print("   5. Analyze final dataset quality")
    print("=" * 60)

    # Initialize embedding infrastructure
    print("🧠 Step 1: Initializing embedding infrastructure...")
    embedding_manager = EmbeddingManager(
        model_name=input_data.embedding_config.model_name,
        similarity_threshold=input_data.embedding_config.similarity_threshold,
        index_type=input_data.embedding_config.index_type,
        device=input_data.embedding_config.device,
    )

    similarity_checker = SimilarityChecker(embedding_manager=embedding_manager, similarity_threshold=input_data.embedding_config.similarity_threshold)

    deduplicator = DataDeduplicator(embedding_manager=embedding_manager, similarity_threshold=input_data.embedding_config.similarity_threshold)
    print(f"   ✅ Embedding system initialized with {input_data.embedding_config.model_name}")
    print("")

    # Generate or validate topics
    print("🎯 Step 2: Preparing topics...")
    if input_data.topics:
        topics = input_data.topics
        print(f"   ✅ Using provided {len(topics)} topics")
    else:
        # Auto-generate topics
        num_topics = min(10, max(3, input_data.num_records // 3))
        topics = await generate_topics(
            user_instruction=input_data.user_instruction,
            num_topics=num_topics,
            model_name=input_data.topic_model_name,
            model_kwargs=input_data.topic_model_kwargs,
        )
        print(f"   ✅ Generated {len(topics)} topics automatically")
    print("")

    # Generate synthetic data with real-time diversity checking
    print("💫 Step 3: Generating synthetic data with diversity checking...")
    generated_items = []
    rejected_count = 0
    generation_attempts = 0
    max_attempts = input_data.num_records * 3  # Allow up to 3x attempts

    while len(generated_items) < input_data.num_records and generation_attempts < max_attempts:
        generation_attempts += 1

        # Select a random topic for this generation
        current_topic = random.choice(topics)

        # Generate a single item
        new_item = await _generate_single_item(
            topic=current_topic,
            user_instruction=input_data.user_instruction,
            model_name=input_data.generation_model_name,
            model_kwargs=input_data.generation_model_kwargs,
        )

        # Check similarity to existing items if enabled
        if input_data.embedding_config.enable_diversity_check and generated_items:
            is_similar, similar_info = similarity_checker.is_similar_to_existing(new_item, generated_items)

            if is_similar:
                rejected_count += 1
                logger.debug(f"Rejected similar item (similarity: {similar_info['similarity']:.3f})")
                continue

        # Add to generated items
        generated_items.append(new_item)

        # Log progress
        if len(generated_items) % 5 == 0:
            print(f"   📊 Generated {len(generated_items)}/{input_data.num_records} items (rejected {rejected_count} similar)")

    print(f"   ✅ Generated {len(generated_items)} items with {rejected_count} rejections")
    print("")

    # Perform final deduplication if enabled
    if input_data.embedding_config.enable_deduplication:
        print("🔍 Step 4: Performing embedding-based deduplication...")
        final_items, dedup_report = deduplicator.deduplicate_comprehensive(generated_items, exact_first=True, keep_best=True)
        print(f"   ✅ Deduplication complete: {dedup_report['original_count']} -> {dedup_report['final_count']} items")
        print(f"   📉 Removed {dedup_report['total_removed']} duplicates ({dedup_report['reduction_percentage']:.1f}% reduction)")
    else:
        final_items = generated_items
        print("⏭️  Step 4: Skipping deduplication (disabled)")
    print("")

    # Analyze final dataset quality
    print("📊 Step 5: Analyzing final dataset quality...")
    if final_items:
        diversity_metrics = similarity_checker.check_diversity_batch(final_items)
        duplicate_analysis = deduplicator.analyze_duplicates(final_items)

        print(f"   📈 Diversity Score: {diversity_metrics['diversity_score']:.3f}")
        print(f"   🔄 Average Similarity: {diversity_metrics['avg_similarity']:.3f}")
        print(f"   📏 Min Distance Met: {diversity_metrics.get('meets_min_distance', 'N/A')}")
        print(f"   🔢 Final Count: {len(final_items)} items")

        # Check if quality thresholds are met
        quality_check = {
            "diversity_sufficient": diversity_metrics["diversity_score"] >= input_data.embedding_config.min_diversity_score,
            "target_count_met": len(final_items) >= input_data.num_records * 0.8,  # Allow 20% shortfall
            "low_duplication": duplicate_analysis["semantic_duplicates"]["percentage"] < 10,
        }

        all_checks_passed = all(quality_check.values())
        print(f"   ✅ Quality Check: {'PASSED' if all_checks_passed else 'NEEDS ATTENTION'}")

        # Add metadata to items
        for i, item in enumerate(final_items):
            if isinstance(item, dict):
                item["_metadata"] = {
                    "generation_id": i,
                    "diversity_score": diversity_metrics["diversity_score"],
                    "similarity_threshold": input_data.embedding_config.similarity_threshold,
                    "embedding_model": input_data.embedding_config.model_name,
                    "quality_checks": quality_check,
                }
    else:
        print("   ⚠️  No items generated - check your configuration")

    print("=" * 60)
    print(f"🎉 Pipeline Complete! Generated {len(final_items)} high-quality diverse items")

    return final_items


async def _generate_single_item(topic: str, user_instruction: str, model_name: str, model_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a single data item for the given topic.

    This is a simplified example - you would replace this with your actual
    generation logic using your LLM infrastructure.
    """
    from starfish.llm.structured_llm import StructuredLLM

    # Create a structured LLM instance
    llm = StructuredLLM(model=model_name, **(model_kwargs or {}))

    # Generate a question-answer pair for the topic
    prompt = f"""Generate a high-quality question and answer pair about: {topic}

Context: {user_instruction}

Requirements:
- Question should be specific and educational
- Answer should be comprehensive but concise
- Both should be related to the topic: {topic}

Please respond with a JSON object containing 'question' and 'answer' fields."""

    try:
        # Generate using the LLM
        result = await llm.generate(prompt)

        # Parse the result (this is simplified - you might need more robust parsing)
        if isinstance(result, dict) and "question" in result and "answer" in result:
            generated_item = {"question": result["question"], "answer": result["answer"], "topic": topic, "instruction_context": user_instruction}
        else:
            # Fallback format
            generated_item = {
                "question": f"What are the key aspects of {topic}?",
                "answer": f"This is a generated answer about {topic} in the context of {user_instruction}",
                "topic": topic,
                "instruction_context": user_instruction,
            }
    except Exception as e:
        logger.warning(f"Generation failed for topic '{topic}': {e}")
        # Fallback item
        generated_item = {
            "question": f"What are the key aspects of {topic}?",
            "answer": f"This is a fallback answer about {topic}",
            "topic": topic,
            "instruction_context": user_instruction,
            "generation_error": str(e),
        }

    return generated_item
