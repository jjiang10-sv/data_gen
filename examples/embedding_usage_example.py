"""Example: Using Starfish Embeddings for Data Generation

This example demonstrates how to use FAISS and SentenceTransformers
for embedding-enhanced data generation and deduplication.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from starfish.embedding import EmbeddingManager, SimilarityChecker, DataDeduplicator
from starfish.data_gen_template.core import data_gen_template


async def basic_embedding_example():
    """Basic example of using the embedding system."""
    print("🔮 Basic Embedding Example")
    print("=" * 50)

    # Initialize embedding manager
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2", similarity_threshold=0.85)

    # Sample texts to embed
    texts = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "What are neural networks?",
        "Explain deep learning concepts",
        "What is supervised learning?",
        "What is machine learning?",  # Duplicate
        "How do neural networks function?",  # Similar to "What are neural networks?"
    ]

    print(f"📝 Processing {len(texts)} sample texts...")

    # Add texts to the index
    indices = embedding_manager.add_texts(texts)
    print(f"✅ Added {len(indices)} texts to the embedding index")

    # Search for similar texts
    query = "Tell me about AI and ML"
    similar_items = embedding_manager.search_similar(query, k=3)

    print(f"\n🔍 Search results for: '{query}'")
    for item in similar_items:
        print(f"   Similarity: {item['similarity']:.3f} | Text: {item['text']}")

    # Find duplicates
    duplicate_groups = embedding_manager.find_duplicates(texts)
    print(f"\n🔄 Found {len(duplicate_groups)} groups of duplicates:")
    for i, group in enumerate(duplicate_groups):
        print(f"   Group {i+1}: {[texts[idx] for idx in group]}")

    print(f"\n📊 Index Stats: {embedding_manager.get_stats()}")


async def similarity_checker_example():
    """Example of using the similarity checker."""
    print("\n🎯 Similarity Checker Example")
    print("=" * 50)

    similarity_checker = SimilarityChecker(similarity_threshold=0.8)

    # Sample data items
    data_items = [
        {"question": "What is Python?", "answer": "Python is a programming language"},
        {"question": "How to learn coding?", "answer": "Start with basic concepts"},
        {"question": "What is programming?", "answer": "Programming is writing code"},
        {"question": "What is Python programming?", "answer": "Python is a popular language"},  # Similar to first
    ]

    print(f"📝 Analyzing {len(data_items)} data items...")

    # Filter similar items
    filtered_items, duplicate_groups = similarity_checker.filter_similar_items(data_items)
    print(f"✅ Filtered to {len(filtered_items)} unique items")

    # Check diversity metrics
    diversity_metrics = similarity_checker.check_diversity_batch(data_items)
    print(f"📈 Diversity Score: {diversity_metrics['diversity_score']:.3f}")
    print(f"🔄 Average Similarity: {diversity_metrics['avg_similarity']:.3f}")

    # Suggest diverse subset
    diverse_subset = similarity_checker.suggest_diverse_subset(data_items, target_size=2)
    print(f"\n🎲 Diverse subset (2 items):")
    for item in diverse_subset:
        print(f"   Q: {item['question']}")


async def deduplicator_example():
    """Example of using the data deduplicator."""
    print("\n🔧 Data Deduplicator Example")
    print("=" * 50)

    deduplicator = DataDeduplicator(similarity_threshold=0.9)

    # Sample dataset with duplicates
    dataset = [
        {"id": "1", "text": "Machine learning is a subset of AI", "quality_score": 0.8},
        {"id": "2", "text": "Deep learning uses neural networks", "quality_score": 0.9},
        {"id": "1", "text": "Machine learning is a subset of AI", "quality_score": 0.7},  # Exact duplicate
        {"id": "3", "text": "ML is part of artificial intelligence", "quality_score": 0.95},  # Semantic duplicate
        {"id": "4", "text": "Natural language processing handles text", "quality_score": 0.85},
    ]

    print(f"📝 Analyzing dataset with {len(dataset)} items...")

    # Analyze duplicates without removing
    analysis = deduplicator.analyze_duplicates(dataset)
    print(f"🔍 Analysis Results:")
    print(f"   Exact duplicates: {analysis['exact_duplicates']['count']}")
    print(f"   Semantic duplicates: {analysis['semantic_duplicates']['count']}")
    print(f"   Diversity score: {analysis['diversity_metrics']['diversity_score']:.3f}")

    # Perform comprehensive deduplication
    clean_dataset, report = deduplicator.deduplicate_comprehensive(dataset)
    print(f"\n✨ Deduplication Results:")
    print(f"   Original: {report['original_count']} items")
    print(f"   Final: {report['final_count']} items")
    print(f"   Reduction: {report['reduction_percentage']:.1f}%")

    print("\n📋 Clean dataset:")
    for item in clean_dataset:
        print(f"   ID: {item['id']} | Score: {item.get('quality_score', 'N/A')} | Text: {item['text'][:50]}...")


async def template_usage_example():
    """Example of using the embedding-enhanced template."""
    print("\n🚀 Embedding-Enhanced Template Example")
    print("=" * 50)

    try:
        # Get the embedding template
        print(data_gen_template.list())
        template = data_gen_template.get("starfish/generate_with_embeddings")

        # Configuration for generation
        config = {
            "num_records": 5,  # Small number for demo
            "user_instruction": "Generate educational Q&A about data science",
            "topics": ["statistics", "data visualization", "machine learning"],
            "generation_model_name": "openai/gpt-4o-mini",
            "embedding_config": {
                "model_name": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.8,
                "enable_deduplication": True,
                "enable_diversity_check": True,
                "min_diversity_score": 0.2,
            },
        }

        print("⚙️  Generating diverse dataset with embedding quality control...")
        results = await template.run(**config)

        print(f"\n✅ Generated {len(results)} high-quality items:")
        for i, item in enumerate(results[:3]):  # Show first 3
            print(f"\n   Item {i+1}:")
            print(f"   Q: {item.get('question', 'N/A')}")
            print(f"   A: {item.get('answer', 'N/A')[:100]}...")
            if "_metadata" in item:
                print(f"   Diversity: {item['_metadata'].get('diversity_score', 'N/A'):.3f}")

    except Exception as e:
        print(f"⚠️  Template example failed: {e}")
        print("   (This might be due to missing API keys or dependencies)")


async def main():
    """Run all examples."""
    print("🎉 Starfish Embedding System Examples")
    print("=" * 60)

    try:
        await basic_embedding_example()
        await similarity_checker_example()
        await deduplicator_example()
        await template_usage_example()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Install dependencies: poetry install")
        print("   2. Set API keys in .env.local")
        print("   3. Try the embedding template in your projects")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies:")
        print("   poetry install")
        print("   # or")
        print("   pip install faiss-cpu sentence-transformers")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Check your Python environment and dependencies")


if __name__ == "__main__":
    asyncio.run(main())
