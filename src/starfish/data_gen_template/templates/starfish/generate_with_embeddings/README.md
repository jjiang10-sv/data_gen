# Generate with Embeddings Template

This template generates diverse synthetic data with advanced embedding-based quality control using FAISS and SentenceTransformers.

## Features

- **Semantic Similarity Checking**: Real-time similarity detection during generation
- **Advanced Deduplication**: Both exact and semantic duplicate removal
- **Diversity Metrics**: Quantitative assessment of dataset diversity
- **Quality Assurance**: Automated quality checks and reporting
- **Configurable Thresholds**: Customizable similarity and diversity parameters
- **Multiple Embedding Models**: Support for various SentenceTransformers models

## Requirements

- `faiss-cpu>=1.7.4` or `faiss-gpu` for high-performance similarity search
- `sentence-transformers>=2.2.2` for text embeddings
- OpenAI API access for text generation (or other supported LLM providers)

## Usage

```python
from starfish.data_gen_template import data_gen_template

# Get the template
template = data_gen_template.get("starfish/generate_with_embeddings")

# Configure generation
config = {
    "num_records": 50,
    "user_instruction": "Generate educational content about artificial intelligence",
    "topics": ["machine learning", "natural language processing", "computer vision"],
    "generation_model_name": "openai/gpt-4o-mini",
    "embedding_config": {
        "model_name": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.85,
        "enable_deduplication": True,
        "enable_diversity_check": True,
        "min_diversity_score": 0.3
    }
}

# Generate diverse data
results = await template.run(**config)
```

## Configuration Options

### Basic Configuration

- `num_records`: Number of data items to generate
- `user_instruction`: Overall instruction for data generation
- `topics`: Optional list of topics (auto-generated if not provided)

### Model Configuration

- `generation_model_name`: LLM model for content generation
- `generation_model_kwargs`: Additional parameters for the generation model
- `topic_model_name`: Model for automatic topic generation

### Embedding Configuration

- `model_name`: SentenceTransformers model (default: "all-MiniLM-L6-v2")
- `similarity_threshold`: Threshold for considering items similar (0-1, default: 0.85)
- `index_type`: FAISS index type ("flat", "ivf", "hnsw")
- `enable_deduplication`: Whether to perform final deduplication
- `enable_diversity_check`: Whether to check similarity during generation
- `min_diversity_score`: Minimum required diversity score
- `device`: Device for embeddings ("cpu", "cuda", "mps")

## Available Embedding Models

### Recommended Models

- `all-MiniLM-L6-v2` (default): Fast, good quality, 384 dimensions
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `all-distilroberta-v1`: Balanced speed/quality, 768 dimensions

### Specialized Models

- `paraphrase-MiniLM-L6-v2`: Optimized for paraphrase detection
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for question-answering
- `msmarco-distilbert-base-v4`: Optimized for search/retrieval

## Quality Metrics

The template provides detailed quality metrics:

### Diversity Metrics
- **Diversity Score**: Overall diversity measure (0-1, higher is better)
- **Average Similarity**: Mean pairwise similarity between items
- **Min Distance Met**: Whether minimum diversity requirements are satisfied

### Deduplication Report
- **Exact Duplicates**: Items with identical content
- **Semantic Duplicates**: Items with high semantic similarity
- **Reduction Percentage**: Percentage of items removed

### Quality Checks
- **Diversity Sufficient**: Meets minimum diversity requirements
- **Target Count Met**: Generated sufficient items (≥80% of target)
- **Low Duplication**: Semantic duplicates <10% of final dataset

## Pipeline Stages

1. **Embedding Infrastructure**: Initialize FAISS index and SentenceTransformers
2. **Topic Preparation**: Use provided topics or auto-generate them
3. **Diverse Generation**: Generate items with real-time similarity checking
4. **Deduplication**: Remove exact and semantic duplicates
5. **Quality Analysis**: Comprehensive quality assessment and reporting

## Performance Considerations

### Memory Usage
- Embedding models require ~100-500MB of memory
- FAISS index memory scales with dataset size
- Consider batch processing for very large datasets

### Speed Optimization
- Use smaller embedding models for faster processing
- Enable GPU acceleration if available
- Use "ivf" or "hnsw" index types for large datasets

### Quality vs Speed Tradeoffs
- Lower similarity thresholds = more diverse but slower generation
- Larger embedding models = better quality but more memory/time
- Real-time checking vs batch deduplication

## Example Output

Each generated item includes:

```json
{
    "question": "What are the key advantages of transformer architectures?",
    "answer": "Transformer architectures offer several key advantages...",
    "topic": "neural networks",
    "instruction_context": "Generate educational content about AI",
    "_metadata": {
        "generation_id": 0,
        "diversity_score": 0.724,
        "similarity_threshold": 0.85,
        "embedding_model": "all-MiniLM-L6-v2",
        "quality_checks": {
            "diversity_sufficient": true,
            "target_count_met": true,
            "low_duplication": true
        }
    }
}
```

## Advanced Usage

### Custom Quality Scoring
```python
def custom_quality_scorer(item):
    score = 0.0
    # Add custom quality metrics
    if len(item.get('answer', '')) > 100:
        score += 1.0
    if 'technical_term' in item.get('question', ''):
        score += 0.5
    return score

# Use in deduplicator configuration
```

### Multi-stage Processing
```python
# Generate initial dataset
initial_results = await template.run(num_records=100, enable_deduplication=False)

# Apply custom filtering
filtered_results = custom_filter(initial_results)

# Final deduplication with custom parameters
final_results = deduplicator.deduplicate_comprehensive(
    filtered_results,
    semantic_threshold=0.9,
    keep_best=True
)
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Use smaller embedding models or batch processing
2. **Slow Generation**: Disable real-time checking or use faster models
3. **Low Diversity**: Lower similarity threshold or increase max attempts
4. **Too Many Duplicates**: Increase similarity threshold or improve prompts

### Error Messages

- `"Index is empty or not initialized"`: Generate some items before similarity checking
- `"Template execution failed"`: Check LLM model availability and API keys
- `"Not enough data to train IVF index"`: Use flat index for small datasets 