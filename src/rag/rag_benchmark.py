import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric
from deepeval.models import OllamaModel
from deepeval.evaluate import AsyncConfig, DisplayConfig
import pandas as pd

print("Loading outputs...", end="")
with open("output/rag/out.json", "r") as f:
    outputs = json.load(f)
print("✅\n")

print("Loading Ollama model...", end="")
model = OllamaModel(
    model="gemma4:31b",
    base_url="http://localhost:11434",
    temperature=0,
)
print("✅\n")

metrics = [
    FaithfulnessMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    ),
    ContextualPrecisionMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )
]

async_config = AsyncConfig(
    run_async=False
)

display_config = DisplayConfig(
    verbose_mode=True,
    results_folder="./output/rag"
)

text_splitter_strategies = ["fixed", "recursive", "semantic"]
results = {}

for strategy in text_splitter_strategies:
    print(f"Evaluating for {strategy} text splitting strategy:\n\n")
    samples = outputs[strategy]
    
    test_cases: list[LLMTestCase] = []
    
    print("Generating Samples...", end="")
    for sample in samples:
        
        test_case = LLMTestCase(
            input = sample["input"],
            actual_output = sample["actual_output"],
            expected_output = sample["expected_output"],
            retrieval_context = sample["retrieval_context"],
        )
        
        test_cases.append(test_case)
        # break
    print("✅\n")
    ev = evaluate(test_cases=test_cases[:5], metrics=metrics, async_config=async_config, display_config=display_config)
    
    results[strategy] = ev


# Build a summary table from EvaluationResult -> TestResult -> MetricData
metric_name_map = {
    "faithfulness": "faithfulness",
    "contextual precision": "context precision",
}

table_data = {}
for strategy, evaluation_result in results.items():
    metric_scores: dict[str, list[float]] = {label: [] for label in metric_name_map.values()}

    for test_result in evaluation_result.test_results:
        for metric_data in test_result.metrics_data:
            normalized_name = metric_data.name.strip().lower()
            if normalized_name in metric_name_map:
                row_label = metric_name_map[normalized_name]
                metric_scores[row_label].append(metric_data.score)

    # Mean score per metric for this strategy
    table_data[strategy] = {
        metric_label: (sum(scores) / len(scores) if scores else float("nan"))
        for metric_label, scores in metric_scores.items()
    }

summary_df = pd.DataFrame(table_data)
summary_df = summary_df.reindex(["faithfulness", "context precision"])

print("\nFinal RAG benchmark summary:\n")
print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

