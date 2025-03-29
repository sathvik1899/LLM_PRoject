import os
import pandas as pd
import numpy as np
import time
import json
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
class Config:
    # Paths
    DATA_DIR = "data/"
    RESULTS_DIR = "results/"

    # Dataset files
    TRAIN_FILE = os.path.join(DATA_DIR, "train.tsv")
    TEST_FILE = os.path.join(DATA_DIR, "test.tsv")
    VALID_FILE = os.path.join(DATA_DIR, "valid.tsv")

    # Labels
    LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    BINARY_MAPPING = {
        "pants-fire": "fake",
        "false": "fake",
        "barely-true": "fake",
        "half-true": "real",
        "mostly-true": "real",
        "true": "real"
    }

    # Model settings
    MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4"
    TEMPERATURE = 0.0  # Lower temperature for more deterministic outputs

    # Test sample size for prompt testing
    TEST_SAMPLE_SIZE = 50


# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")  # Will automatically use OPENAI_API_KEY from environment


def load_data(file_path):
    """Load and parse TSV files."""
    columns = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]

    data = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return data


def preprocess_data(data, binary=True):
    """Preprocess the data."""
    processed_data = data.copy()

    processed_data['statement'] = processed_data['statement'].str.strip()

    if binary:
        processed_data['true_label'] = processed_data['label']
        processed_data['label'] = processed_data['label'].map(Config.BINARY_MAPPING)

    relevant_columns = ['id', 'label', 'statement', 'subject', 'speaker', 'context']
    if binary:
        relevant_columns.append('true_label')

    return processed_data[relevant_columns]


# Define different prompt strategies
prompt_strategies = {
    "basic": {
        "binary": {
            "system": "You are a fact-checker. Classify the statement as either 'real' or 'fake'. Reply with just one word.",
            "user": "Statement: \"{statement}\"\nClassify as 'real' or 'fake':"
        },
        "multiclass": {
            "system": "You are a fact-checker. Classify the statement on this scale: pants-fire, false, barely-true, half-true, mostly-true, true. Reply with just one category.",
            "user": "Statement: \"{statement}\"\nClassify on the truth scale:"
        }
    },

    "detailed": {
        "binary": {
            "system": """You are a highly accurate fake news detector. Your task is to classify statements as either "real" or "fake".
- Classify as "fake" if the statement appears to be false, misleading, or exaggerated.
- Classify as "real" if the statement appears to be factually accurate.
Respond with only one word: either "real" or "fake".""",
            "user": """Statement: "{statement}"
Speaker: {speaker}
Context: {context}
Subject: {subject}

Classify the above statement as "real" or "fake"."""
        },
        "multiclass": {
            "system": """You are a highly accurate fact-checking system. Classify political statements on this scale:
- "pants-fire": Completely false statement
- "false": Not accurate statement
- "barely-true": Contains element of truth but ignores critical facts
- "half-true": Partially accurate but leaves out important details
- "mostly-true": Accurate but needs clarification
- "true": Completely accurate statement
Respond with only one of these six labels.""",
            "user": """Statement: "{statement}"
Speaker: {speaker}
Context: {context}
Subject: {subject}

Classify on the truth scale: pants-fire, false, barely-true, half-true, mostly-true, or true."""
        }
    },

    "efficient": {
        "binary": {
            "system": "Classify: real or fake? One word only.",
            "user": "\"{statement}\""
        },
        "multiclass": {
            "system": "Classify: pants-fire, false, barely-true, half-true, mostly-true, or true? One word only.",
            "user": "\"{statement}\""
        }
    },

    "few_shot": {
        "binary": {
            "system": "You are a fact-checker. Classify statements as 'real' or 'fake'. Follow the examples.",
            "user": """Examples:
Statement: "The U.S. has a $12 billion trade deficit with Canada."
Classification: fake

Statement: "The unemployment rate for African Americans is at the lowest rate recorded."
Classification: real

Statement: "{statement}"
Classification:"""
        },
        "multiclass": {
            "system": "You are a fact-checker. Classify statements on this scale: pants-fire, false, barely-true, half-true, mostly-true, true. Follow the examples.",
            "user": """Examples:
Statement: "The U.S. has a $12 billion trade deficit with Canada."
Classification: false

Statement: "The unemployment rate for African Americans is at the lowest rate recorded."
Classification: mostly-true

Statement: "Says his tax plan would not benefit him."
Classification: pants-fire

Statement: "{statement}"
Classification:"""
        }
    }
}


def format_prompt(row, strategy_name, binary=True):
    """Format a prompt based on the specified strategy."""
    strategy = prompt_strategies[strategy_name]
    mode = "binary" if binary else "multiclass"

    system_template = strategy[mode]["system"]
    user_template = strategy[mode]["user"]

    # Fill in the templates
    statement = row['statement']
    speaker = row['speaker'] if not pd.isna(row['speaker']) else "unknown speaker"
    context = row['context'] if not pd.isna(row['context']) else "unknown context"
    subject = row['subject'] if not pd.isna(row['subject']) else "general topic"

    user_prompt = user_template.format(
        statement=statement,
        speaker=speaker,
        context=context,
        subject=subject
    )

    return system_template, user_prompt


def classify_with_llm(system_prompt, user_prompt):
    """Send a prompt to the LLM API and return the result and metrics."""
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=Config.TEMPERATURE,
            max_tokens=10  # Keep this small for efficiency - we only need one word or category
        )

        prediction = response.choices[0].message.content.strip().lower()
        processing_time = time.time() - start_time
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return prediction, token_usage, processing_time

    except Exception as e:
        print(f"Error in API call: {e}")
        return None, None, None


def evaluate_prompt_strategy(data, strategy_name, binary=True, sample_size=None):
    """Evaluate a prompt strategy on the dataset."""
    if sample_size and len(data) > sample_size:
        eval_data = data.sample(sample_size, random_state=42)
    else:
        eval_data = data

    results = []
    true_labels = []
    predicted_labels = []

    total_tokens = 0
    total_time = 0

    for _, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc=f"Evaluating {strategy_name}"):
        system_prompt, user_prompt = format_prompt(row, strategy_name, binary)
        prediction, token_usage, processing_time = classify_with_llm(system_prompt, user_prompt)

        if prediction and token_usage:
            true_label = row['label']

            # Normalize prediction to match expected labels
            if binary:
                if prediction not in ["real", "fake"]:
                    # Simple heuristic for non-exact matches
                    prediction = "real" if any(t in prediction for t in ["real", "true", "accurate"]) else "fake"
            else:
                # For multiclass, try to match to the closest label
                if prediction not in Config.LABELS:
                    closest = min(Config.LABELS, key=lambda x: abs(len(x) - len(prediction)))
                    prediction = closest

            results.append({
                "id": row['id'],
                "statement": row['statement'],
                "true_label": true_label,
                "predicted_label": prediction,
                "tokens": token_usage,
                "processing_time": processing_time
            })

            true_labels.append(true_label)
            predicted_labels.append(prediction)

            total_tokens += token_usage["total_tokens"]
            total_time += processing_time

    # Calculate metrics
    if binary:
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels, pos_label="real", average="binary"),
            "recall": recall_score(true_labels, predicted_labels, pos_label="real", average="binary"),
            "f1": f1_score(true_labels, predicted_labels, pos_label="real", average="binary")
        }
    else:
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels, average="weighted"),
            "recall": recall_score(true_labels, predicted_labels, average="weighted"),
            "f1": f1_score(true_labels, predicted_labels, average="weighted")
        }

    # Calculate efficiency metrics
    efficiency = {
        "avg_tokens_per_statement": total_tokens / len(results),
        "avg_processing_time": total_time / len(results),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "estimated_cost": (total_tokens / 1000) * 0.002  # Assuming $0.002 per 1K tokens
    }

    return results, metrics, efficiency


def compare_strategies(data, binary=True, sample_size=Config.TEST_SAMPLE_SIZE):
    """Compare different prompt strategies."""
    comparison = {}

    for strategy_name in prompt_strategies.keys():
        print(f"Evaluating strategy: {strategy_name}")
        results, metrics, efficiency = evaluate_prompt_strategy(
            data, strategy_name, binary, sample_size
        )

        comparison[strategy_name] = {
            "metrics": metrics,
            "efficiency": efficiency
        }

    # Create comparison visualizations
    strategies = list(comparison.keys())
    accuracies = [comparison[s]["metrics"]["accuracy"] for s in strategies]
    f1_scores = [comparison[s]["metrics"]["f1"] for s in strategies]
    token_usage = [comparison[s]["efficiency"]["avg_tokens_per_statement"] for s in strategies]

    # Accuracy and F1 comparison
    plt.figure(figsize=(12, 6))

    x = np.arange(len(strategies))
    width = 0.35

    plt.bar(x - width / 2, accuracies, width, label='Accuracy')
    plt.bar(x + width / 2, f1_scores, width, label='F1 Score')

    plt.xlabel('Prompt Strategy')
    plt.ylabel('Score')
    plt.title('Accuracy and F1 Score by Prompt Strategy')
    plt.xticks(x, strategies)
    plt.legend()
    plt.tight_layout()

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"prompt_comparison_{'binary' if binary else 'multiclass'}.png"))
    plt.close()

    # Token usage comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=strategies, y=token_usage)
    plt.xlabel('Prompt Strategy')
    plt.ylabel('Average Tokens per Statement')
    plt.title('Token Usage by Prompt Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"token_usage_{'binary' if binary else 'multiclass'}.png"))
    plt.close()

    # Save comparison data
    with open(os.path.join(Config.RESULTS_DIR, f"prompt_comparison_{'binary' if binary else 'multiclass'}.json"),
              'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


def main():
    # Create directories if they don't exist
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    print("Loading validation data...")
    valid_data = load_data(Config.VALID_FILE)

    print("Preprocessing data...")
    processed_valid = preprocess_data(valid_data, binary=True)

    print("Comparing prompt strategies for binary classification...")
    binary_comparison = compare_strategies(processed_valid, binary=True)

    # Optional: Also compare for multi-class
    # print("Comparing prompt strategies for multi-class classification...")
    # processed_valid_multiclass = preprocess_data(valid_data, binary=False)
    # multiclass_comparison = compare_strategies(processed_valid_multiclass, binary=False)

    # Output best strategy
    best_strategy = max(binary_comparison.keys(), key=lambda s: binary_comparison[s]["metrics"]["f1"])
    best_accuracy = binary_comparison[best_strategy]["metrics"]["accuracy"]
    best_f1 = binary_comparison[best_strategy]["metrics"]["f1"]
    best_token_usage = binary_comparison[best_strategy]["efficiency"]["avg_tokens_per_statement"]
    best_cost = binary_comparison[best_strategy]["efficiency"]["estimated_cost"]

    print("\nBest Prompt Strategy Results:")
    print(f"Strategy: {best_strategy}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"Average Tokens: {best_token_usage:.1f}")
    print(f"Estimated Cost: ${best_cost:.4f}")

    # Save the best strategy for use in the main script
    with open(os.path.join(Config.RESULTS_DIR, "best_strategy.json"), 'w') as f:
        json.dump({
            "strategy_name": best_strategy,
            "binary": binary_comparison[best_strategy],
            # "multiclass": multiclass_comparison[best_strategy] if 'multiclass_comparison' in locals() else None
        }, f, indent=2)


if __name__ == "__main__":
    main()