import os
import pandas as pd
import numpy as np
import json
import time
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Constants
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

    # Evaluation settings
    SAMPLE_SIZE = 200  # Set to None for full dataset evaluation

    # Model settings
    MODEL_NAME = "gpt-3.5-turbo"
    TEMPERATURE = 0.0

    # Strategy settings - will be loaded from best_strategy.json if available
    PROMPT_STRATEGY = "detailed"  # Default if no best strategy found


# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")

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


def load_best_strategy():
    """Load the best prompt strategy if available."""
    strategy_file = os.path.join(Config.RESULTS_DIR, "best_strategy.json")

    if os.path.exists(strategy_file):
        with open(strategy_file, 'r') as f:
            strategy_data = json.load(f)

        return strategy_data["strategy_name"]
    else:
        return Config.PROMPT_STRATEGY


# Define prompt strategies - same as in prompt_engineering.py
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
            max_tokens=10
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


def run_evaluation(data, strategy_name, binary=True, sample_size=None):
    """Run a full evaluation using the specified strategy."""
    if sample_size and len(data) > sample_size:
        eval_data = data.sample(sample_size, random_state=42)
    else:
        eval_data = data

    results = []
    true_labels = []
    predicted_labels = []

    total_tokens = 0
    total_time = 0

    for _, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Evaluating"):
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
            "f1": f1_score(true_labels, predicted_labels, pos_label="real", average="binary"),
            "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=["fake", "real"]).tolist()
        }
    else:
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels, average="weighted"),
            "recall": recall_score(true_labels, predicted_labels, average="weighted"),
            "f1": f1_score(true_labels, predicted_labels, average="weighted"),
            "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=Config.LABELS).tolist()
        }

    # Calculate efficiency metrics
    efficiency = {
        "avg_tokens_per_statement": total_tokens / len(results),
        "avg_processing_time": total_time / len(results),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "estimated_cost": (total_tokens / 1000) * 0.002  # Assuming $0.002 per 1K tokens for GPT-3.5-turbo
    }

    return results, metrics, efficiency


def plot_confusion_matrix(cm, labels, title, output_path):
    """Plot and save a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_errors(results_data):
    """Analyze error patterns in the results."""
    df = pd.DataFrame(results_data)

    # Filter to incorrect predictions
    errors = df[df['true_label'] != df['predicted_label']]

    # Group by true label
    error_by_true_label = errors.groupby('true_label').size().reset_index(name='count')

    # Group by predicted label
    error_by_pred_label = errors.groupby('predicted_label').size().reset_index(name='count')

    # Plot error distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x='true_label', y='count', data=error_by_true_label)
    plt.title('Errors by True Label')
    plt.xlabel('True Label')
    plt.ylabel('Error Count')

    plt.subplot(1, 2, 2)
    sns.barplot(x='predicted_label', y='count', data=error_by_pred_label)
    plt.title('Errors by Predicted Label')
    plt.xlabel('Predicted Label')
    plt.ylabel('Error Count')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "error_analysis.png"))
    plt.close()

    # Find examples of each error type (for binary classification)
    error_types = {}
    if 'fake' in df['true_label'].unique() and 'real' in df['true_label'].unique():
        # False positives (fake classified as real)
        false_positives = errors[(errors['true_label'] == 'fake') & (errors['predicted_label'] == 'real')]
        error_types['false_positives'] = false_positives.head(5).to_dict('records')

        # False negatives (real classified as fake)
        false_negatives = errors[(errors['true_label'] == 'real') & (errors['predicted_label'] == 'fake')]
        error_types['false_negatives'] = false_negatives.head(5).to_dict('records')

    return {
        "error_counts": {
            "by_true_label": error_by_true_label.to_dict('records'),
            "by_predicted_label": error_by_pred_label.to_dict('records')
        },
        "error_examples": error_types
    }


def analyze_performance_by_features(results_data):
    """Analyze performance based on different features."""
    df = pd.DataFrame(results_data)
    df['correct'] = df['true_label'] == df['predicted_label']

    # Statement length analysis
    df['statement_length'] = df['statement'].apply(lambda x: len(x.split()))
    df['length_bin'] = pd.cut(df['statement_length'],
                              bins=[0, 10, 20, 30, 40, 50, float('inf')],
                              labels=['1-10', '11-20', '21-30', '31-40', '41-50', '50+'])

    length_accuracy = df.groupby('length_bin')['correct'].mean().reset_index()
    length_count = df.groupby('length_bin').size().reset_index(name='count')

    # Plot length vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(length_accuracy['length_bin'], length_accuracy['correct'])
    plt.xlabel('Statement Length (words)')
    plt.ylabel('Classification Accuracy')
    plt.title('Accuracy by Statement Length')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "accuracy_by_length.png"))
    plt.close()

    # Analyze by speaker (if available)
    if 'speaker' in df.columns:
        # Count speakers with at least 5 statements
        speaker_counts = df['speaker'].value_counts()
        common_speakers = speaker_counts[speaker_counts >= 5].index

        if len(common_speakers) > 0:
            common_speakers_df = df[df['speaker'].isin(common_speakers)]
            speaker_accuracy = common_speakers_df.groupby('speaker')['correct'].mean().reset_index()
            speaker_accuracy = speaker_accuracy.sort_values('correct', ascending=False)

            # Plot top and bottom 10 speakers by accuracy
            plt.figure(figsize=(12, 8))
            top_speakers = speaker_accuracy.head(10)
            bottom_speakers = speaker_accuracy.tail(10)

            plt.subplot(1, 2, 1)
            sns.barplot(x='correct', y='speaker', data=top_speakers)
            plt.title('Top 10 Speakers by Accuracy')
            plt.xlabel('Accuracy')

            plt.subplot(1, 2, 2)
            sns.barplot(x='correct', y='speaker', data=bottom_speakers)
            plt.title('Bottom 10 Speakers by Accuracy')
            plt.xlabel('Accuracy')

            plt.tight_layout()
            plt.savefig(os.path.join(Config.RESULTS_DIR, "accuracy_by_speaker.png"))
            plt.close()

    return {
        "length_analysis": {
            "accuracy": length_accuracy.to_dict('records'),
            "counts": length_count.to_dict('records')
        }
    }


def compute_cross_dataset_performance(best_strategy_name, binary=True):
    """Evaluate performance across different datasets."""
    print("Computing cross-dataset performance...")

    # Load datasets
    train_data = load_data(Config.TRAIN_FILE)
    test_data = load_data(Config.TEST_FILE)
    valid_data = load_data(Config.VALID_FILE)

    # Preprocess datasets
    processed_train = preprocess_data(train_data, binary)
    processed_test = preprocess_data(test_data, binary)
    processed_valid = preprocess_data(valid_data, binary)

    # Sample from each dataset for evaluation
    sample_size = Config.SAMPLE_SIZE or 100  # Use a smaller sample if none specified

    # Evaluate on each dataset
    train_results, train_metrics, _ = run_evaluation(
        processed_train, best_strategy_name, binary, sample_size
    )
    test_results, test_metrics, _ = run_evaluation(
        processed_test, best_strategy_name, binary, sample_size
    )
    valid_results, valid_metrics, _ = run_evaluation(
        processed_valid, best_strategy_name, binary, sample_size
    )

    # Compile results
    cross_dataset_results = {
        "train": {
            "accuracy": train_metrics["accuracy"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "f1": train_metrics["f1"]
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"]
        },
        "valid": {
            "accuracy": valid_metrics["accuracy"],
            "precision": valid_metrics["precision"],
            "recall": valid_metrics["recall"],
            "f1": valid_metrics["f1"]
        }
    }

    # Visualize cross-dataset performance
    datasets = ["train", "test", "valid"]
    accuracy = [cross_dataset_results[d]["accuracy"] for d in datasets]
    precision = [cross_dataset_results[d]["precision"] for d in datasets]
    recall = [cross_dataset_results[d]["recall"] for d in datasets]
    f1 = [cross_dataset_results[d]["f1"] for d in datasets]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.2

    plt.bar(x - width * 1.5, accuracy, width, label='Accuracy')
    plt.bar(x - width / 2, precision, width, label='Precision')
    plt.bar(x + width / 2, recall, width, label='Recall')
    plt.bar(x + width * 1.5, f1, width, label='F1')

    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title('Performance Metrics Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "cross_dataset_performance.png"))
    plt.close()

    return cross_dataset_results


def generate_final_report(metrics, efficiency, error_analysis, feature_analysis, cross_dataset=None):
    """Generate a comprehensive evaluation report."""
    report = {
        "metrics": metrics,
        "efficiency": efficiency,
        "error_analysis": error_analysis,
        "feature_analysis": feature_analysis
    }

    if cross_dataset:
        report["cross_dataset_performance"] = cross_dataset

    # Save full report to JSON
    with open(os.path.join(Config.RESULTS_DIR, "evaluation_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # Generate a text summary
    with open(os.path.join(Config.RESULTS_DIR, "evaluation_summary.txt"), 'w') as f:
        f.write("FAKE NEWS DETECTION - EVALUATION SUMMARY\n")
        f.write("======================================\n\n")

        f.write("CLASSIFICATION METRICS\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n\n")

        f.write("EFFICIENCY METRICS\n")
        f.write(f"Average tokens per statement: {efficiency['avg_tokens_per_statement']:.2f}\n")
        f.write(f"Average processing time: {efficiency['avg_processing_time']:.2f} seconds\n")
        f.write(f"Total tokens used: {efficiency['total_tokens']}\n")
        f.write(f"Estimated cost: ${efficiency['estimated_cost']:.4f}\n\n")

        # Add cross-dataset performance if available
        if cross_dataset:
            f.write("CROSS-DATASET PERFORMANCE\n")
            for dataset in cross_dataset:
                f.write(f"{dataset.upper()} - F1: {cross_dataset[dataset]['f1']:.4f}, ")
                f.write(f"Accuracy: {cross_dataset[dataset]['accuracy']:.4f}\n")
            f.write("\n")

        f.write("ERROR ANALYSIS\n")
        for label, count in zip(
                [e["true_label"] for e in error_analysis["error_counts"]["by_true_label"]],
                [e["count"] for e in error_analysis["error_counts"]["by_true_label"]]
        ):
            f.write(f"Errors for true label '{label}': {count}\n")

        f.write("\nFor more details, see the full JSON report and visualization files.")

    return report


def main():
    # Create directories if they don't exist
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # Load the best strategy if available
    best_strategy = load_best_strategy()
    print(f"Using prompt strategy: {best_strategy}")

    # Load and preprocess test data
    print("Loading test data...")
    test_data = load_data(Config.TEST_FILE)
    processed_test = preprocess_data(test_data)

    # Run evaluation
    print("Running evaluation...")
    results, metrics, efficiency = run_evaluation(
        processed_test, best_strategy, binary=True, sample_size=Config.SAMPLE_SIZE
    )

    # Plot confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, ["fake", "real"], "Test Set Confusion Matrix",
        os.path.join(Config.RESULTS_DIR, "confusion_matrix.png")
    )

    # Analyze errors
    print("Analyzing errors...")
    error_analysis = analyze_errors(results)

    # Analyze performance by features
    print("Analyzing performance by features...")
    feature_analysis = analyze_performance_by_features(results)

    # Optional: Compute cross-dataset performance
    cross_dataset = None
    # Uncommenting below will evaluate on all datasets, which can be expensive
    # cross_dataset = compute_cross_dataset_performance(best_strategy)

    # Generate final report
    print("Generating final report...")
    report = generate_final_report(metrics, efficiency, error_analysis, feature_analysis, cross_dataset)

    # Print summary results
    print("\nEvaluation Results Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average tokens per statement: {efficiency['avg_tokens_per_statement']:.2f}")
    print(f"Estimated cost: ${efficiency['estimated_cost']:.4f}")

    print("\nEvaluation complete! Results saved to:", Config.RESULTS_DIR)


if __name__ == "__main__":
    main()
