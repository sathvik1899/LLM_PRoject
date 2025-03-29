import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# Configuration class from main script
class Config:
    # Paths
    DATA_DIR = "data/"
    RESULTS_DIR = "results/"
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures/")

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


def load_data(file_path):
    """Load and parse TSV files from the LIAR dataset."""
    # Column names based on the dataset description
    columns = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]

    # Read TSV file
    data = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return data


def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(data, binary=True, clean_statements=True):
    """Clean and preprocess the data."""
    # Create a copy to avoid modifying the original
    processed_data = data.copy()

    # Basic cleaning
    if clean_statements:
        processed_data['statement'] = processed_data['statement'].apply(clean_text)
    else:
        processed_data['statement'] = processed_data['statement'].str.strip()

    # Map multi-class labels to binary if needed
    if binary:
        processed_data['true_label'] = processed_data['label']
        processed_data['label'] = processed_data['label'].map(Config.BINARY_MAPPING)

    # Select relevant columns
    relevant_columns = ['id', 'label', 'statement', 'subject', 'speaker', 'context']
    if binary:
        relevant_columns.append('true_label')

    return processed_data[relevant_columns]


def analyze_label_distribution(data, title="Label Distribution", binary=True):
    """Analyze and plot the distribution of labels."""
    plt.figure(figsize=(10, 6))

    if binary:
        # For binary classification
        label_counts = data['label'].value_counts()
        sns.barplot(x=label_counts.index, y=label_counts.values)
        plt.title(f"{title} (Binary Classification)")
    else:
        # For multi-class classification
        label_counts = data['label'].value_counts()
        sns.barplot(x=label_counts.index, y=label_counts.values)
        plt.title(f"{title} (Multi-class Classification)")
        plt.xticks(rotation=45)

    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, f"label_distribution_{'binary' if binary else 'multiclass'}.png"))
    plt.close()

    return label_counts


def analyze_statement_length(data):
    """Analyze and plot the distribution of statement lengths."""
    data['statement_length'] = data['statement'].apply(lambda x: len(x.split()))

    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='statement_length', bins=50)
    plt.title("Distribution of Statement Lengths")
    plt.xlabel("Number of Words in Statement")
    plt.ylabel("Count")
    plt.axvline(data['statement_length'].mean(), color='red', linestyle='--',
                label=f"Mean: {data['statement_length'].mean():.2f}")
    plt.axvline(data['statement_length'].median(), color='green', linestyle='--',
                label=f"Median: {data['statement_length'].median():.2f}")
    plt.legend()
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "statement_length_dist.png"))
    plt.close()

    # Return statistics
    length_stats = {
        "mean": data['statement_length'].mean(),
        "median": data['statement_length'].median(),
        "min": data['statement_length'].min(),
        "max": data['statement_length'].max(),
        "95th_percentile": data['statement_length'].quantile(0.95)
    }

    return length_stats


def analyze_common_words(data, n=20):
    """Analyze and plot the most common words in statements."""
    stop_words = set(stopwords.words('english'))

    all_words = []
    for statement in data['statement']:
        words = word_tokenize(statement)
        # Filter out stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        all_words.extend(filtered_words)

    word_counts = Counter(all_words)
    most_common = word_counts.most_common(n)

    plt.figure(figsize=(12, 8))
    words, counts = zip(*most_common)
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {n} Most Common Words in Statements")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "common_words.png"))
    plt.close()

    return most_common


def analyze_subjects(data):
    """Analyze the distribution of subjects."""
    # Split multiple subjects (they might be separated by semicolons)
    all_subjects = []
    for subjects in data['subject']:
        if pd.isna(subjects):
            continue
        subjects_list = subjects.split(';')
        all_subjects.extend([s.strip() for s in subjects_list])

    subject_counts = Counter(all_subjects)
    most_common = subject_counts.most_common(15)  # Top 15 subjects

    plt.figure(figsize=(12, 8))
    subjects, counts = zip(*most_common)
    sns.barplot(x=list(counts), y=list(subjects))
    plt.title("Top 15 Most Common Subjects")
    plt.xlabel("Count")
    plt.ylabel("Subject")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "common_subjects.png"))
    plt.close()

    return most_common


def analyze_speakers(data):
    """Analyze the distribution of speakers."""
    speaker_counts = data['speaker'].value_counts().head(15)  # Top 15 speakers

    plt.figure(figsize=(12, 8))
    sns.barplot(x=speaker_counts.values, y=speaker_counts.index)
    plt.title("Top 15 Most Common Speakers")
    plt.xlabel("Count")
    plt.ylabel("Speaker")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "common_speakers.png"))
    plt.close()

    return speaker_counts


def generate_summary_report(train_data, test_data, valid_data):
    """Generate a summary report of the dataset."""
    summary = {
        "dataset_size": {
            "train": len(train_data),
            "test": len(test_data),
            "valid": len(valid_data),
            "total": len(train_data) + len(test_data) + len(valid_data)
        }
    }

    # Save summary to file
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(Config.RESULTS_DIR, "dataset_summary.txt"), 'w') as f:
        f.write("LIAR Dataset Summary\n")
        f.write("===================\n\n")
        f.write(f"Total examples: {summary['dataset_size']['total']}\n")
        f.write(f"Training set: {summary['dataset_size']['train']} examples\n")
        f.write(f"Test set: {summary['dataset_size']['test']} examples\n")
        f.write(f"Validation set: {summary['dataset_size']['valid']} examples\n")

    return summary


def main():
    # Create necessary directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    train_data = load_data(Config.TRAIN_FILE)
    test_data = load_data(Config.TEST_FILE)
    valid_data = load_data(Config.VALID_FILE)

    print("Preprocessing data...")
    processed_train = preprocess_data(train_data, binary=True, clean_statements=False)
    processed_test = preprocess_data(test_data, binary=True, clean_statements=False)
    processed_valid = preprocess_data(valid_data, binary=True, clean_statements=False)

    print("Generating dataset summary...")
    summary = generate_summary_report(train_data, test_data, valid_data)

    print("Analyzing label distribution...")
    binary_label_counts = analyze_label_distribution(processed_train, "Training Set Label Distribution", binary=True)
    multiclass_label_counts = analyze_label_distribution(train_data, "Training Set Label Distribution", binary=False)

    print("Analyzing statement lengths...")
    length_stats = analyze_statement_length(processed_train)

    print("Analyzing common words...")
    common_words = analyze_common_words(processed_train)

    print("Analyzing subjects...")
    common_subjects = analyze_subjects(processed_train)

    print("Analyzing speakers...")
    common_speakers = analyze_speakers(processed_train)

    print("Analysis complete!")

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {summary['dataset_size']['total']}")
    print(f"Binary label distribution: {dict(binary_label_counts)}")
    print(f"Average statement length: {length_stats['mean']:.2f} words")
    print(f"Most common word: {common_words[0][0]} (appears {common_words[0][1]} times)")
    print(f"Most common subject: {common_subjects[0][0]} (appears {common_subjects[0][1]} times)")
    print(f"Most common speaker: {common_speakers.index[0]} (appears {common_speakers.values[0]} times)")


if __name__ == "__main__":
    main()