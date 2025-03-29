import os
import argparse
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='LLM-Based Fake News Detection')
    parser.add_argument('--task', type=str, default='evaluate',
                        choices=['preprocess', 'optimize_prompts', 'evaluate', 'all'],
                        help='Task to perform')
    parser.add_argument('--binary', action='store_true', default=True,
                        help='Use binary classification (real/fake) instead of multi-class')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        choices=['gpt-3.5-turbo', 'gpt-4'],
                        help='OpenAI model to use')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of examples to use (for optimization and evaluation)')

    args = parser.parse_args()

    # Set API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Import appropriate modules based on task
    if args.task == 'preprocess' or args.task == 'all':
        print("Running data preprocessing...")
        import data_preprocessing
        data_preprocessing.main()

    if args.task == 'optimize_prompts' or args.task == 'all':
        print("\nOptimizing prompts...")
        import prompt_engineering
        # Update sample size and model
        prompt_engineering.Config.TEST_SAMPLE_SIZE = args.sample_size
        prompt_engineering.Config.MODEL_NAME = args.model
        prompt_engineering.main()

    if args.task == 'evaluate' or args.task == 'all':
        print("\nRunning evaluation...")
        import evaluation
        # Update sample size and model
        evaluation.Config.SAMPLE_SIZE = args.sample_size
        evaluation.Config.MODEL_NAME = args.model
        evaluation.main()

    print("All tasks completed!")


if __name__ == "__main__":
    main()