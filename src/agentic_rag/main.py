#!/usr/bin/env python
import os
import sys
import warnings
from pathlib import Path

# Add project root and src/ directory to sys.path so direct execution works without editable install
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
SRC_DIR = CURRENT_DIR.parent

for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Dynamic import resolution
try:
    from agentic_rag.crew import AgenticRag
except ImportError:
    try:
        from src.agentic_rag.crew import AgenticRag
    except ImportError:
        from crew import AgenticRag

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    query_text = sys.argv[1] if len(sys.argv) > 1 else 'Who is elon musk?'
    inputs = {
        'query': query_text
    }
    AgenticRag().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    query_text = sys.argv[3] if len(sys.argv) > 3 else "What is DSPy and how does it compile language model prompts?"
    inputs = {
        "query": query_text
    }
    try:
        AgenticRag().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AgenticRag().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    query_text = sys.argv[3] if len(sys.argv) > 3 else "What is DSPy and how does it compile language model prompts?"
    inputs = {
        "query": query_text
    }
    try:
        AgenticRag().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

if __name__ == "__main__":
    run()
