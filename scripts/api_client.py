#!/usr/bin/env python3
"""
Simple client to test the RAG API server.
"""

import requests
import argparse
import json


def ask_question(server_url: str, question: str, include_docs: bool = False):
    """Ask a single question."""
    url = f"{server_url}/ask"

    payload = {
        "question": question,
        "include_docs": include_docs
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()

        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print("-"*80)
        print(f"Answer: {result['answer']}")

        if include_docs and 'retrieved_documents' in result:
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result['retrieved_documents'], 1):
                print(f"\n  {i}. [Score: {doc['score']:.4f}]")
                print(f"     {doc['text'][:200]}...")

        print("="*80)

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def batch_questions(server_url: str, questions: list, include_docs: bool = False):
    """Ask multiple questions at once."""
    url = f"{server_url}/batch"

    payload = {
        "questions": questions,
        "include_docs": include_docs
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()

        print("\n" + "="*80)
        print(f"Batch Results: {result['successful']}/{result['total']} successful")
        print("="*80)

        for i, res in enumerate(result['results'], 1):
            print(f"\n{i}. Q: {res['question']}")
            print(f"   A: {res['answer']}")

        print("="*80)

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def check_health(server_url: str):
    """Check if server is healthy."""
    url = f"{server_url}/health"

    try:
        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        print("\nServer Health:")
        print(json.dumps(result, indent=2))

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def get_info(server_url: str):
    """Get server/model information."""
    url = f"{server_url}/info"

    try:
        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        print("\nServer Info:")
        print(json.dumps(result, indent=2))

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def interactive_mode(server_url: str):
    """Interactive question-answering mode."""
    print("\n" + "="*80)
    print("RAG API Client - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - Type 'docs' to toggle showing retrieved documents")
    print("  - Type 'info' to see server information")
    print("  - Type 'health' to check server health")
    print("  - Type 'quit' or 'exit' to stop")
    print("="*80)

    show_docs = False

    while True:
        try:
            user_input = input("\n\nYour question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'docs':
                show_docs = not show_docs
                print(f"Document display: {'ON' if show_docs else 'OFF'}")
                continue

            if user_input.lower() == 'info':
                get_info(server_url)
                continue

            if user_input.lower() == 'health':
                check_health(server_url)
                continue

            ask_question(server_url, user_input, include_docs=show_docs)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Client for RAG API server"
    )
    parser.add_argument("--server", type=str, default="http://localhost:5000",
                       help="API server URL")
    parser.add_argument("--question", type=str, default=None,
                       help="Single question to ask")
    parser.add_argument("--questions", type=str, nargs='+', default=None,
                       help="Multiple questions to ask")
    parser.add_argument("--include_docs", action="store_true",
                       help="Include retrieved documents in response")
    parser.add_argument("--health", action="store_true",
                       help="Check server health")
    parser.add_argument("--info", action="store_true",
                       help="Get server info")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive mode")

    args = parser.parse_args()

    if args.health:
        check_health(args.server)
    elif args.info:
        get_info(args.server)
    elif args.question:
        ask_question(args.server, args.question, args.include_docs)
    elif args.questions:
        batch_questions(args.server, args.questions, args.include_docs)
    else:
        # Default to interactive mode
        interactive_mode(args.server)


if __name__ == "__main__":
    main()
