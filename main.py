from workflows.insurance_workflow import execute_workflow
from dotenv import load_dotenv
import os
from langsmith import trace

# Load environment variables
load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")  # Organize logs under this project
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

LANGCHAIN_VERBOSE = os.getenv("LANGCHAIN_VERBOSE")
os.environ["LANGCHAIN_VERBOSE"] = LANGCHAIN_VERBOSE  # Enable verbose logging

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2  # Enable verbose logging


def main():
    print("Starting Insurance Multi-Agent System...")
    print("Type 'quit' or 'exit' to end the conversation.")

    while True:
        print("\nPlease describe your query or issue:")
        user_input = input("Input Content:").strip()
        # user_input = "This is Kevin. I bought an accident insurance plan from AIA through you a while back. A few months ago, I had a fall while traveling for work and ended up needing minor surgery on my leg. I've recovered after about three months of rest. I was wondering—can I make a claim for the medical expenses?"
        # user_input = "How to file a claim"

        if user_input.lower() in ["quit", "exit", "clear", "done", "bye"]:
            print("Exiting the chat. Goodbye!")
            break

        result = execute_workflow(user_input)  # Removed logs return

        print("Response:", result)


if __name__ == "__main__":
    main()
