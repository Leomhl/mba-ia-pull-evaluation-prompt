"""
Script to pull prompts from the LangSmith Prompt Hub.

This script:
1. Connects to LangSmith using the credentials from .env
2. Pulls the prompts from the Hub
3. Saves them locally in prompts/bug_to_user_story_v1.yml

SIMPLIFIED: uses LangChain's native serialisation to extract the prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# Low-quality prompt provided by the challenge.
SOURCE_PROMPT = "leonanluppi/bug_to_user_story_v1"
OUTPUT_FILE = "prompts/bug_to_user_story_v1.yml"


def extract_messages(prompt) -> dict:
    """
    Splits the messages of a ChatPromptTemplate into system and user prompt.

    The Hub returns the prompt as a LangChain object. Each message carries its
    text in .prompt.template; the class type (System/Human) says where it
    belongs. Messages without a template (e.g. MessagesPlaceholder) are ignored.
    """
    system_parts = []
    user_parts = []

    for message in getattr(prompt, "messages", []):
        template = getattr(getattr(message, "prompt", None), "template", None)
        if not template:
            continue

        if "System" in type(message).__name__:
            system_parts.append(template)
        else:
            user_parts.append(template)

    return {
        "system_prompt": "\n\n".join(system_parts),
        "user_prompt": "\n\n".join(user_parts),
    }


def pull_prompts_from_langsmith():
    """
    Pulls the initial prompt from the Hub and writes the local YAML.

    Returns:
        Dictionary of the saved prompt, or None on failure.
    """
    print(f"Pulling prompt from the LangSmith Hub: {SOURCE_PROMPT}")

    try:
        prompt = hub.pull(SOURCE_PROMPT)
    except Exception as e:
        print(f"❌ Could not pull '{SOURCE_PROMPT}': {e}")
        print("\nCheck that LANGSMITH_API_KEY is correct in .env and that the")
        print("prompt is still public on the Hub.")
        return None

    print("   ✓ Prompt loaded")

    messages = extract_messages(prompt)
    if not messages["system_prompt"] and not messages["user_prompt"]:
        print("❌ The prompt came back without any recognisable text message.")
        return None

    # Mirrors the structure the rest of the project expects in the YAML.
    prompt_name = SOURCE_PROMPT.split("/")[-1]
    prompt_data = {
        prompt_name: {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": messages["system_prompt"],
            "user_prompt": messages["user_prompt"],
            "version": "v1",
            "source": SOURCE_PROMPT,
            "tags": ["bug-analysis", "user-story", "product-management"],
        }
    }

    if not save_yaml(prompt_data, OUTPUT_FILE):
        return None

    print(f"   ✓ Saved to {OUTPUT_FILE}")
    return prompt_data


def main():
    """Main function"""
    print_section_header("PULL PROMPTS FROM THE LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    prompt_data = pull_prompts_from_langsmith()
    if prompt_data is None:
        return 1

    content = next(iter(prompt_data.values()))
    print("\nSummary of the downloaded prompt:")
    print(f"  - system_prompt: {len(content['system_prompt'])} characters")
    print(f"  - user_prompt:   {len(content['user_prompt'])} characters")

    print("\nNext step: optimise the prompt in prompts/bug_to_user_story_v2.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
