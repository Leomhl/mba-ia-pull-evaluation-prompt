"""
Script to push optimised prompts to the LangSmith Prompt Hub.

This script:
1. Reads the optimised prompts from prompts/bug_to_user_story_v2.yml
2. Validates the prompts
3. Pushes them PUBLICLY to the LangSmith Hub
4. Adds metadata (tags, description, techniques used)

SIMPLIFIED: cleaner and more straightforward code.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

PROMPT_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    """
    Builds the ChatPromptTemplate from the YAML.

    The system prompt carries the instructions and the few-shot examples; the
    user prompt carries only the bug report. Keeping the variable in the user
    turn alone avoids the v1 bug, where {bug_report} appeared twice.
    """
    return ChatPromptTemplate.from_messages([
        ("system", prompt_data["system_prompt"]),
        ("human", prompt_data["user_prompt"]),
    ])


def build_description(prompt_data: dict) -> str:
    """Description published on the Hub, including the applied techniques."""
    techniques = ", ".join(prompt_data.get("techniques_applied", []))
    description = prompt_data.get("description", PROMPT_KEY)
    return f"{description} | Techniques: {techniques}" if techniques else description


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Pushes the optimised prompt to the LangSmith Hub (PUBLIC).

    Args:
        prompt_name: Prompt name
        prompt_data: Prompt data

    Returns:
        True on success, False otherwise
    """
    print(f"Publishing: {prompt_name}")

    try:
        chat_prompt = build_chat_prompt(prompt_data)
    except KeyError as e:
        print(f"   ❌ Required field missing in the YAML: {e}")
        return False

    # Tags must be plain strings; the version goes in alongside the techniques
    # so the Hub shows what was applied without opening the prompt.
    tags = list(prompt_data.get("tags", []))
    tags += [t.lower().replace(" ", "-") for t in prompt_data.get("techniques_applied", [])]
    tags.append(prompt_data.get("version", "v2"))

    try:
        url = hub.push(
            prompt_name,
            chat_prompt,
            new_repo_is_public=True,
            new_repo_description=build_description(prompt_data),
            tags=sorted(set(tags)),
        )
    except Exception as e:
        print(f"   ❌ Push failed: {e}")
        print("\nCheck that LANGSMITH_API_KEY and USERNAME_LANGSMITH_HUB are")
        print("correct in .env and that the username matches the one on the Hub.")
        return False

    print("   ✓ Published as PUBLIC")
    print(f"   ✓ {url}")
    return True


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Validates the basic structure of a prompt (simplified version).

    Args:
        prompt_data: Prompt data

    Returns:
        (is_valid, errors) - tuple with the status and the list of errors
    """
    errors = []

    for field in ("description", "system_prompt", "user_prompt", "version"):
        if not str(prompt_data.get(field, "")).strip():
            errors.append(f"Required field empty or missing: {field}")

    system_prompt = str(prompt_data.get("system_prompt", ""))
    user_prompt = str(prompt_data.get("user_prompt", ""))

    if "TODO" in system_prompt or "TODO" in user_prompt:
        errors.append("The prompt still contains [TODO] markers")

    # The dataset delivers the bug in {bug_report}; without that variable in the
    # user turn the evaluation chain would have nowhere to inject the report.
    if "{bug_report}" not in user_prompt:
        errors.append("user_prompt must contain the {bug_report} variable")

    if "{bug_report}" in system_prompt:
        errors.append("system_prompt must not repeat {bug_report} (the v1 problem)")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(f"At least 2 techniques required, found: {len(techniques)}")

    return (len(errors) == 0, errors)


def main():
    """Main function"""
    print_section_header("PUSH OPTIMISED PROMPTS")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    prompts = load_yaml(PROMPT_FILE)
    if not prompts:
        print(f"\nCreate the file {PROMPT_FILE} with the optimised prompt.")
        return 1

    prompt_data = prompts.get(PROMPT_KEY)
    if not prompt_data:
        print(f"❌ Key '{PROMPT_KEY}' not found in {PROMPT_FILE}")
        return 1

    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Invalid prompt:")
        for error in errors:
            print(f"   - {error}")
        return 1

    print("✓ Prompt validated")
    print(f"  Techniques: {', '.join(prompt_data.get('techniques_applied', []))}\n")

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    if not push_prompt_to_langsmith(f"{username}/{PROMPT_KEY}", prompt_data):
        return 1

    print("\nNext step: python src/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
