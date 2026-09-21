"""
Automated tests for prompt validation.
"""
import pytest
import yaml
import re
import sys
from pathlib import Path

# Add src to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Loads prompts from the YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt():
    """Optimised prompt (v2) loaded from the YAML."""
    prompts = load_prompts(PROMPT_FILE)
    assert prompts is not None, f"Could not load {PROMPT_FILE}"
    assert PROMPT_KEY in prompts, f"Key '{PROMPT_KEY}' missing in {PROMPT_FILE}"
    return prompts[PROMPT_KEY]


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt):
        """Checks that the 'system_prompt' field exists and is not empty."""
        assert "system_prompt" in prompt, "Field 'system_prompt' does not exist"

        system_prompt = prompt["system_prompt"]
        assert isinstance(system_prompt, str), "'system_prompt' must be a string"
        assert system_prompt.strip(), "'system_prompt' is empty"

        # A system prompt that is too short carries no instructions, format or examples.
        assert len(system_prompt) > 500, (
            f"'system_prompt' has only {len(system_prompt)} characters; "
            "a detailed prompt was expected"
        )

    def test_prompt_has_role_definition(self, prompt):
        """Checks that the prompt defines a persona (e.g. "Você é um Product Manager")."""
        system_prompt = prompt["system_prompt"]

        # The prompt itself is written in Portuguese, so the expected wording is too.
        assert re.search(r"Você é um[ao]?\s+\w+", system_prompt), (
            "The prompt does not define a persona in the 'Você é um...' form"
        )

        # The persona must be a concrete role, not a generic "assistant" as in v1.
        roles = ["product manager", "product owner", "analista", "engenheiro"]
        found = [r for r in roles if r in system_prompt.lower()]
        assert found, (
            f"Persona is too generic; one of these roles was expected: {roles}"
        )

    def test_prompt_mentions_format(self, prompt):
        """Checks that the prompt requires Markdown or the standard User Story format."""
        system_prompt = prompt["system_prompt"].lower()

        assert "user story" in system_prompt, "The prompt does not mention 'User Story'"

        # Standard user story template.
        for part in ["como um", "eu quero", "para que"]:
            assert part in system_prompt, (
                f"The prompt does not require the '{part}' part of the User Story template"
            )

        assert "critérios de aceitação" in system_prompt, (
            "The prompt does not require the Acceptance Criteria section"
        )

        # Gherkin format, so the criteria are testable.
        for keyword in ["dado", "quando", "então"]:
            assert keyword in system_prompt, (
                f"The prompt does not require the Dado/Quando/Então format (missing '{keyword}')"
            )

    def test_prompt_has_few_shot_examples(self, prompt):
        """Checks that the prompt contains input/output examples (Few-shot technique)."""
        system_prompt = prompt["system_prompt"]
        lower = system_prompt.lower()

        assert "exemplo" in lower, "The prompt contains no example at all"

        # Few-shot requires input/output pairs, not just the word "exemplo".
        assert "entrada:" in lower, "The examples do not mark the Input"
        assert "saída:" in lower, "The examples do not mark the Output"

        examples = re.findall(r"^\s*##\s*Exemplo\s+\d+", system_prompt, re.M)
        assert len(examples) >= 2, (
            f"Few-shot requires at least 2 examples, found: {len(examples)}"
        )

        # At least one example must demonstrate the output in the required format.
        assert "como um" in lower and "critérios de aceitação:" in lower, (
            "No example demonstrates the output in User Story format"
        )

    def test_prompt_no_todos(self, prompt):
        """Makes sure no `[TODO]` was left behind in the text."""
        # Case-sensitive on purpose: in Portuguese "todo" is a common word, and
        # ellipses in prose are legitimate. What must not remain is a leftover
        # placeholder marker.
        markers = [r"TODO", r"FIXME", r"XXX", r"PREENCHER",
                   r"\[\s*\.\.\.\s*\]", r"<\s*preencher\s*>"]

        for field in ("description", "system_prompt", "user_prompt"):
            text = str(prompt.get(field, ""))
            for marker in markers:
                assert not re.search(marker, text), (
                    f"Field '{field}' still contains the marker '{marker}'"
                )

    def test_minimum_techniques(self, prompt):
        """Checks (through the YAML metadata) that at least 2 techniques were listed."""
        techniques = prompt.get("techniques_applied", [])

        assert isinstance(techniques, list), "'techniques_applied' must be a list"
        assert len(techniques) >= 2, (
            f"At least 2 techniques required, found: {len(techniques)}"
        )

        # Few-shot is mandatory per the challenge statement; the others are free choice.
        assert any("few-shot" in t.lower() for t in techniques), (
            "Few-shot Learning is mandatory and is not listed in 'techniques_applied'"
        )

        # The full YAML structure must be valid as well.
        is_valid, errors = validate_prompt_structure(prompt)
        assert is_valid, f"Invalid prompt structure: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
