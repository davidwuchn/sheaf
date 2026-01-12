#!/usr/bin/env python3
# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Help system for Sheaf REPL.
Loads and formats documentation from docs/reference/ markdown files.
"""

import os
import re


def find_doc_file(name):
    """Find the documentation file for a given function/special-form name."""
    # Get the project root (sheaf/repl/help.py -> sheaf -> root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Search in functions, special-forms, and macros
    search_paths = [
        os.path.join(project_root, "docs", "reference", "functions", f"{name}.md"),
        os.path.join(project_root, "docs", "reference", "special-forms", f"{name}.md"),
        os.path.join(project_root, "docs", "reference", "macros", f"{name}.md"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            return path

    return None


def parse_markdown_doc(filepath):
    """Parse a markdown documentation file and extract key sections."""
    with open(filepath, "r") as f:
        content = f.read()

    # Split by lines
    lines = content.split("\n")

    # Extract YAML frontmatter
    frontmatter = {}
    if lines[0] == "---":
        i = 1
        while i < len(lines) and lines[i] != "---":
            if ":" in lines[i]:
                key, value = lines[i].split(":", 1)
                frontmatter[key.strip()] = value.strip()
            i += 1
        # Skip the closing ---
        lines = lines[i + 1 :]

    # Find main sections
    sections = {}
    current_section = None
    current_content = []

    for line in lines:
        # Check if this is a heading
        if line.startswith("## "):
            # Save previous section
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            # Start new section
            current_section = line[3:].strip()
            current_content = []
        elif line.startswith("# "):
            # Skip the main title (h1)
            continue
        else:
            current_content.append(line)

    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return frontmatter, sections


def strip_markdown(text):
    """Remove markdown formatting for terminal display."""
    # Remove code blocks
    text = re.sub(r"```[\w]*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    return text


def format_help_for_terminal(name, frontmatter, sections):
    """Format documentation sections for terminal display."""
    output = []

    # Title
    doc_type = frontmatter.get("type", "function")
    output.append(f"\n{name} ({doc_type})")
    output.append("=" * (len(name) + len(doc_type) + 3))
    output.append("")

    # Signature
    if "signature" in frontmatter:
        output.append(f"Signature: {frontmatter['signature']}")
        output.append("")

    # Description (from Signature section's first paragraph or content before it)
    if "Signature" in sections:
        # Get text before the code block
        sig_content = sections["Signature"]
        if sig_content:
            # Extract description before code blocks
            desc_match = re.match(r"^(.*?)```", sig_content, re.DOTALL)
            if desc_match:
                desc = desc_match.group(1).strip()
                if desc:
                    output.append(strip_markdown(desc))
                    output.append("")

    # Parameters
    if "Parameters" in sections:
        output.append("Parameters:")
        params_text = strip_markdown(sections["Parameters"])
        for line in params_text.split("\n"):
            line = line.strip()
            if line and line.startswith("-"):
                output.append(f"  {line[1:].strip()}")
        output.append("")

    # Examples (max 3 for brevity)
    if "Examples" in sections:
        output.append("Examples:")
        examples_text = sections["Examples"]

        # Extract code blocks
        code_blocks = re.findall(r"```sheaf\n(.*?)\n```", examples_text, re.DOTALL)

        for i, block in enumerate(code_blocks[:3]):  # Max 3 examples
            output.append("")
            for line in block.split("\n"):
                output.append(f"  {line}")

        if len(code_blocks) > 3:
            output.append("")
            output.append(f"  ... and {len(code_blocks) - 3} more examples")
        output.append("")

    # See Also
    if "See Also" in sections:
        output.append("See also:")
        see_also_text = strip_markdown(sections["See Also"])
        for line in see_also_text.split("\n"):
            line = line.strip()
            if line and line.startswith("-"):
                # Extract just the function name (before the dash)
                match = re.match(r"- \[?`?([^\]`\-]+)", line)
                if match:
                    output.append(f"  {match.group(1).strip()}")
        output.append("")

    return "\n".join(output)


def get_help(name):
    """Get help for a function or special form."""
    # Find the doc file
    doc_file = find_doc_file(name)

    if not doc_file:
        return f"\nNo documentation found for '{name}'.\n\nTry:\n  :env    to see available functions\n  :help   for REPL commands\n"

    # Parse and format
    try:
        frontmatter, sections = parse_markdown_doc(doc_file)
        return format_help_for_terminal(name, frontmatter, sections)
    except Exception as e:
        return f"\nError loading documentation for '{name}': {e}\n"
