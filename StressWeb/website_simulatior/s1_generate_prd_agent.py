#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate PRD (outline + draft) via Claude Agent SDK.

This script mirrors the v1 PRD generation but calls the agent SDK instead of the
raw API. It streams the outline and draft, then saves them under a timestamped
folder inside SimWeb_v2/.
"""

import os
import sys
import asyncio
import re
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment variables required by CCR / Anthropic SDK
# ---------------------------------------------------------------------------
os.environ["ANTHROPIC_BASE_URL"] = os.environ.get("ANTHROPIC_BASE_URL", "")
os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("ANTHROPIC_AUTH_TOKEN", "ccr-local")
os.environ["ANTHROPIC_MODEL"] = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# Workspace paths
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR

# Try importing the agent SDK early to surface missing dependency
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
    SDK_AVAILABLE = True
except ImportError as e:
    SDK_AVAILABLE = False
    IMPORT_ERROR = e

# ---------------------------------------------------------------------------
# Length presets
# ---------------------------------------------------------------------------
LENGTH_PRESETS = {
    "brief": {
        "description": "Brief version - quick overview",
        "outline_max_tokens": 2048,
        "draft_max_tokens": 4096,
    },
    "standard": {
        "description": "Balanced version - detail vs readability",
        "outline_max_tokens": 4096,
        "draft_max_tokens": 8192,
    },
    "detailed": {
        "description": "Detailed version - comprehensive and deep",
        "outline_max_tokens": 6144,
        "draft_max_tokens": 12288,
    },
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
OUTLINE_PROMPT = """
You are a world-class product manager. Your task is to create a structured
outline for a Product Requirements Document (PRD) based on a given project idea.
The outline should cover all standard sections of a PRD, including:
1.  Executive Summary
2.  Problem Statement & User Personas
3.  Goals & Success Metrics
4.  Functional Requirements
5.  Non-Functional Requirements (Performance, Security, etc.)
6.  Risks & Mitigations
Please generate a Markdown-formatted outline for the following project idea:
**Project Idea:** "{idea}"
**Length Target:** {length_guidance}
**Instructions:**
- Use Markdown headings (`#`, `##`, `###`) to structure the document.
- For each section, include a brief, one-sentence placeholder description of what it will contain.
- Do NOT write the full content of the PRD yet. Just the outline.
- Limit the numbers of sections, subsections and subsubsections to maintain readability.
"""

DRAFT_PROMPT = """
You are a world-class product manager. Your task is to expand a given PRD
outline into a full first draft.
Use the provided outline and flesh out each section with clear, and
concise content. Make reasonable assumptions where necessary, but clearly
state them.
**PRD Outline to Draft:**
```markdown
{outline}
```
**Length Target:** {length_guidance}
**Instructions:**
- Write comprehensive, clear and concise content for every section of the outline.
- Use brief, clear and professional language.
- Format the output as a complete Markdown document.
- Ensure the functional requirements are specific and actionable.
- Limit the numbers of sections, subsections and subsubsections to maintain readability.
- Stay within the target length while maintaining quality.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_project_folder(pname: str | None = None) -> tuple[Path, str]:
    """Create a timestamped project folder under SimWeb_v2."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if pname:
        clean = re.sub(r"[^a-z0-9_]", "", pname.lower().replace(" ", "_"))
        folder_name = f"PRD_{timestamp}_{clean}" if clean else f"PRD_{timestamp}"
    else:
        folder_name = f"PRD_{timestamp}"
    folder_path = CURRENT_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path, folder_name


def save_to_file(content: str, path: Path) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"[INFO] Saved to: {path}")


async def run_agent(
    prompt: str,
    options: ClaudeAgentOptions,
    label: str,
    heartbeat_secs: int = 15,
    overall_timeout: int | None = None,
) -> str:
    """Stream agent output with heartbeat; returns concatenated text.

    Heartbeat helps detect if the agent is waiting (e.g., on permissions/CCR).
    """

    print(f"[AGENT] {label} | options: cwd={options.cwd}, max_turns={getattr(options, 'max_turns', None)}")
    start = time.time()
    chunks: list[str] = []

    async def _consume() -> str:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        chunks.append(text)
                        print(text, end='', flush=True)
        return "".join(chunks)

    task = asyncio.create_task(_consume())
    waited = 0

    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=heartbeat_secs, return_when=asyncio.FIRST_COMPLETED)

            if task in done:
                result = task.result()
                break

            waited += heartbeat_secs
            print(f"[WAIT] {label}: still running after {waited}s... (if no streaming appears, check CCR/tool permissions)")

            if overall_timeout and (time.time() - start) >= overall_timeout:
                task.cancel()
                raise TimeoutError(f"{label} exceeded timeout of {overall_timeout}s")

        duration = time.time() - start
        print()  # newline after stream
        print(f"[AGENT] {label} finished in {duration:.2f}s\n")
        return result
    except asyncio.CancelledError:
        # Propagate cancellation cleanly with context
        raise


async def generate_prd_agent(
    project_idea: str,
    model: str,
    length_preset: str,
    pname: str | None = None,
    max_turns: int = 15,
    heartbeat_secs: int = 15,
    overall_timeout: int | None = None,
) -> dict:
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    if length_preset not in LENGTH_PRESETS:
        print(f"[WARN] Invalid length preset '{length_preset}', defaulting to 'standard'.")
        length_preset = "standard"
    preset = LENGTH_PRESETS[length_preset]

    print("=" * 70)
    print(f"PROJECT IDEA: {project_idea}")
    print(f"LENGTH PRESET: {length_preset.upper()} - {preset['description']}")
    if pname:
        print(f"PROJECT NAME: {pname}")
    print("=" * 70)

    project_folder, folder_name = create_project_folder(pname)
    print(f"[INFO] Created project folder: {folder_name}\n")

    # Build common agent options
    # Use "auto" permission mode to avoid manual approvals
    base_options = {
        "cwd": str(WORKSPACE_DIR),
        "permission_mode": "bypassPermissions",  # Auto-approve all actions
        "allowed_tools": "*",  # Allow all tools to avoid blocking
        "max_turns": max_turns,
        "system_prompt": """You are a helpful product manager assistant. Write concise, well-structured PRD content in Markdown.
IMPORTANT: Generate the entire response as text output. Do NOT use tools like Write/Edit to create files - just output the markdown content directly.""",
    }

    # 1) Outline
    outline_prompt = OUTLINE_PROMPT.format(
        idea=project_idea,
        length_guidance=preset["description"],
    )
    outline_options = ClaudeAgentOptions(**base_options)
    outline = await run_agent(
        outline_prompt,
        outline_options,
        label="PRD Outline",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )
    outline_path = project_folder / "01_prd_outline.md"
    save_to_file(outline, outline_path)

    # 2) Draft
    draft_prompt = DRAFT_PROMPT.format(
        outline=outline,
        length_guidance=preset["description"],
    )
    draft_options = ClaudeAgentOptions(**base_options)
    draft = await run_agent(
        draft_prompt,
        draft_options,
        label="PRD Draft",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )
    draft_path = project_folder / "02_prd_full_draft.md"
    save_to_file(draft, draft_path)

    # Save original idea
    idea_path = project_folder / "00_project_idea.txt"
    save_to_file(project_idea, idea_path)

    print("=" * 70)
    print("[SUCCESS] PRD generation via agent completed!")
    print(f"[INFO] All files saved in: {project_folder}")
    print("=" * 70)

    return {"folder": str(project_folder), "outline": outline, "prd": draft}


def parse_args() -> tuple[str, str, str | None, int, int, int | None, str]:
    import argparse

    parser = argparse.ArgumentParser(description="Generate PRD via Claude Agent SDK")
    parser.add_argument("idea", help="Project idea text (quoted if it has spaces)")
    parser.add_argument("--preset", "-p", default="standard", choices=list(LENGTH_PRESETS.keys()), help="Length preset")
    parser.add_argument("--pname", "-n", default=None, help="Optional folder suffix")
    parser.add_argument("--max-turns", type=int, default=15, help="Max turns for the agent")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval to show agent still running")
    parser.add_argument("--timeout-secs", type=int, default=3600, help="Overall timeout for each agent call (default: 600s/10min)")
    parser.add_argument("--model", "-m", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"), help="Model name")

    args = parser.parse_args()
    return (
        args.idea,
        args.preset,
        args.pname,
        args.max_turns,
        args.heartbeat_secs,
        args.timeout_secs,
        args.model,
    )


def main() -> None:
    if not SDK_AVAILABLE:
        print(f"[ERROR] claude_agent_sdk not installed: {IMPORT_ERROR}")
        print("Install with: pip install claude-agent-sdk")
        sys.exit(1)

    (
        project_idea,
        length_preset,
        pname,
        max_turns,
        heartbeat_secs,
        timeout_secs,
        model,
    ) = parse_args()

    os.environ["ANTHROPIC_MODEL"] = model  # ensure downstream uses the selected model

    try:
        asyncio.run(
            generate_prd_agent(
                project_idea,
                model=model,
                length_preset=length_preset,
                pname=pname,
                max_turns=max_turns,
                heartbeat_secs=heartbeat_secs,
                overall_timeout=timeout_secs,
            )
        )
    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()