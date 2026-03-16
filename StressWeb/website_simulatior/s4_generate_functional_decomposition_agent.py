#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a functional decomposition via Claude Agent SDK.

The agent reads the PRD file itself, produces a full functional decomposition,
and saves the result to functional_decomposition.md inside the project folder.
"""

import os
import sys
import asyncio
import re
import time
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Environment variables required by CCR / Anthropic SDK
# ---------------------------------------------------------------------------
os.environ["ANTHROPIC_BASE_URL"] = os.environ.get("ANTHROPIC_BASE_URL", "")
os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("ANTHROPIC_AUTH_TOKEN", "ccr-local")
os.environ["ANTHROPIC_MODEL"] = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

CURRENT_DIR = Path(__file__).resolve().parent

# Try importing the agent SDK
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
    SDK_AVAILABLE = True
except ImportError as e:
    SDK_AVAILABLE = False
    IMPORT_ERROR = e

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
AGENT_FUNCTIONAL_DECOMPOSITION_PROMPT = """
You are a senior software architect. Analyze the PRD and generate a comprehensive functional decomposition in clean Markdown format. Output the FULL complete document—do not abbreviate or use placeholders.

**Input:** {prd_file_path}

**Your Output (plain text, complete Markdown document):**

Generate a document with these 3 sections:

### Section 1: Database Schema Design
List 4-6 core tables. For each table:
- SQL CREATE TABLE statement with field names, types, constraints
- Brief purpose
- 2-3 sample data rows in a markdown table

### Section 2: Core Functions / APIs
Group related functions. For each function:
- Function name and exact signature (with parameter types)
- 1-line purpose
- Input parameters (with types)
- Return type/shape
- Database operations (which tables, read/write)
- 2-3 line logic summary

List 10-15 functions total.

### Section 3: User Scenarios (3-5 scenarios)
For each scenario:
- Scenario name
- 5-8 step-by-step flow
- Each step: user action → function called → DB change (if any)

**Output format:**
- Use Markdown syntax (headings, code blocks, tables)
- Use exact field/table names from PRD
- Be concrete, no placeholders like [example]
- Output the COMPLETE full document

Start. Output only the Markdown document, nothing else.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def run_agent(
    prompt: str,
    options: ClaudeAgentOptions,
    label: str,
    output_file: Path | None = None,
    heartbeat_secs: int = 60,
    overall_timeout: int | None = None,
    file_timeout: int = 3000,
) -> str:
    """Stream agent output with heartbeat and file creation check."""
    print(f"[AGENT] {label} | cwd={options.cwd}, max_turns={getattr(options, 'max_turns', None)}")
    start = time.time()
    file_check_start = time.time()
    chunks: list[str] = []

    async def _consume() -> str:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        chunks.append(text)
                        print(text, end="", flush=True)
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
            elapsed = time.time() - start
            print(f"\n[WAIT] {label}: still running after {waited}s...")
            
            # Check if output file exists yet
            if output_file and (time.time() - file_check_start) > file_timeout:
                if not output_file.exists():
                    print(f"[WARN] {output_file.name} not created after {file_timeout}s, cancelling agent...")
                    task.cancel()
                    raise TimeoutError(f"{label}: output file not created after {file_timeout}s")
                else:
                    print(f"[INFO] Output file detected: {output_file.name}")
                    task.cancel()
                    print("[INFO] Stopping agent after output file created")
                    break
            
            if overall_timeout and elapsed >= overall_timeout:
                task.cancel()
                raise TimeoutError(f"{label} exceeded timeout of {overall_timeout}s")

        duration = time.time() - start
        print()
        print(f"[AGENT] {label} finished in {duration:.2f}s\n")
        return result
    except asyncio.CancelledError:
        raise


async def generate_functional_decomposition_agent(
    prd_folder: Path,
    max_turns: int = 15,
    heartbeat_secs: int = 60,
    overall_timeout: int = 600,
) -> dict:
    """Run agent to generate functional decomposition text (no file write by agent)."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    print("=" * 70)
    print("FUNCTIONAL DECOMPOSITION (Agent)")
    print("=" * 70)

    full_prd_path = prd_folder / "02_prd_full_draft.md"
    outline_path = prd_folder / "01_prd_outline.md"
    idea_path = prd_folder / "00_project_idea.txt"

    if full_prd_path.exists():
        prd_file_path = str(full_prd_path)
        print(f"[INFO] PRD file: {full_prd_path.name}")
    elif outline_path.exists():
        prd_file_path = str(outline_path)
        print(f"[INFO] PRD file: {outline_path.name}")
    elif idea_path.exists():
        prd_file_path = str(idea_path)
        print(f"[INFO] PRD file: {idea_path.name}")
    else:
        raise FileNotFoundError(f"No PRD files found in {prd_folder}")

    # Read PRD content ourselves
    print(f"[INFO] Reading PRD content...")
    prd_content = (Path(prd_file_path)).read_text(encoding="utf-8")
    if len(prd_content) > 50000:
        prd_content = prd_content[:50000] + "\n\n[... content truncated ...]"
    print(f"[INFO] PRD size: {len(prd_content)} chars\n")

    prompt = AGENT_FUNCTIONAL_DECOMPOSITION_PROMPT.format(
        prd_file_path=prd_file_path,
        output_file_path="functional_decomposition.md",
    )

    agent_options = ClaudeAgentOptions(
        cwd=str(prd_folder),
        permission_mode="bypassPermissions",
        allowed_tools="*",
        max_turns=max_turns,
        system_prompt="""You are a senior software architect. Generate a functional decomposition NOW.

Your task: Output a COMPLETE Markdown document (not partial or abbreviated) with:
1. Database Schema Design (4-6 tables with SQL, purpose, sample rows)
2. Core Functions / APIs (10-15 functions with full signatures, inputs, outputs, DB ops)
3. User Scenarios (3-5 detailed scenarios with step-by-step flows)

Requirements:
- Output ONLY the Markdown document—no explanations or preamble
- Use concrete names from the PRD (no placeholders)
- Include complete SQL CREATE TABLE statements
- Make all function signatures explicit with types
- Be thorough—this is the complete deliverable

Start outputting the Markdown now.
""",
    )

    print("[STEP] Agent generating functional decomposition...")
    print("-" * 70)

    response = await run_agent(
        prompt,
        agent_options,
        label="Functional Decomposition",
        output_file=None,
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
        file_timeout=9999,  # Disable file check since we're not using Write tool
    )

    # Save response to file
    output_file = prd_folder / "functional_decomposition.md"
    
    # Extract markdown from response (in case there's extra text)
    markdown_content = response.strip()
    
    # Save to file
    try:
        output_file.write_text(markdown_content, encoding="utf-8")
        print(f"\n[INFO] Saved to: {output_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        return {"files_created": [], "output_folder": str(prd_folder)}

    files_created = []
    if output_file.exists():
        files_created.append(output_file.name)
        file_size = output_file.stat().st_size
        print(f"[INFO] File size: {file_size:,} bytes")

    print("\n" + "=" * 70)
    print("[DONE] Functional decomposition completed")
    print(f"[INFO] Output folder: {prd_folder}")
    if files_created:
        print(f"[INFO] Files created: {', '.join(files_created)}")
    print("=" * 70)

    return {
        "files_created": files_created,
        "output_folder": str(prd_folder),
    }


def find_project_folder(pname: str | None = None) -> Path:
    """Find project folder by pname or pick the latest.
    
    Matching strategy for pname:
    1. Try EXACT word boundary match at end: _PNAME$ (e.g., "_a$" for pname="a")
    2. Try word boundary match: _PNAME(?:_|$) (e.g., "_gm_" or "_gm$")
    3. Fall back to substring match, preferring pname that appears later in name
    """
    prd_folders = [
        f for f in CURRENT_DIR.iterdir()
        if f.is_dir() and f.name.startswith("PRD_")
    ]

    if not prd_folders:
        raise FileNotFoundError("No PRD folders found")

    if pname:
        clean_pname = re.sub(r"[^a-z0-9_]", "", pname.lower().replace(" ", "_"))
        
        # Strategy 1: Try EXACT word boundary match at end (e.g., "_a$" not "_sa")
        pattern_exact = rf"_{clean_pname}$"
        matching = [f for f in prd_folders if re.search(pattern_exact, f.name.lower())]
        
        if matching:
            selected_folder = sorted(matching, reverse=True)[0]
            return selected_folder
        
        # Strategy 2: Try word boundary match (pname as separate segment)
        pattern = rf"_{clean_pname}(?:_|$)"
        matching = [f for f in prd_folders if re.search(pattern, f.name.lower())]
        
        if matching:
            selected_folder = sorted(matching, reverse=True)[0]
            return selected_folder
        
        # Strategy 3: Fall back to substring match (prefer pname that appears later)
        potential = [f for f in prd_folders if clean_pname in f.name.lower()]
        if potential:
            selected_folder = sorted(potential, key=lambda f: (-f.name.lower().rfind(clean_pname), f.name), reverse=True)[0]
            return selected_folder
        
        raise FileNotFoundError(f"No PRD folder found with pname: {pname}")
    else:
        selected_folder = sorted(prd_folders, reverse=True)[0]
        return selected_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Generate functional decomposition via Claude Agent SDK")
    parser.add_argument("--pname", "-n", default=None, help="PRD project name to find folder")
    parser.add_argument("--max-turns", type=int, default=15, help="Max turns for the agent")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval")
    parser.add_argument("--timeout-secs", type=int, default=3600, help="Overall timeout (default: 600s/10min)")
    parser.add_argument("--model", "-m", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"))
    return parser.parse_args()


def main():
    if not SDK_AVAILABLE:
        print(f"[ERROR] claude_agent_sdk not installed: {IMPORT_ERROR}")
        print("Install with: pip install claude-agent-sdk")
        sys.exit(1)

    args = parse_args()
    os.environ["ANTHROPIC_MODEL"] = args.model

    print("\n" + "=" * 70)
    print("FUNCTIONAL DECOMPOSITION GENERATOR (Agent SDK)")
    print("=" * 70 + "\n")

    try:
        prd_folder = find_project_folder(args.pname)
        print(f"[INFO] Using PRD folder: {prd_folder.name}\n")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        result = asyncio.run(
            generate_functional_decomposition_agent(
                prd_folder,
                max_turns=args.max_turns,
                heartbeat_secs=args.heartbeat_secs,
                overall_timeout=args.timeout_secs,
            )
        )

        print("\n[SUMMARY]")
        if result["files_created"]:
            for file in result["files_created"]:
                print(f"  - {file}")
        else:
            print("  - No output file detected. Check agent logs.")
        print(f"Output folder: {result['output_folder']}")

    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
