#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate structured trajectory documents via Claude Agent SDK.

Reads page trajectory documents and extracts/restructures them into
clean, consistent structured format with database operations summary.
"""

import os
import sys
import asyncio
import re
import json
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
# Prompt template for structured extraction
# ---------------------------------------------------------------------------
AGENT_STRUCTURED_EXTRACTION_PROMPT = """
You are a database and system architecture expert. Extract and restructure page trajectory information into a clean, consistent structured format.

**Input Page Trajectory Document:**
{trajectory_content}

**Database Schema Reference:**
{db_schema}

**Your Task:**
Generate a COMPLETE Markdown document following this structure:

# Structured Page Trajectory: [Scenario Title]

## Scenario Overview
- **User Goal:** [what user wants to accomplish]
- **Total Pages:** [count]
- **Total Database Operations:** [approximate count]
- **Estimated Duration:** [steps/time]

---

## Page_0: [Page Title]

### pagename
`filename.html` - [brief description of page purpose]

### db_snapshot
**Involved Database Tables Before This Page:**
[Show all table data states in SQL or JSON format with actual values, IDs, timestamps]

### pagedata
**Data Displayed on UI:**
[Show UI elements in structured format - form fields, lists, buttons, data displayed]

### Action_within_page_0

#### user_action_affecting_db

**Action 1: [Action Description]**
- User manual operation: [what user does]
- Triggered function: `functionName(params)` [if any]
- Database changes: [SQL INSERT/UPDATE/DELETE statements]
- Result: [what changed]

**Action 2: [Next Action]**
[repeat format]

---

## Page_1: [Next Page Title]
[Repeat same structure for each page]

---

## Summary

### Page Flow Sequence
```
Page_0 → Page_1 → Page_2 → ... → Page_N
  ↓        ↓        ↓             ↓
Action   Action   Action        Final
```

### Database Operations Summary
- **Total Read Operations (SELECT):** [count]
- **Total Write Operations (INSERT/UPDATE/DELETE):** [count]
- **Tables Read:** [list]
- **Tables Modified:** [list]

### Functions Used
[List all functions called with their purposes]

### Key Statistics
- **Minimum User Steps:** [count]
- **Maximum User Steps:** [count]
- **Automatic Steps:** [count]
- **Optional Steps:** [count]

**Important Requirements:**
- Extract ONLY information from input document
- Use exact data values, IDs, timestamps
- Distinguish READ (SELECT) and WRITE (INSERT/UPDATE/DELETE)
- Be precise about which tables are affected
- Group operations clearly by page
- Use valid SQL/JSON syntax for data
- Output ONLY Markdown (no preamble)

Generate the complete structured document now.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def run_agent(
    prompt: str,
    options: ClaudeAgentOptions,
    label: str,
    heartbeat_secs: int = 60,
    overall_timeout: int | None = None,
) -> str:
    """Stream agent output with heartbeat."""
    print(f"[AGENT] {label}")
    start = time.time()
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

            if overall_timeout and elapsed >= overall_timeout:
                task.cancel()
                raise TimeoutError(f"{label} exceeded timeout of {overall_timeout}s")

        duration = time.time() - start
        print()
        print(f"[AGENT] {label} finished in {duration:.2f}s\n")
        return result
    except asyncio.CancelledError:
        raise


async def generate_structured_trajectory(
    trajectory_path: Path,
    db_schema: str,
    output_folder: Path,
    query_id: str,
    heartbeat_secs: int = 60,
    overall_timeout: int = 360,
) -> dict:
    """Generate structured trajectory for a single trajectory file."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    # Read trajectory content
    trajectory_content = trajectory_path.read_text(encoding="utf-8")
    if len(trajectory_content) > 40000:
        trajectory_content = trajectory_content[:40000] + "\n\n[... content truncated ...]"

    prompt = AGENT_STRUCTURED_EXTRACTION_PROMPT.format(
        trajectory_content=trajectory_content,
        db_schema=db_schema[:5000] if db_schema else "No schema provided",
    )

    agent_options = ClaudeAgentOptions(
        cwd=str(output_folder),
        permission_mode="bypassPermissions",
        allowed_tools="*",
        max_turns=10,
        system_prompt="""You are a database and system architecture expert. Restructure trajectory documents.

Your task: Extract page trajectory information and generate ONE complete structured Markdown document.

Requirements:
- Output ONLY Markdown (no preamble)
- Show each page with: pagename, db_snapshot, pagedata, actions
- Include database operation summary (read/write counts, tables)
- Use exact data values, timestamps, IDs
- Group operations clearly by page
- Distinguish READ (SELECT) vs WRITE (INSERT/UPDATE/DELETE)
- Be thorough and complete

Start outputting Markdown now.
""",
    )

    print("=" * 70)
    print(f"Query {query_id}: {trajectory_path.name}")
    print("=" * 70)
    print("[STEP] Agent generating structured trajectory...")
    print("-" * 70)

    response = await run_agent(
        prompt,
        agent_options,
        label=f"Query {query_id} Structuring",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )

    # Save to file
    output_file = output_folder / f"query_{query_id}_structured_trajectory.md"
    markdown_content = response.strip()

    try:
        output_file.write_text(markdown_content, encoding="utf-8")
        file_size = output_file.stat().st_size
        print(f"[INFO] Saved: {output_file.name} ({file_size:,} bytes)")
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        return {"query_id": query_id, "success": False, "error": str(e)}

    print()
    return {
        "query_id": query_id,
        "input_file": trajectory_path.name,
        "output_file": output_file.name,
        "success": True,
    }


async def process_trajectories_batch(
    trajectory_files: list[Path],
    db_schema: str,
    output_folder: Path,
    heartbeat_secs: int = 60,
    query_timeout: int = 360,
) -> dict:
    """Process multiple trajectory files."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    print("=" * 70)
    print(f"GENERATING STRUCTURED TRAJECTORIES ({len(trajectory_files)} files)")
    print("=" * 70 + "\n")

    results = []
    for i, trajectory_path in enumerate(trajectory_files, 1):
        query_id = extract_query_id(trajectory_path.name)
        print(f"\n[{i}/{len(trajectory_files)}] Processing query {query_id}...")

        try:
            result = await generate_structured_trajectory(
                trajectory_path,
                db_schema,
                output_folder,
                query_id,
                heartbeat_secs=heartbeat_secs,
                overall_timeout=query_timeout,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Query {query_id}: {e}")
            results.append({"query_id": query_id, "success": False, "error": str(e)})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\n✓ Successfully processed: {len(successful)}/{len(trajectory_files)}")

    if successful:
        print("\n[FILES CREATED]")
        for r in successful:
            print(f"  - Query {r['query_id']}: {r['input_file']}")
            print(f"    → {r['output_file']}")

    if failed:
        print(f"\n⚠️  Failed: {len(failed)} trajectories")
        for r in failed:
            print(f"  - Query {r['query_id']}: {r.get('error', 'Unknown error')}")

    print(f"\n[OUTPUT LOCATION]")
    print(f"  {output_folder}")
    print("=" * 70 + "\n")

    # Create index file
    if successful:
        index_content = "# Structured Page Trajectories\n\n"
        index_content += f"Generated: {len(successful)} structured trajectory documents\n\n"
        index_content += "## Available Trajectories\n\n"

        for r in sorted(successful, key=lambda x: int(x["query_id"])):
            index_content += f"- [Query {r['query_id']}](./{r['output_file']})\n"

        index_file = output_folder / "structured_trajectories_index.md"
        try:
            index_file.write_text(index_content, encoding="utf-8")
            print(f"[INFO] Created index file: {index_file.name}\n")
        except Exception as e:
            print(f"[WARN] Failed to create index: {e}\n")

    return {
        "total": len(trajectory_files),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
        "output_folder": str(output_folder),
    }


def extract_query_id(filename: str) -> str:
    """Extract query ID from trajectory filename."""
    match = re.search(r"query_(\d+)_page_trajectory\.md", filename)
    if match:
        return match.group(1)
    return "unknown"


def find_trajectory_files(prd_folder: Path) -> list[Path]:
    """Find all page trajectory files in PRD folder, sorted numerically."""
    trajectory_files = list(prd_folder.glob("query_*_page_trajectory.md"))
    
    # Sort numerically by query ID (not alphabetically)
    def get_query_id(path: Path) -> int:
        """Extract query ID from filename for numeric sorting."""
        match = re.search(r'query_(\d+)_page_trajectory', path.name)
        return int(match.group(1)) if match else 0
    
    trajectory_files.sort(key=get_query_id)
    return trajectory_files


def load_db_schema(prd_folder: Path) -> str:
    """Load functional decomposition document (database schema)."""
    schema_file = prd_folder / "functional_decomposition.md"
    if schema_file.exists():
        content = schema_file.read_text(encoding="utf-8")
        print(f"[INFO] Loaded DB schema: {len(content)} chars")
        return content
    else:
        print("[WARN] functional_decomposition.md not found")
        return ""


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
    parser = argparse.ArgumentParser(description="Generate structured trajectories via Agent SDK")
    parser.add_argument("--pname", "-n", default=None, help="PRD project name to find folder")
    parser.add_argument("--query-id", "-q", type=int, default=None, help="Specific query ID to process (e.g., -q 1 processes only query 1)")
    parser.add_argument("--queries", type=int, default=None, help="Number of trajectories to process (default: all)")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval")
    parser.add_argument("--query-timeout", type=int, default=1200, help="Timeout per query (default: 1200s/20min)")
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
    print("STRUCTURED TRAJECTORY GENERATOR (Agent SDK)")
    print("=" * 70 + "\n")

    try:
        prd_folder = find_project_folder(args.pname)
        print(f"[INFO] Using PRD folder: {prd_folder.name}\n")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Find trajectory files
    trajectory_files = find_trajectory_files(prd_folder)
    if not trajectory_files:
        print("[ERROR] No page trajectory files found")
        print("[INFO] Expected files like: query_1_page_trajectory.md")
        print("[INFO] Run s5 first to generate trajectory files")
        sys.exit(1)

    print(f"[INFO] Found {len(trajectory_files)} trajectory file(s)")

    # Load DB schema
    db_schema = load_db_schema(prd_folder)
    if not db_schema:
        print("[WARN] Database schema not available, proceeding without it...")
        db_schema = ""

    print()

    # Determine which files to process
    if args.query_id is not None:
        # Process specific query ID
        # Find trajectory file matching this query ID
        matching_files = [f for f in trajectory_files if f"query_{args.query_id}_" in f.name]
        if not matching_files:
            print(f"[ERROR] No trajectory file found for query ID {args.query_id}")
            print(f"[INFO] Available queries: {', '.join([f.stem.split('_')[1] for f in trajectory_files])}")
            sys.exit(1)
        files_to_process = matching_files
        print(f"[INFO] Processing specific query ID: {args.query_id}")
    else:
        # Default: process all queries (or first N if --queries specified)
        if args.queries is not None:
            files_to_process = trajectory_files[:args.queries]
            print(f"[INFO] Processing first {len(files_to_process)} queries")
        else:
            files_to_process = trajectory_files
            print(f"[INFO] Processing all {len(files_to_process)} queries (default behavior)")

    try:
        result = asyncio.run(
            process_trajectories_batch(
                files_to_process,
                db_schema,
                prd_folder,
                heartbeat_secs=args.heartbeat_secs,
                query_timeout=args.query_timeout,
            )
        )

        print("[FINAL STATS]")
        print(f"Processed: {result['successful']}/{result['total']} successful")
        print(f"Output folder: {result['output_folder']}")

    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
