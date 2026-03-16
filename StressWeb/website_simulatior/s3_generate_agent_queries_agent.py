#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate agent test queries via Claude Agent SDK.

This script reads a PRD and generates 15 realistic user queries for testing
an AI web navigation agent. Queries should be clear, specific, and actionable.
"""

import os
import sys
import asyncio
import re
import time
from pathlib import Path
import argparse
import json

# ---------------------------------------------------------------------------
# Environment variables required by CCR / Anthropic SDK
# ---------------------------------------------------------------------------
os.environ["ANTHROPIC_BASE_URL"] = os.environ.get("ANTHROPIC_BASE_URL", "")
os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("ANTHROPIC_AUTH_TOKEN", "ccr-local")
os.environ["ANTHROPIC_MODEL"] = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# Workspace paths
CURRENT_DIR = Path(__file__).resolve().parent

# Try importing the agent SDK
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
    SDK_AVAILABLE = True
except ImportError as e:
    SDK_AVAILABLE = False
    IMPORT_ERROR = e

# ---------------------------------------------------------------------------
# Query generation prompt template
# ---------------------------------------------------------------------------
AGENT_QUERY_GENERATION_PROMPT = """
Your task is to generate 15 user queries for testing an AI web navigation agent based on the PRD.

**Instructions:**

1. Read the PRD file from: {prd_file_path}
2. Analyze the website features and user workflows
3. Generate 15 specific, actionable, executable test queries with clear difficulty gradient
4. Save the queries as a JSON file

**Query Requirements:**

1. **Specificity & Clarity:**
     - All text inputs must be SPECIFIC (e.g., "Search for 'software engineer'" not "Search for a job")
     - All numbers must be EXACT (e.g., "Find top 5 posts" not "Find several posts")
     - Include specific field names, button labels, and navigation paths

2. **Difficulty Gradient (Easy → Hard):**
     - Queries 1-4: Simple tasks (3+ pages)
     - Queries 5-13: Medium tasks (5+ pages)
     - Queries 14-15: Complex tasks (8+ pages)

3. **Realistic & Actionable:**
     - Use actual features from the PRD
     - Include specific data (names, numbers, dates, keywords)
     - Cover different user personas and goals
     - Test various capabilities of the agent in this setting

4. **Executable (NO passive “read-only” tasks):**
     - Every query must produce an observable result: create/update/delete data,save to collection, submit form
     - Avoid non-operational verbs like "read/check/see" without an action, if used, must come with a follow-up action
     - Specify exact counts/text/filters so the agent can validate completion
     - Do not use commands that cannot be executed such as verify, and do not appear commands such as screenshot that are not related to the business scenario
**Output Format:**

Create a JSON file named `agent_queries.json` with this structure:

```json
{{
    "project_name": "Exact project name from PRD",
    "total_queries": 15,
    "queries": [
        {{
            "id": 1,
            "difficulty": "easy",
            "query": "Specific, clear, executable instruction with exact text/numbers and validation",
            "required_capabilities": ["navigation", "form_fill", "download"],
            "estimated_steps": 2,
            "description": "What this query tests"
        }}
    ]
}}
```

**Example Query (GOOD - Executable):**
"Search for 'Senior Software Engineer' roles in 'San Francisco', filter salary to '$150k-$200k', save the top 2 postings to Favorites"

**Example Query (BAD - Non-actionable):**
"Open the About page and read the mission statement"

**Workflow:**

1. Read PRD file using Read tool
2. Analyze features and identify testable workflows
3. Generate 15 specific, executable queries with exact text/numbers and validation steps
4. Use Write tool to save as `agent_queries.json`
5. Create `agent_queries.md` for human-readable format
6. Create `agent_queries_list.txt` for simple list

Start now by reading the PRD and generating executable, verifiable queries.
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
    print(f"[AGENT] {label} | cwd={options.cwd}, max_turns={getattr(options, 'max_turns', None)}")
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
            print(f"\n[WAIT] {label}: still running after {waited}s...")
            if overall_timeout and (time.time() - start) >= overall_timeout:
                task.cancel()
                raise TimeoutError(f"{label} exceeded timeout of {overall_timeout}s")
        
        duration = time.time() - start
        print()
        print(f"[AGENT] {label} finished in {duration:.2f}s\n")
        return result
    except asyncio.CancelledError:
        raise


async def generate_agent_queries_agent(
    prd_folder: Path,
    max_turns: int = 20,
    heartbeat_secs: int = 60,
    overall_timeout: int = 900,
) -> dict:
    """Generate agent test queries via agent."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    print("=" * 70)
    print("GENERATING AGENT QUERIES FROM PRD")
    print("=" * 70)

    # Find PRD file path
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

    print(f"[INFO] Generating 15 test queries\n")

    # Build prompt - let agent read the file itself
    query_prompt = AGENT_QUERY_GENERATION_PROMPT.format(
        prd_file_path=prd_file_path,
    )

    # Agent options - give it file access
    agent_options = ClaudeAgentOptions(
        cwd=str(prd_folder),
        permission_mode="bypassPermissions",
        allowed_tools="*",
        max_turns=max_turns,
        system_prompt="""You are an expert UX designer and AI agent test engineer.

    Your task:
    1. Read the PRD file using the Read tool
    2. Analyze website features and user workflows
    3. Generate 15 specific, actionable, EXECUTABLE test queries with difficulty gradient (easy/medium/hard)
    4. Ensure ALL queries have specific text inputs, exact numbers, and verifiable outcomes
    5. Use Write tool to save queries as JSON, Markdown, and text files

    Critical requirements:
    - Be SPECIFIC: "Search for 'software engineer'" not "Search for jobs"
    - Use EXACT numbers: "View top 5 results" not "View some results"
    - Include CLEAR actions that produce observable results (save, submit, export, compare) — avoid passive "read/check" only
    - Each query must have completion/validation criteria (e.g., item appears in Favorites, CSV downloaded, status updated)
    - Test various features: navigation, search, filter, create, compare, analyze

    Work systematically and save all output files.""",
    )

    print("[STEP] Agent will now read PRD and generate query files...")
    print("-" * 70)
    
    response = await run_agent(
        query_prompt,
        agent_options,
        label="Query Generation",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )

    # Check if files were created
    json_file = prd_folder / "agent_queries.json"
    md_file = prd_folder / "agent_queries.md"
    txt_file = prd_folder / "agent_queries_list.txt"

    queries_data = None
    if json_file.exists():
        try:
            queries_data = json.loads(json_file.read_text(encoding="utf-8"))
            print(f"[INFO] Loaded queries from: {json_file.name}")
        except json.JSONDecodeError as e:
            print(f"[WARN] Could not parse {json_file.name}: {e}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Agent completed query generation!")
    print(f"[INFO] Output folder: {prd_folder}")
    
    files_created = []
    if json_file.exists():
        files_created.append("agent_queries.json")
    if md_file.exists():
        files_created.append("agent_queries.md")
    if txt_file.exists():
        files_created.append("agent_queries_list.txt")
    
    if files_created:
        print(f"[INFO] Files created: {', '.join(files_created)}")
    else:
        print("[WARN] No query files were created")
    
    print("=" * 70)

    return {
        "queries_data": queries_data,
        "files_created": files_created,
        "output_folder": str(prd_folder),
    }


def find_project_folder(pname: str | None = None) -> Path:
    """Find project folder by pname or return latest.
    
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
    parser = argparse.ArgumentParser(description="Generate agent test queries via Claude Agent SDK")
    parser.add_argument("--pname", "-n", default=None, help="PRD project name to find folder")
    parser.add_argument("--max-turns", type=int, default=20, help="Max turns for the agent")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval")
    parser.add_argument("--timeout-secs", type=int, default=3600, help="Overall timeout (default: 3600s/60min)")
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
    print("AGENT QUERY GENERATOR (Agent SDK)")
    print("=" * 70 + "\n")

    try:
        prd_folder = find_project_folder(args.pname)
        print(f"[INFO] Using PRD folder: {prd_folder.name}\n")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        result = asyncio.run(
            generate_agent_queries_agent(
                prd_folder,
                max_turns=args.max_turns,
                heartbeat_secs=args.heartbeat_secs,
                overall_timeout=args.timeout_secs,
            )
        )

        print("\n[SUMMARY]")
        
        if result['queries_data']:
            queries = result['queries_data']
            print(f"✓ Project: {queries.get('project_name', 'N/A')}")
            print(f"✓ Total Queries: {queries.get('total_queries', 0)}")
            
            # Difficulty distribution
            difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
            for q in queries.get('queries', []):
                diff = q.get('difficulty', 'medium')
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            
            print("\n[DIFFICULTY DISTRIBUTION]")
            print(f"  - Easy: {difficulty_counts.get('easy', 0)} queries")
            print(f"  - Medium: {difficulty_counts.get('medium', 0)} queries")
            print(f"  - Hard: {difficulty_counts.get('hard', 0)} queries")
            
            # Sample queries
            print("\n[SAMPLE QUERIES]")
            for q in queries.get('queries', [])[:3]:
                print(f"  {q['id']}. [{q['difficulty'].upper()}] {q['query'][:80]}...")
        
        if result['files_created']:
            print(f"\n[OUTPUT FILES]")
            for file in result['files_created']:
                print(f"  - {file}")
        
        print("\n[NEXT STEPS]")
        print("1. Review agent_queries.json for the complete query list")
        print("2. Check agent_queries.md for human-readable format")
        print("3. Use these queries to test your AI navigation agent")

    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
