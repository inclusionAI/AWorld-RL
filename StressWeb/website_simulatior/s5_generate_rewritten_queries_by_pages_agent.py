#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate page-level trajectory documents via Claude Agent SDK.

For each test query in agent_queries_combined.json (original + s33 generated),
the agent creates a detailed page-by-page data flow document showing database
snapshots, page data, and user actions at each step.
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
# Prompt template for single query trajectory
# ---------------------------------------------------------------------------
AGENT_QUERY_TRAJECTORY_PROMPT = """
You are a UX expert and database specialist. Transform this user query into a detailed page-by-page trajectory document.

**Original Query:**
{query_text}

**Query Metadata:**
- Difficulty: {difficulty}
- Required Capabilities: {capabilities}
- Estimated Steps: {estimated_steps}
- Description: {description}

**PRD Context (first 3000 chars):**
{prd_context}

**Your Task:**
Generate a COMPLETE Markdown document showing the page-level data flow. Structure:

1. **User Goal** - What the user wants to accomplish
2. **Page Sequence** (Page_0 → Page_1 → ... → Page_N)
   - For each page include:
     - pagename (e.g., search.html, results.html)
     - db_snapshot (database table states BEFORE actions on this page)
     - pagedata (actual UI elements/data shown)
     - Action_within_page_N (user actions and database changes)
3. **Page Flow Summary** - Complete sequence and statistics
4. **Database Operation Statistics** - Read/write counts
5. **Technical Implementation Points** - Code examples if relevant

**Key Requirements:**
- Use concrete page names based on the task (not generic)
- Show realistic SQL data with IDs, timestamps, relationships
- Include 4-8 pages for medium/hard queries, 2-3 for easy queries
- Distinguish manual actions (❌), automatic actions (✅), optional (⭕)
- Show actual SQL INSERT/UPDATE/DELETE statements
- Be comprehensive but realistic—complete deliverable

**Output Format:**
Generate ONLY the Markdown document. No preamble or explanations. Start with "# Scenario:" heading.

Generate the complete page trajectory document now.
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


async def generate_trajectory_for_query(
    query_data: dict,
    prd_context: str,
    output_folder: Path,
    heartbeat_secs: int = 60,
    overall_timeout: int = 480,
) -> dict:
    """Generate page trajectory document for a single query."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    query_id = query_data["id"]
    query_text = query_data["query"]
    difficulty = query_data.get("difficulty", "N/A")
    capabilities = ", ".join(query_data.get("required_capabilities", []))
    estimated_steps = query_data.get("estimated_steps", "N/A")
    description = query_data.get("description", "N/A")

    print("=" * 70)
    print(f"Query {query_id}: {query_text[:60]}...")
    print("=" * 70)

    prompt = AGENT_QUERY_TRAJECTORY_PROMPT.format(
        query_text=query_text,
        difficulty=difficulty,
        capabilities=capabilities,
        estimated_steps=estimated_steps,
        description=description,
        prd_context=prd_context[:3000],
    )

    agent_options = ClaudeAgentOptions(
        cwd=str(output_folder),
        permission_mode="bypassPermissions",
        allowed_tools="*",
        max_turns=12,
        system_prompt="""You are a UX expert and database specialist. Generate page trajectory documents.

Your task: Create ONE complete Markdown document for a user query showing:
- Page sequence with database snapshots
- UI data at each page
- User actions and database changes
- Statistics (pages, operations, tables)

Requirements:
- Output ONLY Markdown document (no preamble)
- Use concrete page names
- Show realistic data with IDs, timestamps
- Include 4-8 pages (based on complexity)
- Mark actions: ❌ manual, ✅ automatic, ⭕ optional
- Be thorough and complete

Start outputting Markdown now.
""",
    )

    print("[STEP] Agent generating page trajectory...")
    print("-" * 70)

    response = await run_agent(
        prompt,
        agent_options,
        label=f"Query {query_id} Trajectory",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )

    # Save to file
    output_file = output_folder / f"query_{query_id}_page_trajectory.md"
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
        "query_text": query_text[:50],
        "file": str(output_file),
        "success": True,
    }


async def generate_trajectories_batch(
    queries: list,
    prd_context: str,
    output_folder: Path,
    num_queries: int | None = None,
    heartbeat_secs: int = 60,
    query_timeout: int = 480,
) -> dict:
    """Generate trajectories for multiple queries."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    queries_to_process = queries[: num_queries if num_queries else len(queries)]

    print("=" * 70)
    print(f"GENERATING PAGE TRAJECTORIES ({len(queries_to_process)} queries)")
    print(f"Processing queries from: agent_queries_combined.json (s3 + s33)")
    print("=" * 70 + "\n")

    results = []
    for i, query_data in enumerate(queries_to_process, 1):
        print(f"\n{'='*70}")
        print(f"[PROGRESS] Query {i}/{len(queries_to_process)}")
        print(f"[QUERY ID] {query_data['id']}")
        print(f"[QUERY TEXT] {query_data.get('query', '')[:80]}...")
        print(f"{'='*70}\n")

        try:
            result = await generate_trajectory_for_query(
                query_data,
                prd_context,
                output_folder,
                heartbeat_secs=heartbeat_secs,
                overall_timeout=query_timeout,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Query {query_data['id']}: {e}")
            results.append({"query_id": query_data["id"], "success": False, "error": str(e)})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r.get("success"))
    print(f"\n✓ Successfully generated: {successful}/{len(queries_to_process)}")

    if successful > 0:
        print("\n[FILES CREATED]")
        for r in results:
            if r.get("success"):
                print(f"  - Query {r['query_id']}: {r['query_text']}...")
                print(f"    → {Path(r['file']).name}")

    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\n⚠️  Failed: {len(failed)} queries")
        for r in failed:
            print(f"  - Query {r['query_id']}: {r.get('error', 'Unknown error')}")

    print(f"\n[OUTPUT LOCATION]")
    print(f"  {output_folder}")
    print("=" * 70 + "\n")

    return {
        "total": len(queries_to_process),
        "successful": successful,
        "failed": len(failed),
        "results": results,
        "output_folder": str(output_folder),
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


def load_queries(prd_folder: Path) -> tuple[list, str]:
    """Load queries and PRD context.
    
    Priority:
    1. agent_queries_combined.json (s33 output with all queries)
    2. agent_queries.json (fallback to s3 output)
    """
    # Try to load combined queries first (from s33)
    combined_queries_file = prd_folder / "agent_queries_combined.json"
    queries_file = prd_folder / "agent_queries.json"
    
    queries_data = None
    source_file = None
    
    if combined_queries_file.exists():
        try:
            queries_data = json.loads(combined_queries_file.read_text(encoding="utf-8"))
            source_file = "agent_queries_combined.json (s33 output)"
            print(f"[INFO] Loaded combined queries from: {source_file}")
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse agent_queries_combined.json: {e}")
            print(f"[INFO] Falling back to agent_queries.json...")
    
    if queries_data is None and queries_file.exists():
        try:
            queries_data = json.loads(queries_file.read_text(encoding="utf-8"))
            source_file = "agent_queries.json (s3 output)"
            print(f"[INFO] Loaded queries from: {source_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse agent_queries.json: {e}")
    
    if queries_data is None:
        raise FileNotFoundError(f"Neither agent_queries_combined.json nor agent_queries.json found in {prd_folder}")
    
    queries = queries_data.get("queries", [])
    if not queries:
        raise ValueError("No queries found in queries file")

    # Load PRD context
    full_prd = prd_folder / "02_prd_full_draft.md"
    outline_prd = prd_folder / "01_prd_outline.md"

    prd_context = ""
    if full_prd.exists():
        prd_context = full_prd.read_text(encoding="utf-8")
        print(f"[INFO] Loaded full PRD: {len(prd_context)} chars")
    elif outline_prd.exists():
        prd_context = outline_prd.read_text(encoding="utf-8")
        print(f"[INFO] Loaded PRD outline: {len(prd_context)} chars")

    return queries, prd_context


def parse_args():
    parser = argparse.ArgumentParser(description="Generate page trajectories for all queries (s3 + s33 combined)")
    parser.add_argument("--pname", "-n", default=None, help="PRD project name to find folder")
    parser.add_argument("--queries", "-q", type=int, default=None, help="Number of queries to process (default: all)")
    parser.add_argument("--all", action="store_true", help="Process all queries (default behavior)")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval")
    parser.add_argument("--query-timeout", type=int, default=1200, help="Timeout per query (default: 1200s)")
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
    print("PAGE TRAJECTORY GENERATOR (Agent SDK)")
    print("=" * 70 + "\n")

    try:
        prd_folder = find_project_folder(args.pname)
        print(f"[INFO] Using PRD folder: {prd_folder.name}\n")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        queries, prd_context = load_queries(prd_folder)
        print(f"[INFO] Loaded {len(queries)} queries\n")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Determine number of queries to process
    # Default: process all queries
    if args.queries is not None:
        num_to_process = min(args.queries, len(queries))
        print(f"[INFO] Processing {num_to_process} queries (limited by --queries {args.queries})")
    else:
        num_to_process = len(queries)
        print(f"[INFO] Processing all {num_to_process} queries (from s3 + s33)")

    try:
        result = asyncio.run(
            generate_trajectories_batch(
                queries,
                prd_context,
                prd_folder,
                num_queries=num_to_process,
                heartbeat_secs=args.heartbeat_secs,
                query_timeout=args.query_timeout,
            )
        )

        print("\n[FINAL STATS]")
        print(f"Processed: {result['successful']}/{result['total']} successful")
        print(f"Output folder: {result['output_folder']}")

    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
