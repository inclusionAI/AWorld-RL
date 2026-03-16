#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate website layout HTML designs via Claude Agent SDK.

This script reads a PRD from s1 output and generates multiple design variations
for each core page using the agent SDK with streaming output.
"""

import os
import sys
import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
import argparse

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
# Design generation prompt template
# ---------------------------------------------------------------------------
LAYOUT_DESIGN_PROMPT = """
You are a senior front-end designer tasked with creating {num_variations} design variations for website pages.

**Your Task:**

1. Read the PRD file from: {prd_file_path}
2. Analyze the PRD content to understand the website requirements
3. Identify the core pages (or use these if specified: {page_list})
4. Create {num_variations} design variations for EACH core page
5. Save each variation as a separate HTML file in the design_iterations folder


**Design Requirements:**

1. **Create Complete HTML Pages**: Each variation should be a fully functional, standalone HTML page
2. **Use Tailwind CSS CDN**: Include via CDN for all styling
3. **Responsive Design**: Must work perfectly on mobile, tablet, and desktop
4. **No Images**: Use CSS to create placeholder visuals (colored divs, gradients, etc.)
5. **Icons**: You may use lightweight icons or placeholders (inline SVG, Heroicons/FontAwesome CDN, emoji).
6. **Black & White Text Only**: All text must be either black or white
7. **4pt/8pt Spacing System**: All margins, padding, and sizes must be multiples of 4pt or 8pt
8. **Design Style**: Elegant minimalism with functional design, subtle shadows, modular cards, refined rounded corners
9. **Reference Real Websites**: If needed, you may refer to the layout and design style of real-world websites for inspiration.

**File Naming Convention:**

Use lowercase with underscores, format: pagename_v1.html, pagename_v2.html, etc.

Examples:
- home_v1.html, home_v2.html, home_v3.html
- profile_v1.html, profile_v2.html
- feed_v1.html, feed_v2.html

**HTML Template Structure:**

Each HTML file should follow this structure:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title - Variation N</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <!-- Your responsive design here -->
</body>
</html>

**Workflow:**

1. Use the Read tool to read the PRD file
2. Analyze and identify core pages (typically 5 main pages)
3. For each page, create {num_variations} HTML files with different design approaches:
   - Variation 1: Traditional/Classic approach
   - Variation 2: Modern/Bold approach  
   - Variation 3: Minimal/Clean approach
4. Use the Write tool to save each HTML file to: design_iterations/pagename_vN.html
5. Create a README.md index file listing all generated designs

Start by reading the PRD and then generate all the design files.
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


async def generate_website_layout_agent(
    prd_folder: Path,
    num_variations: int = 3,
    pages_to_design: list[str] | None = None,
    max_turns: int = 30,
    heartbeat_secs: int = 60,
    overall_timeout: int = 1800,
) -> dict:
    """Generate website layout HTML designs via agent."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    print("=" * 70)
    print("GENERATING WEBSITE DESIGN VARIATIONS FROM PRD")
    print("=" * 70)

    # Find PRD file path
    full_prd_path = prd_folder / "02_prd_full_draft.md"
    outline_path = prd_folder / "01_prd_outline.md"
    
    if full_prd_path.exists():
        prd_file_path = str(full_prd_path)
        print(f"[INFO] PRD file: {full_prd_path.name}")
    elif outline_path.exists():
        prd_file_path = str(outline_path)
        print(f"[INFO] PRD file: {outline_path.name}")
    else:
        raise FileNotFoundError(f"No PRD files found in {prd_folder}")

    # Prepare page list hint
    page_list_hint = "any core pages you identify from the PRD"
    if pages_to_design:
        page_list_hint = ", ".join(pages_to_design)

    print(f"[INFO] Generating {num_variations} variations per page")
    print(f"[INFO] Pages: {page_list_hint}\n")

    # Build prompt - let agent read the file itself
    design_prompt = LAYOUT_DESIGN_PROMPT.format(
        prd_file_path=prd_file_path,
        num_variations=num_variations,
        page_list=page_list_hint,
    )

    # Agent options - give it file access and allow it to create files
    agent_options = ClaudeAgentOptions(
        cwd=str(prd_folder),
        permission_mode="bypassPermissions",
        allowed_tools="*",
        max_turns=max_turns,
        system_prompt="""You are a senior front-end designer and developer.

Your task:
1. Read the PRD file using the Read tool
2. Analyze it to understand the website requirements and identify core pages
3. For each page, create multiple HTML design variations
4. Use the Write tool to save each HTML file to the design_iterations/ folder
5. Create a README.md index file

Each HTML file should be:
- Complete and standalone with Tailwind CSS CDN
- Fully responsive (mobile, tablet, desktop)
- Black/white text only, no images
- Following 4pt/8pt spacing system
- Elegant minimalist design with subtle shadows and rounded corners

Work systematically: read PRD → identify pages → create variations → save files.""",
    )

    print("[STEP] Agent will now read PRD and generate all design files...")
    print("-" * 70)
    
    response = await run_agent(
        design_prompt,
        agent_options,
        label="Design Generation",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )

    # Check if design_iterations folder was created
    design_folder = prd_folder / "design_iterations"
    if not design_folder.exists():
        print("[WARN] design_iterations folder was not created by agent")
        design_folder.mkdir(exist_ok=True)

    # List generated files
    html_files = list(design_folder.glob("*.html"))
    readme_file = design_folder / "README.md"

    print("\n" + "=" * 70)
    print("[SUCCESS] Agent completed design generation!")
    print(f"[INFO] Design folder: {design_folder}")
    print(f"[INFO] HTML files: {len(html_files)}")
    if readme_file.exists():
        print(f"[INFO] Index: README.md")
    print("=" * 70)

    return {
        "design_folder": str(design_folder),
        "html_files": [f.name for f in html_files],
        "total_files": len(html_files),
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
    parser = argparse.ArgumentParser(description="Generate website layout designs via Claude Agent SDK")
    parser.add_argument("--pname", "-n", default=None, help="PRD project name to find folder")
    parser.add_argument("--num-variations", "-v", type=int, default=3, help="Number of design variations per page")
    parser.add_argument("--pages", "-p", default=None, help="Comma-separated page names (or let agent decide)")
    parser.add_argument("--max-turns", type=int, default=30, help="Max turns for the agent")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval")
    parser.add_argument("--timeout-secs", type=int, default=3600, help="Overall timeout (default: 1800s/30min)")
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
    print("WEBSITE DESIGN GENERATOR (Agent SDK)")
    print("=" * 70 + "\n")

    try:
        prd_folder = find_project_folder(args.pname)
        print(f"[INFO] Using PRD folder: {prd_folder.name}\n")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Parse pages if provided
    pages_to_design = None
    if args.pages:
        pages_to_design = [p.strip() for p in args.pages.split(",")]
        print(f"[INFO] Custom pages: {', '.join(pages_to_design)}\n")

    try:
        result = asyncio.run(
            generate_website_layout_agent(
                prd_folder,
                num_variations=args.num_variations,
                pages_to_design=pages_to_design,
                max_turns=args.max_turns,
                heartbeat_secs=args.heartbeat_secs,
                overall_timeout=args.timeout_secs,
            )
        )

        print("\n[SUMMARY]")
        print(f"✓ Design folder: {result['design_folder']}")
        print(f"✓ HTML files generated: {result['total_files']}")
        if result['html_files']:
            print(f"✓ Files: {', '.join(result['html_files'][:5])}")
            if len(result['html_files']) > 5:
                print(f"  ... and {len(result['html_files']) - 5} more")
        print("\n[NEXT STEPS]")
        print("1. Open design_iterations/README.md to see all designs")
        print("2. Open any .html file in your browser to preview")
        print("3. Compare variations and choose your favorite")

    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
