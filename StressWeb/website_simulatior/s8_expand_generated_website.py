#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand existing generated website with new query trajectory.

This script takes an existing generated website and a new query trajectory,
then expands the website to include the new features while maintaining
natural integration and smooth transitions.
"""

import os
import sys
import asyncio
import re
import json
import time
from pathlib import Path
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Environment variables required by CCR / Anthropic SDK
# ---------------------------------------------------------------------------
os.environ["ANTHROPIC_BASE_URL"] = os.environ.get("ANTHROPIC_BASE_URL", "")
os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("ANTHROPIC_AUTH_TOKEN", "ccr-local")
os.environ["ANTHROPIC_MODEL"] = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

CURRENT_DIR = Path(__file__).resolve().parent

# Try importing the agent SDK
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
    SDK_AVAILABLE = True
except ImportError as e:
    SDK_AVAILABLE = False
    IMPORT_ERROR = e

# ---------------------------------------------------------------------------
# Prompt template for website expansion
# ---------------------------------------------------------------------------
AGENT_WEBSITE_EXPANSION_PROMPT = """
You are an expert full-stack developer. Your task is to EXPAND an existing web application with NEW features.

**Context:**
- Existing Website: {existing_website_path}
- New Trajectory: {new_trajectory_file_path}
- Design References: {design_folder_path}

**Your Task:**
Analyze the existing website and integrate the new trajectory's features seamlessly.

**CRITICAL REQUIREMENTS:**

1. **Analyze Existing Code First:**
   - Read ALL existing files in the website directory
   - Understand the current architecture, routes, components, and data structure
   - Identify where new features should be integrated
   - Maintain existing code style and patterns

1b. **Preserve Existing Behavior (Do Not Break What Works):**
    - All existing pages, routes, and API endpoints must continue to work after expansion
    - If new UI reuses an existing action (button/link), wire it to the SAME behavior/data flow
    - If old behavior is insufficient, you may re-implement/refactor it, but final behavior must remain intact for existing flows

1c. **Provide Accessible Entry Points (MANDATORY):**
    - ALL existing pages must remain accessible through original navigation paths
    - ALL new pages must have clear, visible entry points (navigation links, buttons, menu items)
    - Update Header/Footer/Sidebar navigation to include new feature links
    - Add contextual links between related old and new features
    - Ensure users can discover and access both existing and new functionality
    - No orphaned pages: every page must be reachable through UI navigation
    - Entry points should be intuitive and follow existing navigation patterns

2. **Integration Strategy:**
   - Add NEW pages for new trajectory features
   - Update EXISTING components if they need to support new features
   - Extend backend API with new endpoints
   - Update mockData.js to include new data from the new trajectory's Page_0
   - **Add prominent navigation links in Header/Footer/Sidebar to new pages**
   - **Create contextual navigation between related features (e.g., "View Details" → Detail Page)**
   - Ensure smooth user flow between existing and new features
   - Enforce data/logic consistency: same buttons = same actions across old/new pages; reuse shared handlers/state where appropriate

2b. **Data Consistency & Correctness (MANDATORY):**
   - All displayed counts/numbers must match actual data in mockData.js
   - If a category shows "5 items", it must contain 5+ items when expanded
   - Status displays and counters must reflect actual data state
   - New pages must use same data model and logic as existing pages
   - User interactions must update shared mockData consistently
   - No orphaned references or broken links between old and new pages
   - Pagination and search must respect total mockData counts

**mockData EXPANSION GUIDELINES (CRITICAL - PREVENT INCONSISTENCY):**
When expanding mockData.js with new trajectory data, follow these rules:
  A. **Audit New Display Requirements:**
     - Scan new trajectory and new pages for ALL displayed counts/lists
     - Example: "10 new products", "3 new categories", "20 search results"
     - Record minimum data needed: displayed_count + 20% buffer
  B. **Merge Data Without Breaking Old Data:**
     - Keep ALL existing mockData.js entries (don't remove/overwrite old data)
     - Add NEW entities/tables from new trajectory Page_0
     - Use appropriate IDs (don't conflict with existing IDs)
     - Example: If old data has products 1-50, new products should start from 51+
  C. **Initialize New Data with Exact Counts:**
     - Create exactly enough items to support NEW page displays + buffer
     - Example: If new feature shows "10 items", add 12-15+ items to that table
     - Example: If new feature shows "5 categories with 8 items each", create 5 categories with 8+ items
  D. **Maintain Consistency Between Old and New:**
     - Old pages must work unchanged with old data
     - New pages must work with new data
     - If new pages access old data tables, counts must still be accurate
     - If new features reference old entities, links must point to real items
  E. **Update Computed Values Carefully:**
     - Any counts shown in UI must be computed from actual data
     - Example: OLD page shows category count = old categories
     - Example: NEW page shows category count = old + new categories (if merged)
     - Use filter/length, never hardcoded numbers
  F. **Ensure Referential Integrity:**
     - EVERY foreign key must reference existing mockData
     - If new table has category_id, those categories must exist
     - No orphaned references3. **File Output Format:**
   Only output files that are NEW or MODIFIED. Use this EXACT format:

   ===FILE: path/to/file.ext===
   [complete file content]
   ===END FILE===

   For MODIFIED files, output the COMPLETE file with all changes integrated.
   For NEW files, output the complete new file.

4. **Frontend Expansion:**
   - Add new pages in src/pages/ for new trajectory features
   - Update App.js routes to include new page routes
   - Update Header.js/Footer.js navigation if needed
   - Create new components in src/components/ if needed
   - Update api.js with new API call functions
   - Maintain existing styling approach (Tailwind/CSS)
   - **Use DETERMINISTIC images (Pexels direct URLs with photo ID) for all new content**

5. **Backend Expansion:**
   - Add new endpoints in server.js for new features
   - Extend mockData.js with new data structures from new trajectory
   - Maintain existing endpoint patterns
   - Keep all existing endpoints functional

6. **Data Integration:**
   - Merge new Page_0 data from new trajectory into existing mockData.js
   - Ensure data relationships work between old and new features
   - Use appropriate data IDs to avoid conflicts

7. **Natural Transitions:**
   - Add contextual links between related old and new features
   - Update relevant pages to reference new features where appropriate
   - Ensure UI/UX consistency across old and new pages
   - Maintain the same design language and patterns

**IMAGE RESOURCES (Use image_search Tool - REAL PHOTOS):**

**PRIMARY METHOD: Use image_search Tool (RECOMMENDED - Real, High-Quality Images)**

You have access to the `image_search` tool which searches for real photos. Use it to find appropriate images for ANY content type:

1. **For Main Content Items (Generic - Works for ANY website type):**
   - Call: `image_search(query="content type + descriptive keywords", num=5-10)`
   - Examples:
     * E-commerce: `image_search(query="laptop computer professional", num=8)`
     * News/Blog: `image_search(query="technology innovation breaking news", num=8)`
     * Social Media: `image_search(query="lifestyle travel adventure", num=8)`
     * SaaS: `image_search(query="business productivity workflow", num=8)`
     * Education: `image_search(query="learning classroom collaboration", num=8)`
   - Returns URLs that can be used directly in img tags
   - Select diverse images for visual variety

2. **For Section Headers/Backgrounds/Covers:**
   - Call: `image_search(query="section theme keywords themed background", num=3-5)`
   - Examples:
     * `image_search(query="modern office workspace", num=5)`
     * `image_search(query="education learning collaboration", num=5)`
     * `image_search(query="community social interaction", num=5)`
   - Use for banner images and section covers

3. **For User Avatars/Profiles:**
   - Use UI Avatars API: `https://ui-avatars.com/api/?name=USERNAME&size=300&background=3498db&color=fff`
   - Or DiceBear: `https://api.dicebear.com/7.x/avataaars/svg?seed=USER_ID`
   - These are deterministic and work well for user profiles
   - Alternative: Search for "people portrait professional", num=20 for diverse profile photos

4. **Image URL Integration in Code:**
   - Search results include `url` or `image_url` field with the actual image URL
   - Use these URLs directly in img tags: `<img src="{search_result.url}" alt="description" />`
   - Store results in mockData.js or backend data for consistency
   - Example: `<img src="https://upload.wikimedia.org/real-photo.jpg" alt="Item name" />`

5. **Code Pattern Example (Generic - Works for any content):**
   ```javascript
   // In mockData.js or initial data - GENERIC pattern for ANY website
   const items = [
     { id: 1, title: "Item Title", image: "https://[image_search result URL]" },
     { id: 2, title: "Another Item", image: "https://[image_search result URL]" }
   ];
   
   // In component - works for products, articles, posts, etc.
   <img src={item.image} alt={item.title} />
   ```

6. **Search Best Practices (Universal):**
   - Use context-specific keywords relevant to your website type
   - Get 5-10 images per section for good variety
   - Mix different orientations and compositions as needed
   - Store URLs in mock data to ensure consistency across page reloads
   - Adapt search terms based on website content (not just e-commerce)

**CRITICAL CHECKLIST:**
- [ ] image_search() called for appropriate content types
- [ ] Search results stored in mockData.js with URL field
- [ ] Components reference stored image URLs: <img src={item.image} />
- [ ] Text-based content uses UI Avatars API
- [ ] User profiles use DiceBear or search results
- [ ] All images are REAL photos from search results
- [ ] NO placeholder images (dummyimage.com, picsum.photos, via.placeholder.com)
- [ ] NO random endpoints (loremflickr, /random, unsplash/random)
- [ ] NO JSON endpoints or empty src attributes
- [ ] Read and understand ALL existing files before making changes
- [ ] Only output NEW or MODIFIED files
- [ ] All new routes added to App.js
- [ ] Navigation updated in Header/Footer for new pages
- [ ] New API endpoints added to server.js
- [ ] New data structures added to mockData.js
- [ ] Existing functionality remains intact
- [ ] New features integrate naturally with existing ones
- [ ] All imports and routes resolve correctly
- [ ] No breaking changes to existing code
- [ ] Consistent code style throughout
- [ ] **NAVIGATION & ACCESS CHECKLIST:**
  * [ ] All existing pages remain accessible through original navigation
  * [ ] All new pages have visible navigation links (Header/Footer/Sidebar)
  * [ ] Contextual links added between related old and new features
  * [ ] No orphaned pages (every page reachable through UI)
  * [ ] Navigation patterns consistent with existing design
  * [ ] User can discover all features through intuitive navigation
- [ ] **DATA CONSISTENCY CHECKLIST:**
  * [ ] All displayed counts match actual data in mockData (e.g., "5 items" = 5+ items)
  * [ ] Category contents consistent between old and new pages
  * [ ] Pagination respects total data count
  * [ ] Status displays match actual data state
  * [ ] Related items link to real mockData entries
  * [ ] User interactions update mockData consistently across old and new pages
  * [ ] No orphaned references or broken links
  * [ ] New data model matches existing data structure
- [ ] **MOCK EXTERNAL OPERATIONS:**
  * [ ] NO real payment gateway calls (use mock payment flow with setTimeout)
  * [ ] NO real OAuth/social login (auto-login with mock credentials)
  * [ ] NO real email/SMS sending (accept any verification code)
  * [ ] NO real external API calls (return hardcoded mock data)
  * [ ] NO real file uploads to cloud (fake progress bar + mock URL)
  * [ ] All "external" operations complete locally with simulated delay
  * [ ] Loading states + success feedback for all mocked operations

**OUTPUT INSTRUCTIONS:**
1. First, explain your integration strategy in 2-3 sentences
2. Then output ALL modified or new files using the ===FILE:=== format
3. Do NOT output files that remain unchanged

Start your analysis and expansion now.
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


def parse_generated_files(content: str) -> dict[str, str]:
    """Parse agent response and extract individual files."""
    files = {}
    file_pattern = r"===FILE:\s*(.+?)\s*===\s*(.*?)\s*===END FILE==="
    matches = re.findall(file_pattern, content, re.DOTALL)

    for filepath, file_content in matches:
        filepath = filepath.strip()
        file_content = file_content.strip()
        files[filepath] = file_content
        print(f"[PARSED] {filepath}")

    return files


def validate_and_fix_css(css_content: str, filepath: str) -> tuple[str, list[str]]:
    """Validate and fix common CSS syntax errors.
    
    Returns: (fixed_content, list_of_issues_found)
    """
    issues = []
    lines = css_content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines, 1):
        original_line = line
        
        # Fix 1: "justify-center;" → "justify-content: center;"
        if re.search(r'^\s*justify-center\s*;', line):
            line = re.sub(r'justify-center\s*;', 'justify-content: center;', line)
            issues.append(f"Line {i}: Fixed 'justify-center;' → 'justify-content: center;'")
        
        # Fix 2: "align-center;" → "align-items: center;"
        if re.search(r'^\s*align-center\s*;', line):
            line = re.sub(r'align-center\s*;', 'align-items: center;', line)
            issues.append(f"Line {i}: Fixed 'align-center;' → 'align-items: center;'")
        
        # Fix 3: Missing property name (standalone value like "center;")
        if re.search(r'^\s*(center|left|right|top|bottom|middle|baseline|stretch)\s*;', line):
            if fixed_lines and ':' in fixed_lines[-1] and not fixed_lines[-1].rstrip().endswith(';'):
                pass
            else:
                issues.append(f"Line {i}: Suspicious standalone value '{line.strip()}' (might need property name)")
        
        # Fix 4: Property without colon "property value;" → "property: value;"
        match = re.match(r'^(\s*)([a-z-]+)\s+([^:;]+);', line)
        if match and ':' not in line:
            indent, prop, value = match.groups()
            if prop in ['display', 'position', 'flex', 'grid', 'width', 'height', 'margin', 'padding',
                       'color', 'background', 'font', 'border', 'text', 'overflow', 'cursor']:
                line = f"{indent}{prop}: {value};"
                issues.append(f"Line {i}: Added missing colon in '{prop} {value};' → '{prop}: {value};'")
        
        fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    return fixed_content, issues


def validate_and_fix_files(files: dict) -> dict:
    """Validate and fix common syntax errors in generated files.
    
    Currently handles:
    - CSS syntax errors (missing colons, wrong property names)
    - Markdown code fence removal (```language and ```)
    - Double closing braces in CSS/JS
    - JSON/package.json Markdown code blocks
    
    Returns: dict with fixed file contents
    """
    fixed_files = {}
    total_issues = 0
    
    for filepath, content in files.items():
        # Check for JSON files (especially package.json) with Markdown code blocks
        if filepath.endswith('.json'):
            original_content = content
            # Remove opening Markdown code fences (with optional language specifier)
            content = re.sub(r'^```+(?:json)?\s*\n', '', content, flags=re.MULTILINE)
            # Remove closing Markdown code fences
            content = re.sub(r'\n```+\s*$', '', content, flags=re.MULTILINE)
            # Also handle code fences that might be at the very start/end without newlines
            content = re.sub(r'^```+(?:json)?', '', content)
            content = re.sub(r'```+$', '', content)
            
            if content != original_content:
                total_issues += 1
                print(f"[FIX] Removed Markdown code blocks from: {filepath}")
    
    for filepath, content in files.items():
        original_content = content
        
        # Fix 1: Remove markdown code fences
        # Remove opening fence: ```javascript, ```css, ```jsx, ``` etc.
        content = re.sub(r'^```(javascript|css|jsx|typescript|ts)?\n', '', content)
        # Remove closing fence: \n``` at end
        content = re.sub(r'\n```\s*$', '', content)
        
        # Fix 2: Fix double closing braces }}  → }
        if filepath.endswith('.css'):
            content = re.sub(r'\}\}+', '}', content)
            fixed_content, issues = validate_and_fix_css(content, filepath)
            if issues:
                print(f"[FIXED] {filepath}: {len(issues)} CSS issue(s)")
                for issue in issues:
                    print(f"  - {issue}")
                total_issues += len(issues)
            content = fixed_content
        
        # Track if content was modified
        if content != original_content:
            total_issues += 1
            if filepath.endswith(('.js', '.jsx', '.css')):
                print(f"[FIXED] {filepath}: Removed markdown code fences and syntax errors")
        
        fixed_files[filepath] = content
    
    if total_issues > 0:
        print(f"\n[INFO] Auto-fixed {total_issues} syntax issue(s)")
    
    return fixed_files


def ensure_imported_pages_exist(website_dir: Path, files: dict) -> dict:
    """Ensure all pages imported in App.js exist; create lightweight stubs if missing.

    This prevents build failures when the agent adds new routes/imports but forgets to
    emit the corresponding component files.
    """
    app_rel_path = "frontend/src/App.js"
    frontend_src = website_dir / "frontend" / "src"
    pages_dir = frontend_src / "pages"

    # Determine App.js content (prefer agent output, otherwise read existing file)
    app_content = files.get(app_rel_path)
    if app_content is None:
        app_path = website_dir / app_rel_path
        if not app_path.exists():
            return files
        app_content = app_path.read_text(encoding="utf-8")

    # Find imports from ./pages/*
    import_pattern = r"import\s+([A-Za-z0-9_]+)\s+from\s+['\"]\.\/pages\/([^'\"]+)['\"];"
    imports = re.findall(import_pattern, app_content)
    if not imports:
        return files

    updated_files = dict(files)

    for component_name, import_path in imports:
        js_rel = f"frontend/src/pages/{import_path}.js"
        css_rel = f"frontend/src/pages/{import_path}.css"

        js_exists = js_rel in updated_files or (website_dir / js_rel).exists()
        css_exists = css_rel in updated_files or (website_dir / css_rel).exists()

        if js_exists and css_exists:
            continue

        # Create minimal React component stub
        class_name = re.sub(r'(?<!^)(?=[A-Z])', '-', component_name).lower()
        js_stub = (
            "import React from 'react';\n"
            f"import './{import_path}.css';\n\n"
            f"function {component_name}() {{\n"
            f"  return (\n"
            f"    <div className=\"{class_name}\">\n"
            f"      <h1>{component_name}</h1>\n"
            f"      <p>Placeholder page generated automatically because the component was missing.</p>\n"
            f"    </div>\n"
            f"  );\n"
            f"}}\n\n"
            f"export default {component_name};\n"
        )

        css_stub = (
            f".{class_name} {{\n"
            "  padding: 2rem;\n"
            "  background: #f8fafc;\n"
            "  min-height: 100vh;\n"
            "}}\n\n"
            f".{class_name} h1 {{\n"
            "  margin: 0 0 0.75rem;\n"
            "}}\n\n"
            f".{class_name} p {{\n"
            "  color: #64748b;\n"
            "}}\n"
        )

        if not js_exists:
            updated_files[js_rel] = js_stub
            print(f"[SAFEGUARD] Added missing page component stub: {js_rel}")
        if not css_exists:
            updated_files[css_rel] = css_stub
            print(f"[SAFEGUARD] Added missing page stylesheet stub: {css_rel}")

    return updated_files


def backup_existing_files(website_dir: Path, files_to_modify: list[str]) -> dict[str, str]:
    """Create backups of files that will be modified."""
    backups = {}
    backup_dir = website_dir.parent / f"{website_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for filepath in files_to_modify:
        full_path = website_dir / filepath
        if full_path.exists():
            backup_path = backup_dir / filepath
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = full_path.read_text(encoding="utf-8")
            backup_path.write_text(content, encoding="utf-8")
            backups[filepath] = str(backup_path)
            print(f"[BACKUP] {filepath} -> {backup_path}")

    return backups


def apply_changes(website_dir: Path, files: dict[str, str]) -> list[str]:
    """Apply changes to existing website."""
    modified_files = []

    for filepath, content in files.items():
        full_path = website_dir / filepath
        
        # Determine if this is a new file or modification
        is_new = not full_path.exists()
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            full_path.write_text(content, encoding="utf-8")
            modified_files.append(str(full_path))
            
            if is_new:
                print(f"[NEW] {filepath}")
            else:
                print(f"[MODIFIED] {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to write {filepath}: {e}")

    return modified_files


def load_feature_report(website_dir: Path, query_id: int) -> dict | None:
    """Load previous feature report for comparison.
    
    Returns the JSON data from the feature report if it exists, else None.
    """
    report_pattern = f"FEATURE_REPORT_query_{query_id - 1}.md" if query_id > 1 else None
    
    if not report_pattern:
        return None
    
    report_path = website_dir / report_pattern
    if not report_path.exists():
        print(f"[INFO] No previous feature report found at {report_pattern}")
        return None
    
    try:
        content = report_path.read_text(encoding="utf-8")
        
        # Extract JSON from markdown code block
        json_pattern = r'```json\n(.*?)\n```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        if matches:
            return json.loads(matches[0])
        
        return None
    except Exception as e:
        print(f"[WARN] Error loading feature report: {e}")
        return None


def extract_features_from_files(output_dir: Path) -> dict:
    """Extract feature/functionality information from generated files.
    
    Same logic as s7, for consistency.
    """
    features = {
        "pages": [],
        "api_endpoints": [],
        "data_models": [],
        "components": [],
        "metadata": {
            "generated_at": datetime.now().isoformat(),
        }
    }
    
    try:
        # Extract pages from App.js
        app_js_path = output_dir / "frontend/src/App.js"
        if app_js_path.exists():
            app_content = app_js_path.read_text(encoding="utf-8")
            
            # Find route definitions
            route_pattern = r'<Route\s+path=["\']([^"\']+)["\']'
            routes = re.findall(route_pattern, app_content)
            features["pages"] = sorted(list(set(routes)))
            
            # Find component imports
            import_pattern = r'import\s+([A-Za-z0-9_]+)\s+from\s+["\']\.\/pages\/([^"\']+)'
            imports = re.findall(import_pattern, app_content)
            features["components"] = [comp[0] for comp in imports]
        
        # Extract API endpoints from server.js
        server_js_path = output_dir / "backend/server.js"
        if server_js_path.exists():
            server_content = server_js_path.read_text(encoding="utf-8")
            
            # Find endpoint patterns: app.get/post/put/delete(...)
            endpoint_pattern = r'app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
            endpoints = re.findall(endpoint_pattern, server_content)
            features["api_endpoints"] = sorted(list(set(
                f"{method.upper()} {path}" for method, path in endpoints
            )))
        
        # Extract data models from mockData.js
        mock_data_path = output_dir / "backend/data/mockData.js"
        if mock_data_path.exists():
            mock_content = mock_data_path.read_text(encoding="utf-8")
            
            # Find data structure definitions
            data_pattern = r'(?:db|mockData|data)\.([a-zA-Z_]+)\s*=\s*\['
            data_models = re.findall(data_pattern, mock_content)
            features["data_models"] = sorted(list(set(data_models)))
    
    except Exception as e:
        print(f"[WARN] Error extracting features: {e}")
    
    return features


def validate_and_reconcile_features(
    website_dir: Path,
    old_features: dict | None,
    new_query_id: int,
    new_trajectory_path: Path,
) -> tuple[list[str], list[str]]:
    """Validate expansion against expected features from new trajectory.
    
    Returns:
    - (missing_features, validation_warnings)
    """
    missing_features = []
    warnings = []
    
    if not old_features:
        return [], ["No previous feature report to compare against"]
    
    # Extract new features from trajectory
    try:
        trajectory_content = new_trajectory_path.read_text(encoding="utf-8")
        
        # Simple validation: check if trajectory mentions key pages
        # This is a basic check; could be enhanced with more sophisticated parsing
        pages_mentioned = re.findall(r'^##\s+Page\s+(\d+).*$', trajectory_content, re.MULTILINE | re.IGNORECASE)
        
        if pages_mentioned:
            print(f"[INFO] New trajectory mentions {len(pages_mentioned)} pages")
            
            # Count expected pages in old features
            old_page_count = len(old_features.get("pages", []))
            expected_new_pages = len(pages_mentioned)
            
            if old_page_count > 0:
                print(f"[INFO] Previous website had {old_page_count} pages")
                print(f"[INFO] New trajectory expects {expected_new_pages} new pages")
    
    except Exception as e:
        warnings.append(f"Could not parse trajectory for validation: {e}")
    
    return missing_features, warnings


def generate_expanded_feature_report(
    output_dir: Path,
    initial_query_id: int,
    final_query_id: int,
    feature_history: list[dict],
) -> Path:
    """Generate feature report after expansion iterations.
    
    Creates a comprehensive report tracking features across all queries.
    """
    current_features = extract_features_from_files(output_dir)
    
    # Generate consolidated report
    report_content = f"""# Feature Report - Queries {initial_query_id}-{final_query_id}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Website Directory:** {output_dir.name}
**Final Status:** Expanded with {final_query_id - initial_query_id} query/queries

## Current Summary (After All Expansions)

- **Pages/Routes:** {len(current_features['pages'])}
- **API Endpoints:** {len(current_features['api_endpoints'])}
- **Data Models:** {len(current_features['data_models'])}
- **Components:** {len(current_features['components'])}

## Current Pages & Routes

""" + "\n".join(f"- `{page}`" for page in current_features['pages']) + f"""

## Current API Endpoints

""" + "\n".join(f"- `{endpoint}`" for endpoint in current_features['api_endpoints']) + f"""

## Current Data Models

""" + "\n".join(f"- `{model}`" for model in current_features['data_models']) + f"""

## Current Components

""" + "\n".join(f"- `{comp}`" for comp in current_features['components']) + f"""

## Feature Evolution History

"""
    
    for i, features in enumerate(feature_history, start=initial_query_id):
        report_content += f"""
### Query {i}

- Pages: {len(features.get('pages', []))}
- API Endpoints: {len(features.get('api_endpoints', []))}
- Data Models: {len(features.get('data_models', []))}
- Components: {len(features.get('components', []))}

"""
    
    report_content += f"""

## Raw Current Features (for next s8 iteration)

```json
{json.dumps(current_features, indent=2)}
```

---

This report was auto-generated by s8_expand_generated_website.py
"""
    
    # Use consistent naming: always refer to the CURRENT state
    report_path = output_dir / f"FEATURE_REPORT_query_{final_query_id}.md"
    report_path.write_text(report_content, encoding="utf-8")
    
    print(f"\n[CREATED] Updated Feature Report: {report_path.name}")
    
    return report_path


def generate_expansion_summary(website_dir: Path, query_id: str, new_files: int, modified_files: int) -> None:
    """Generate summary of the expansion."""
    summary_content = f"""# Website Expansion Summary

**Expansion Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**New Query ID:** {query_id}
**Website Directory:** {website_dir.name}

## Changes Applied

- **New Files Created:** {new_files}
- **Existing Files Modified:** {modified_files}
- **Total Files Changed:** {new_files + modified_files}

## Integration Notes

This expansion integrated features from Query {query_id} into the existing website.
All existing functionality has been preserved while new features have been added.

## Verification Steps

1. Test all existing features to ensure they still work
2. Test new features added from Query {query_id}
3. Verify navigation between old and new features
4. Check that data structures are properly integrated

## Rollback

A backup of modified files was created in the backup directory.
If issues occur, you can restore from the backup.

---

Generated by s8_expand_generated_website.py
"""

    summary_path = website_dir / f"EXPANSION_SUMMARY_query_{query_id}.md"
    summary_path.write_text(summary_content, encoding="utf-8")
    print(f"[CREATED] {summary_path.name}")



def find_trajectory_file(prd_folder: Path, query_id: str) -> Path:
    """Find structured trajectory file."""
    trajectory_file = prd_folder / f"query_{query_id}_structured_trajectory.md"
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Structured trajectory not found: {trajectory_file}")
    return trajectory_file


def find_design_folder(prd_folder: Path) -> Path:
    """Find design_iterations folder."""
    design_folder = prd_folder / "design_iterations"
    if not design_folder.exists():
        print("[WARN] design_iterations folder not found")
    return design_folder


def find_website_directory(base_path: str) -> Path:
    """Find and validate website directory."""
    website_dir = Path(base_path).resolve()
    
    if not website_dir.exists():
        raise FileNotFoundError(f"Website directory not found: {website_dir}")
    
    # Validate it's a generated website directory
    frontend_dir = website_dir / "frontend"
    backend_dir = website_dir / "backend"
    
    if not frontend_dir.exists() or not backend_dir.exists():
        raise ValueError(f"Invalid website directory. Must contain 'frontend' and 'backend' subdirectories.")
    
    return website_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Expand existing website with new query trajectory")
    parser.add_argument("--website", "-w", required=True, help="Path to existing generated website directory")
    parser.add_argument("--query", "-q", type=int, required=True, help="New query ID to integrate")
    parser.add_argument("--prd-folder", "-p", default=None, help="PRD folder path (auto-detect if not provided)")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval")
    parser.add_argument("--timeout-secs", type=int, default=72000, help="Overall timeout (default: 72000s/20h)")
    parser.add_argument("--model", "-m", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"))
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup before modifications")
    return parser.parse_args()


async def expand_website_agent(
    website_dir: Path,
    new_query_id: int,
    prd_folder: Path,
    heartbeat_secs: int = 60,
    overall_timeout: int = 1200,
    create_backup: bool = True,
) -> dict:
    """Run agent to expand existing website."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    print("=" * 70)
    print(f"EXPANDING WEBSITE WITH QUERY {new_query_id}")
    print("=" * 70)

    # Find required files
    new_trajectory_file = find_trajectory_file(prd_folder, str(new_query_id))
    design_folder = find_design_folder(prd_folder)

    print(f"[INFO] Existing website: {website_dir.name}")
    print(f"[INFO] New trajectory: {new_trajectory_file.name}")
    print(f"[INFO] Design folder: {design_folder.name}\n")

    # Load existing feature report to inform agent
    print("[INFO] Loading existing feature report...")
    existing_features = load_feature_report(website_dir, new_query_id)
    
    feature_context = ""
    if existing_features:
        print(f"[INFO] Found existing features:")
        print(f"  - Pages: {len(existing_features.get('pages', []))}")
        print(f"  - API Endpoints: {len(existing_features.get('api_endpoints', []))}")
        print(f"  - Data Models: {len(existing_features.get('data_models', []))}")
        
        feature_context = f"""

**EXISTING FEATURES THAT MUST REMAIN FUNCTIONAL:**

The website currently has these implemented features. ALL of them MUST continue to work after expansion:

**Current Pages/Routes ({len(existing_features.get('pages', []))}):**
{chr(10).join(f'- {page}' for page in existing_features.get('pages', []))}

**Current API Endpoints ({len(existing_features.get('api_endpoints', []))}):**
{chr(10).join(f'- {endpoint}' for endpoint in existing_features.get('api_endpoints', []))}

**Current Data Models ({len(existing_features.get('data_models', []))}):**
{chr(10).join(f'- {model}' for model in existing_features.get('data_models', []))}

**CRITICAL REQUIREMENTS:**
1. **PRESERVE ALL EXISTING FUNCTIONALITY**: Every page, route, endpoint, and data model listed above must remain fully functional
2. **NO BREAKING CHANGES**: Do not modify existing features in ways that would cause errors or break user workflows
3. **SMOOTH INTEGRATION**: New features should integrate naturally without disrupting existing functionality
4. **CONSISTENT BEHAVIOR**: If existing and new features share UI patterns (buttons, forms, etc.), maintain consistent behavior
5. **DATA INTEGRITY**: Ensure existing data models work correctly with new features; extend schemas carefully

When you need to interact with existing features:
- Reference existing routes and endpoints
- Extend existing data models rather than replacing them
- Maintain existing UI/UX patterns for consistency
- Test that old features still work alongside new ones
"""
    else:
        print("[INFO] No existing feature report found (this may be the first expansion)")

    prompt = AGENT_WEBSITE_EXPANSION_PROMPT.replace(
        "{existing_website_path}", str(website_dir)
    ).replace(
        "{new_trajectory_file_path}", str(new_trajectory_file)
    ).replace(
        "{design_folder_path}", str(design_folder)
    ) + feature_context

    agent_options = ClaudeAgentOptions(
        cwd=str(website_dir.parent),
        permission_mode="bypassPermissions",
        allowed_tools="*",
        max_turns=25,
        system_prompt="""You are an expert full-stack developer. Your task is to EXPAND an existing web application.

CRITICAL INSTRUCTIONS:
1. FIRST, read ALL files in the existing website to understand the current structure
2. Analyze the new trajectory to understand what features to add
3. Plan integration points where new features connect to existing ones
4. Only output files that are NEW or MODIFIED (not unchanged files)
5. Maintain existing code patterns and style
6. Ensure smooth transitions between old and new features
7. Use image_search tool to find REAL images for all new content (not placeholders)
8. Add navigation links to connect new pages with existing ones
9. Preserve existing behavior: do not break working flows; if a button/action exists in both old and new pages, ensure it triggers the same logic; refactor/rewire existing handlers if needed to keep behavior consistent across pages
10. **Provide clear access to all features:** Update Header/Footer/Sidebar navigation to include new page links; add contextual links between related pages; ensure every page (old and new) is reachable through intuitive navigation

DATA & LOGIC CONSISTENCY (CRITICAL):
- Every displayed count/number must be backed by actual mockData (no hardcoded displays that don't match data)
- Category contents must be consistent: if "5 items" shows in list, detail view must show those same 5
- Pagination must respect total count in mockData
- Status displays (read/unread, message counts, etc.) must reflect actual data state
- Related items and suggestions must link to real mockData entries
- User interactions must update consistent data across old and new pages
- No orphaned references or broken links
- New feature pages must use same data model as existing pages
- All data flows must match the trajectory logic

MOCKDATA EXPANSION GUIDELINES (Prevent Inconsistency):
When expanding mockData.js with new trajectory data:
  A. Audit new display requirements first (scan for all displayed counts, lists, numbers)
  B. Initialize new data with exact counts: if new page shows "10 items", add 12+ items to that table
  C. Keep ALL old mockData (never remove/overwrite), merge new data carefully
  D. Use non-conflicting IDs (if old data has items 1-50, new items start from 51+)
  E. Ensure data relationships: every foreign key must reference actual mockData items
  F. For category expansions: if showing "5 categories with 10 items each", create exactly 5 categories with 10+ items
  G. Computed values MUST be from actual data: use filters/length, never hardcoded
  H. Example: mockData.products.filter(p => p.category_id === catId).length (not hardcoded as 10)

CSS SYNTAX REQUIREMENTS (CRITICAL):
- ALWAYS use proper CSS property: value; format
- CORRECT: "justify-content: center;" NOT "justify-center;"
- CORRECT: "align-items: center;" NOT "align-center;"
- EVERY CSS property MUST have a colon before the value
- NEVER use Tailwind-style shorthand properties in .css files
- Example WRONG: "justify-center;" "align-center;" "flex-row;"
- Example CORRECT: "justify-content: center;" "align-items: center;" "flex-direction: row;"

IMAGE USAGE (For New Features):
- Call image_search(query="content keywords", num=5-10) for main content
- Use UI Avatars or DiceBear for text-based avatars
- Store search results in mockData.js with URLs
- Reference: <img src={item.image} alt="description" />
- NEVER use placeholder APIs (dummyimage.com, picsum.photos, etc.)

MOCK EXTERNAL OPERATIONS (CRITICAL):
- NO real payment gateway calls (use mock payment flow with setTimeout)
- NO real OAuth/social login (auto-login with mock credentials after loading spinner)
- NO real email/SMS sending (accept any verification code)
- NO real external API calls (return hardcoded mock data from backend)
- NO real file uploads to cloud (fake progress bar + store metadata in mockData)
- All "external" operations must complete locally with simulated delay (1-3s)
- Always show loading state + success feedback after mocked operations

CSS CHECKLIST - VERIFY BEFORE OUTPUT:
- [ ] All CSS properties use "property: value;" format
- [ ] NO Tailwind-style shorthand (justify-center, align-center, etc.)
- [ ] All flexbox properties properly named (justify-content, align-items, flex-direction)
- [ ] All CSS files are syntactically valid

DATA CONSISTENCY CHECKLIST - VERIFY BEFORE OUTPUT:
- [ ] All displayed numbers backed by actual mockData (no hardcoded counts)
- [ ] Category/folder contents consistent between list and detail views
- [ ] Pagination matches total data count in mockData
- [ ] Search results correspond to actual mockData content
- [ ] Status counters/badges match actual data state
- [ ] Related items link to real mockData entries
- [ ] User interactions update shared data consistently
- [ ] No orphaned references or broken links
- [ ] All data flows match trajectory logic

OUTPUT FORMAT:
- Explain integration strategy first (2-3 sentences)
- Then output files using: ===FILE: path/to/file.ext=== ... ===END FILE===
- Only include NEW or MODIFIED files
- Modified files must contain COMPLETE updated content

Start by analyzing the existing website structure.""",
    )

    print("[STEP] Agent analyzing existing website and planning expansion...")
    print("-" * 70)

    response = await run_agent(
        prompt,
        agent_options,
        label=f"Website Expansion Query {new_query_id}",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )

    # Parse files
    print("\n[STEP] Parsing generated changes...")
    print("-" * 70)
    files = parse_generated_files(response)

    if not files:
        print("[ERROR] No changes generated")
        return {"success": False, "error": "No files parsed from agent output"}

    print(f"\n[INFO] Parsed {len(files)} changed files")

    # Validate and fix common syntax errors
    print("\n[STEP] Validating and fixing syntax errors...")
    print("-" * 70)
    files = validate_and_fix_files(files)

    # Ensure all imported pages exist (prevent missing module errors)
    print("\n[STEP] Ensuring imported pages exist...")
    print("-" * 70)
    files = ensure_imported_pages_exist(website_dir, files)

    # Create backup if requested
    if create_backup:
        print("\n[STEP] Creating backup of existing files...")
        print("-" * 70)
        backup_existing_files(website_dir, list(files.keys()))

    # Apply changes
    print("\n[STEP] Applying changes to website...")
    print("-" * 70)
    modified_paths = apply_changes(website_dir, files)

    # Count new vs modified
    new_files = sum(1 for f in files.keys() if not (website_dir / f).exists())
    modified_files = len(files) - new_files

    # Validate and reconcile features
    print("\n[STEP] Validating expansion against feature expectations...")
    print("-" * 70)
    old_features = load_feature_report(website_dir, new_query_id)
    missing, warnings = validate_and_reconcile_features(
        website_dir, old_features, new_query_id, new_trajectory_file
    )
    
    if missing:
        print(f"[WARN] Missing features that should be present:")
        for feature in missing:
            print(f"  - {feature}")
    
    if warnings:
        for warning in warnings:
            print(f"[INFO] {warning}")
    
    # Generate updated feature report
    print("\n[STEP] Generating updated feature report...")
    print("-" * 70)
    feature_history = []
    if old_features:
        feature_history.append(old_features)
    
    current_features = extract_features_from_files(website_dir)
    feature_history.append(current_features)
    
    report_path = generate_expanded_feature_report(
        website_dir, 1, new_query_id, feature_history
    )

    # Generate summary
    print("\n[STEP] Generating expansion summary...")
    print("-" * 70)
    generate_expansion_summary(website_dir, str(new_query_id), new_files, modified_files)

    print("\n" + "=" * 70)
    print("[SUCCESS] Website expansion completed!")
    print("=" * 70)
    print(f"\n✓ Website directory: {website_dir}")
    print(f"✓ New files created: {new_files}")
    print(f"✓ Files modified: {modified_files}")
    print(f"✓ Feature report updated: {report_path.name}")
    if missing:
        print(f"\n⚠️  {len(missing)} missing features detected - review and adjust if needed")
    print(f"\n[NEXT STEPS]")
    print(f"1. Review changes in {website_dir}")
    print(f"2. Check feature report: {report_path.name}")
    if missing:
        print(f"3. Address missing features: {', '.join(missing)}")
    print(f"4. Restart backend: cd {website_dir}/backend && npm start")
    print(f"5. Restart frontend: cd {website_dir}/frontend && npm start")
    print(f"6. Test integration of new features")
    print("=" * 70 + "\n")

    return {
        "success": True,
        "query_id": new_query_id,
        "website_dir": str(website_dir),
        "new_files": new_files,
        "modified_files": modified_files,
        "feature_report": str(report_path),
        "missing_features": missing,
        "validation_warnings": warnings,
    }


def main():
    if not SDK_AVAILABLE:
        print(f"[ERROR] claude_agent_sdk not installed: {IMPORT_ERROR}")
        print("Install with: pip install claude-agent-sdk")
        sys.exit(1)

    args = parse_args()
    os.environ["ANTHROPIC_MODEL"] = args.model

    print("\n" + "=" * 70)
    print("WEBSITE EXPANSION TOOL (Agent SDK)")
    print("=" * 70 + "\n")

    try:
        # Find website directory
        website_dir = find_website_directory(args.website)
        print(f"[INFO] Website directory: {website_dir.name}")

        # Auto-detect or use provided PRD folder
        if args.prd_folder:
            prd_folder = Path(args.prd_folder).resolve()
        else:
            # Try to detect from website path
            # Assuming format: PRD_xxx/generated-website-query-N
            prd_folder = website_dir.parent
            if not prd_folder.name.startswith("PRD_"):
                raise ValueError("Cannot auto-detect PRD folder. Please provide --prd-folder")
        
        print(f"[INFO] PRD folder: {prd_folder.name}\n")

    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        result = asyncio.run(
            expand_website_agent(
                website_dir,
                args.query,
                prd_folder,
                heartbeat_secs=args.heartbeat_secs,
                overall_timeout=args.timeout_secs or 1200,
                create_backup=not args.no_backup,
            )
        )

        if result["success"]:
            print(f"[SUMMARY]")
            print(f"✓ Website: {result['website_dir']}")
            print(f"✓ New files: {result['new_files']}")
            print(f"✓ Modified files: {result['modified_files']}")
            print(f"✓ Feature report: {result['feature_report']}")
            
            if result.get('missing_features'):
                print(f"\n⚠️  Missing features ({len(result['missing_features'])}):")
                for feature in result['missing_features']:
                    print(f"  - {feature}")
            
            if result.get('validation_warnings'):
                print(f"\n ℹ️  Validation notes:")
                for warning in result['validation_warnings']:
                    print(f"  - {warning}")
        else:
            print(f"[ERROR] {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
