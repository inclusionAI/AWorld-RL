#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate website code via Claude Agent SDK.

The agent reads structured trajectory and design files, then generates
a complete monorepo with frontend (React) and backend (Node.js/Express) code.
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
# Prompt template for website generation
# ---------------------------------------------------------------------------
AGENT_WEBSITE_GENERATION_PROMPT = """
You are an expert full-stack developer. Generate a COMPLETE monorepo web application.

**Input Files:**
- Structured Trajectory: {trajectory_file_path}
- Design References: Available in {design_folder_path}

**Your Task:**
Generate ALL files needed for a working React + Node.js/Express application.

**CRITICAL REQUIREMENTS:**

0. **Data Consistency & Detail Correctness (MANDATORY):**
   - If a page shows "10 items" or lists "10 results", mockData.js must contain exactly 10+ items to support it
   - If a category header shows "5 products", drilling down must show those same 5 products (not 2 or 3)
   - Category counts in list views must match detailed view counts
   - All data displayed on pages MUST be initialized in mockData.js Page_0 db_snapshot
   - Backend endpoints must return complete data sets, not truncated versions
   - Pagination: if showing "1-10 of 50", total count must be >= 50
   - Search results must reflect actual mockData content, not fake numbers
   - Related items/suggestions must link to real items in mockData
   - User interactions (add to cart, like, save) must update consistent data across all pages
   - If email shows "3 new messages", sidebar counter must also show "3"
   - Status displays (online/offline, read/unread) must reflect actual mockData state

**mockData GENERATION GUIDELINES (CRITICAL FOR CONSISTENCY):**
When initializing mockData.js, follow these rules:
  A. **Audit Display Requirements First:**
     - Scan trajectory and frontend code for ALL displayed counts, lists, and numbers
     - Example: "10 products", "5 categories", "3 new messages", "50 total items"
     - Record MINIMUM data required: displayed_count + 20% buffer minimum
  B. **Initialize with Exact Counts:**
     - Create exactly enough items to support displayed counts + buffer
     - Example: If page shows "10 items", create 12-15 items in mockData
     - Example: If showing "5 categories with 10 items each", create 5 categories with 10+ items each
     - Example: If pagination shows "1-10 of 50", create exactly 50+ total items
  C. **Maintain Referential Integrity:**
     - EVERY foreign key in displayed data must reference actual items in mockData
     - EVERY count displayed must be computed from actual items, not hardcoded
     - No orphaned references (category_id in product must exist in categories table)
  D. **Match Trajectory Flow:**
     - mockData Page_0 db_snapshot defines INITIAL state (what exists before user actions)
     - Page_N snapshots should show RESULTS after each user action
     - Data counts should INCREASE/DECREASE based on actions (create, delete, update)
  E. **Category/Subcategory Consistency:**
     - If listing "5 categories", create 5 categories in mockData
     - If showing "10 items in category", ensure that category has 10+ items
     - Drill-down to category detail should show ALL items (not subset)
  F. **Computed Values Must Match:**
     - Total counts = calculated from actual data (use filter/reduce, not hardcoded)
     - Example: total_products = mockData.products.length (not hardcoded as 50)
     - Example: category_count = mockData.categories.filter(c => c.parent_id === cat.id).length
     - NEVER hardcode display numbers—compute from data

1. **File Structure (MUST USE "frontend/" and "backend/" prefixes):**
   ```
   project/
   ├── frontend/
   │   ├── package.json
   │   ├── .env
   │   ├── public/
   │   │   └── index.html
   │   └── src/
   │       ├── index.js
   │       ├── App.js
   │       ├── App.css
   │       ├── services/
   │       │   └── api.js
   │       ├── components/
   │       │   ├── Header.js/css
   │       │   ├── Footer.js/css
   │       │   └── [other components]
   │       └── pages/
   │           └── [all pages from trajectory]
   └── backend/
       ├── package.json
       ├── server.js
       └── data/
           └── mockData.js
   ```

2. **Frontend Requirements:**
   - React 18+ with React Router v6+
   - All pages must be separate .js/.css file pairs in src/pages/
   - Create Header, Footer, and any shared components
   - **Navigation:** Header/Footer must include links to ALL pages from trajectory
   - **Accessibility:** Every page must be reachable through UI navigation (no orphaned pages)
   - Use Tailwind CSS or inline styles (NOT inline style objects)
   - API service file (src/services/api.js) with axios for all backend calls
   - App.js uses React Router with routes for all pages
   - Error boundaries and loading states
   - **IMAGES & ICONS IN COMPONENTS:**
     * MANDATORY: All pages MUST include real images
     * Always include `<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />` in public/index.html
     * Use real image sources (details in IMAGE RESOURCES section below)
     * Font Awesome icons for UI elements
     * NEVER use: dummyimage.com, picsum.photos, via.placeholder.com, empty src, relative paths

3. **Backend Requirements:**
   - Express.js server on port 5001
   - CORS enabled for localhost:3000
   - Body-parser middleware configured
   - In-memory data store from mockData.js
   - RESTful endpoints matching the trajectory's database operations
   - All endpoints from Page_0 onwards with proper CRUD logic
   - Error handling for all endpoints

4. **Database/Data:**
   - mockData.js initializes ONLY from Page_0 db_snapshot
   - Backend endpoints support GET, POST, PUT, DELETE as needed
   - In-memory data modifications persist during session

5. **File Output Format:**
Use EXACT format for each file:

===FILE: path/to/file.ext===
[complete file content - must be syntactically correct]
===END FILE===

6. **Critical Checklist:**
   - [ ] frontend/.env file present
   - [ ] frontend/public/index.html with Font Awesome CDN link in <head>
   - [ ] frontend/src/index.js renders App
   - [ ] frontend/src/App.js with React Router and all page routes
   - [ ] frontend/src/services/api.js with all API calls
   - [ ] frontend/src/components/Header.js and Footer.js
   - [ ] **Header/Footer navigation includes links to ALL pages (no orphaned pages)**
   - [ ] frontend/src/pages/[PageName].js and .css for EACH page
   - [ ] All img tags use REAL images from Unsplash/Pexels API (see IMAGE RESOURCES section)
   - [ ] All Font Awesome icons use proper class names
   - [ ] backend/package.json with express, cors, body-parser, uuid
   - [ ] backend/server.js with complete Express app and all endpoints
   - [ ] backend/data/mockData.js with initial data from trajectory
   - [ ] All imports/requires resolve correctly
   - [ ] All CSS classes used in JSX exist in .css files
   - [ ] No syntax errors
   - [ ] **DATA CONSISTENCY CHECKLIST:**
     * [ ] List pages show correct count of items (e.g., "10 items" = 10+ items in mockData)
     * [ ] Detailed pages show complete data for selected item
     * [ ] Category/folder counts in lists match detail view counts
     * [ ] Pagination parameters match total mockData count
     * [ ] Search results correspond to actual mockData content
     * [ ] Status indicators (counts, badges) update consistently across pages
     * [ ] Related items/links point to real mockData entries
     * [ ] User action side effects update data consistently (cart, likes, etc.)
     * [ ] All displayed numbers are backed by actual data in mockData.js

7. **Mock External Operations (CRITICAL - No Real External Calls):**
   When implementing features that require external services or network operations, use FAKE/MOCK implementations:
   
   **Payment Operations:**
   - NO real payment gateway integration (Stripe, PayPal, Alipay, WeChat Pay, etc.)
   - Show payment form with fake credit card input fields
   - On submit: Show loading spinner → Redirect to success page with mock order confirmation
   - Update mockData to reflect "payment completed" status
   - Example: `/payment → loading (2s) → /order-success?orderId=12345`
   
   **Third-Party Login (OAuth, Social Login):**
   - NO real OAuth flow (Google, Facebook, GitHub, WeChat, etc.)
   - Show "Login with [Provider]" button
   - On click: Show loading spinner → Auto-login with mock user account
   - Update mockData to set user as logged in with fake profile data
   - Example: Click "Login with Google" → loading (1s) → Logged in as "mock.user@gmail.com"
   
   **Email Verification / SMS Verification:**
   - NO real email/SMS sending
   - Show verification code input form
   - Accept ANY 6-digit code as valid (or auto-fill "123456")
   - On submit: Show success message → Update user status to "verified"
   - Example: Enter any code → "Verification successful!"
   
   **External API Calls (Weather, Maps, Stock Data, etc.):**
   - NO real API calls to external services
   - Return hardcoded mock data from backend
   - Example: Weather API → Always return "Sunny, 72°F"
   - Example: Map API → Show static map image or simple placeholder
   
   **File Upload to Cloud Storage:**
   - NO real file upload to S3, CDN, or cloud storage
   - Accept file selection, show upload progress bar (fake)
   - Store file name/metadata in mockData (don't actually upload)
   - Show success message with fake file URL
   - Example: Upload image → progress 0%→100% (2s) → "Uploaded to /uploads/fake-image.jpg"
   
   **Push Notifications / Real-time Updates:**
   - NO WebSocket connections or real-time APIs
   - Use setTimeout to simulate delayed notifications
   - Example: Click "Subscribe" → After 3s show in-app notification "You have a new message"
   
   **Implementation Pattern:**
   ```javascript
   // CORRECT - Mock payment
   const handlePayment = async (paymentData) => {
     setLoading(true);
     // Simulate API call delay
     await new Promise(resolve => setTimeout(resolve, 2000));
     
     // Update mock data (no real payment)
     const orderId = Date.now();
     // Add order to mockData...
     
     setLoading(false);
     navigate(`/order-success?orderId=${orderId}`);
   };
   
   // WRONG - Real payment API call
   // const response = await fetch('https://api.stripe.com/...'); // ❌ NO!
   ```
   
   **Key Principle:**
   - All "external" operations must complete locally without network calls
   - Use loading states + setTimeout to simulate network delay (1-3 seconds)
   - Always show success feedback (redirect, message, updated UI)
   - Update mockData to reflect state changes as if operation succeeded
   - User should feel like the feature "works" even though it's mocked

**FRONTEND .env FILE CONFIGURATION:**
- frontend/.env must include:
  ```
  REACT_APP_API_URL=http://localhost:5001
  REACT_APP_UNSPLASH_CLIENT_ID=DEMO
  PORT=3000
  DANGEROUSLY_DISABLE_HOST_CHECK=true
  ```
- DANGEROUSLY_DISABLE_HOST_CHECK=true is required to fix Dev Server allowedHosts validation error
- Without this, npm start will fail with "options.allowedHosts[0] should be a non-empty string" error

**Design Guidelines - IMAGES & VISUALS:**
- Read design files from {design_folder_path} for visual reference
- Adapt HTML structure and CSS patterns from designs
- Maintain consistent styling across pages
- Use responsive design principles

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
   - Mix different orientations and compositions as needed
   - Store URLs in mock data to ensure consistency across page reloads
   - Adapt search terms based on website content

**CRITICAL: ALWAYS search for real images using the image_search tool. Real photos look much better than placeholders. Integrate search results into mockData.js for consistent display.**

Generate the complete application now. Output ALL files with REAL image URLs from image_search results.
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
        # Only trigger if it's a suspicious standalone keyword
        if re.search(r'^\s*(center|left|right|top|bottom|middle|baseline|stretch)\s*;', line):
            # Check if previous line has a property without value
            if fixed_lines and ':' in fixed_lines[-1] and not fixed_lines[-1].rstrip().endswith(';'):
                # This might be a value for previous property - keep as is
                pass
            else:
                issues.append(f"Line {i}: Suspicious standalone value '{line.strip()}' (might need property name)")
        
        # Fix 4: Property without colon "property value;" → "property: value;"
        match = re.match(r'^(\s*)([a-z-]+)\s+([^:;]+);', line)
        if match and ':' not in line:
            indent, prop, value = match.groups()
            # Common properties that should have colons
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


def create_project_structure(files: dict, output_dir: Path) -> list[str]:
    """Create project directory structure and write files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    for filepath, content in files.items():
        full_path = output_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            full_path.write_text(content, encoding="utf-8")
            created_files.append(str(full_path))
            print(f"[CREATED] {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to write {filepath}: {e}")

    return created_files


def extract_features_from_files(output_dir: Path) -> dict:
    """Extract feature/functionality information from generated files.
    
    Analyzes:
    - Frontend routes and pages
    - API endpoints from server.js
    - Data structures from mockData.js
    - React components
    
    Returns dict with structured feature information.
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
            
            # Find data structure definitions (e.g., db.users, db.products)
            data_pattern = r'(?:db|mockData|data)\.([a-zA-Z_]+)\s*=\s*\['
            data_models = re.findall(data_pattern, mock_content)
            features["data_models"] = sorted(list(set(data_models)))
    
    except Exception as e:
        print(f"[WARN] Error extracting features: {e}")
    
    return features


def generate_feature_report(output_dir: Path, query_id: str) -> Path:
    """Generate feature report after website generation.
    
    Creates a markdown report documenting:
    - All pages/routes implemented
    - All API endpoints
    - All data models
    - Feature summary
    
    Returns path to the generated report file.
    """
    features = extract_features_from_files(output_dir)
    
    report_content = f"""# Feature Report - Query {query_id}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Website Directory:** {output_dir.name}

## Summary

- **Pages/Routes:** {len(features['pages'])}
- **API Endpoints:** {len(features['api_endpoints'])}
- **Data Models:** {len(features['data_models'])}
- **Components:** {len(features['components'])}

## Pages & Routes

""" + "\n".join(f"- `{page}`" for page in features['pages']) + f"""

## API Endpoints

""" + "\n".join(f"- `{endpoint}`" for endpoint in features['api_endpoints']) + f"""

## Data Models

""" + "\n".join(f"- `{model}`" for model in features['data_models']) + f"""

## Components

""" + "\n".join(f"- `{comp}`" for comp in features['components']) + f"""

## Raw Features Data (for s8 validation)

```json
{json.dumps(features, indent=2)}
```

---

This report was auto-generated by s7_generate_website_with_validation_agent.py
"""
    
    report_path = output_dir / f"FEATURE_REPORT_query_{query_id}.md"
    report_path.write_text(report_content, encoding="utf-8")
    
    print(f"\n[CREATED] Feature Report: FEATURE_REPORT_query_{query_id}.md")
    
    return report_path


def generate_readme(output_dir: Path, query_id: str, file_count: int) -> None:
    """Generate README.md with setup instructions."""
    readme_content = f"""# Generated Website - Query {query_id}

This is a complete full-stack web application generated from a page trajectory document.

## Project Structure

```
.
├── backend/           # Node.js + Express.js API server
│   ├── package.json
│   ├── server.js
│   └── data/
│       └── mockData.js
├── frontend/          # React.js web application
│   ├── package.json
│   ├── .env
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js
│       ├── App.js
│       ├── services/
│       │   └── api.js
│       ├── components/
│       └── pages/
└── README.md          # This file
```

## Setup Instructions

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the server:
   ```bash
   npm start
   ```

   Backend will run on: **http://localhost:5001**

### Frontend Setup

1. Open a new terminal and navigate to frontend:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   Frontend will run on: **http://localhost:3000**

## Technology Stack

- **Frontend:** React.js 18+, React Router 6+, Axios
- **Backend:** Node.js, Express.js
- **Data:** In-memory mock data store (no database required)
- **Styling:** CSS / Tailwind CSS

## Project Information

- **Total Files Generated:** {file_count}
- **Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source:** Page Trajectory Query {query_id}

## Development Workflow

1. Start backend server first (it initializes the data)
2. Start frontend application (it connects to backend on startup)
3. Browser will open at http://localhost:3000
4. All API calls go through the Axios service (src/services/api.js)
5. Backend maintains in-memory state during the session

## Important Notes

- The backend uses in-memory storage. Data is reset when the server restarts.
- The application is fully functional for testing the page trajectories.
- All endpoints from the trajectory are implemented.
- Error handling is included for better debugging.

## Troubleshooting

**Port already in use?**
- Backend: Modify PORT in backend/server.js or backend/package.json scripts
- Frontend: Set PORT=3001 npm start

**CORS errors?**
- Make sure backend is running on port 5001
- Check that API calls in frontend point to correct base URL

**Module not found?**
- Run `npm install` in both frontend and backend directories
- Clear node_modules: `rm -rf node_modules && npm install`

---

For questions or issues, review the generated code structure and the source trajectory document.
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[CREATED] README.md")


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
            # Sort by date (newest first) to get the most recent
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
            # Sort by: (1) pname position (prefer later), (2) date (prefer newer)
            selected_folder = sorted(potential, key=lambda f: (-f.name.lower().rfind(clean_pname), f.name), reverse=True)[0]
            return selected_folder
        
        raise FileNotFoundError(f"No PRD folder found with pname: {pname}")
    else:
        selected_folder = sorted(prd_folders, reverse=True)[0]
        return selected_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Generate website via Agent SDK")
    parser.add_argument("--pname", "-n", default=None, help="PRD project name")
    parser.add_argument("--query", "-q", type=int, default=1, help="Query ID (default: 1)")
    parser.add_argument("--heartbeat-secs", type=int, default=60, help="Heartbeat interval")
    parser.add_argument("--timeout-secs", type=int, default=3600, help="Overall timeout (default: 3600s/60min)")
    parser.add_argument("--model", "-m", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"))
    return parser.parse_args()


async def generate_website_agent(
    prd_folder: Path,
    query_id: int,
    heartbeat_secs: int = 60,
    overall_timeout: int = 900,
) -> dict:
    """Run agent to generate website code."""
    if not SDK_AVAILABLE:
        raise ImportError(f"claude_agent_sdk not installed: {IMPORT_ERROR}")

    print("=" * 70)
    print(f"GENERATING WEBSITE FOR QUERY {query_id}")
    print("=" * 70)

    # Find required files
    trajectory_file = find_trajectory_file(prd_folder, str(query_id))
    design_folder = find_design_folder(prd_folder)

    print(f"[INFO] Trajectory: {trajectory_file.name}")
    print(f"[INFO] Design folder: {design_folder.name}\n")

    prompt = AGENT_WEBSITE_GENERATION_PROMPT.replace(
        "{trajectory_file_path}", str(trajectory_file)
    ).replace(
        "{design_folder_path}", str(design_folder)
    )

    agent_options = ClaudeAgentOptions(
        cwd=str(prd_folder),
        permission_mode="bypassPermissions",
        allowed_tools="*",
        max_turns=20,
        system_prompt="""You are an expert full-stack developer. Generate a COMPLETE working application with REAL images on EVERY page.

Your task:
1. Read the structured trajectory from the file path
2. Review design files in design_folder for visual reference
3. Generate ALL files for frontend (React) and backend (Node.js/Express)
4. Use format: ===FILE: path/to/file.ext=== ... ===END FILE===

CRITICAL:
- Use "frontend/" prefix for ALL frontend files
- Use "backend/" prefix for ALL backend files
- Generate EVERY component, page, and file
- NO placeholders or TODOs
- All imports must resolve
- **Header/Footer must include navigation links to ALL pages (every page must be accessible)**
- All CSS must be complete and SYNTACTICALLY VALID
- Backend must have complete Express server with all endpoints
- mockData.js must initialize from trajectory Page_0 db_snapshot

DATA CONSISTENCY & CORRECTNESS (CRITICAL):
- Every displayed count/number MUST be backed by actual mockData items
- If showing "10 items" or "5 results", mockData must have 10+ or 5+ items
- Drill-down pages must show complete data for selected items, not truncated versions
- Category counts in list views MUST match detail view counts
- Pagination: if showing "1-10 of 50", total must be >= 50 items
- Search results must reflect ACTUAL mockData content
- Status displays (read/unread, online/offline, message counts) must match actual state
- Related items and suggestions must link to real entries in mockData
- User interactions (add to cart, like, mark as read) must update consistent data
- No orphaned references or broken links to non-existent data
- All displayed information must follow the trajectory's data flow logic

CSS SYNTAX REQUIREMENTS (CRITICAL):
- ALWAYS use proper CSS property: value; format
- CORRECT: "justify-content: center;" NOT "justify-center;"
- CORRECT: "align-items: center;" NOT "align-center;"
- EVERY CSS property MUST have a colon before the value
- NEVER use Tailwind-style shorthand properties in .css files
- Example WRONG: "justify-center;" "align-center;" "flex-row;"
- Example CORRECT: "justify-content: center;" "align-items: center;" "flex-direction: row;"

IMAGES ON EVERY PAGE (MANDATORY - USE image_search Tool):
- EVERY img tag MUST use REAL images from image_search tool
- Call image_search(query="appropriate keywords", num=N) to find images
- Store search results in mockData.js with URLs
- Reference stored URLs in components: <img src={item.image} alt="..." />
- UI Avatars for text-based avatars: https://ui-avatars.com/api/?name=TEXT&size=SIZE&background=COLOR&color=fff
- DiceBear for user avatars: https://api.dicebear.com/7.x/avataaars/svg?seed=USER_ID
- CRITICAL: Use image_search for all main content images, not placeholder APIs
- REQUIREMENT: All images should be REAL photos from web search
- NEVER: Random endpoints, dummyimage.com, picsum.photos, placeholder APIs, empty src

IMAGE CHECKLIST - VERIFY BEFORE OUTPUT:
- [ ] image_search() called for appropriate content types
- [ ] Search results stored in mockData.js with URL field
- [ ] Components reference stored image URLs: <img src={item.image} />
- [ ] Text-based content uses UI Avatars API
- [ ] User profiles use DiceBear or search results
- [ ] All images are REAL photos from search results
- [ ] NO placeholder images (dummyimage.com, picsum.photos, via.placeholder.com)
- [ ] NO random endpoints (loremflickr, /random, unsplash/random)
- [ ] NO JSON endpoints or empty src attributes
- [ ] Images display on every page refresh

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

Start outputting files now.
""",
    )

    print("[STEP] Agent generating website code...")
    print("-" * 70)

    response = await run_agent(
        prompt,
        agent_options,
        label=f"Website Generation Query {query_id}",
        heartbeat_secs=heartbeat_secs,
        overall_timeout=overall_timeout,
    )

    # Parse files
    print("\n[STEP] Parsing generated files...")
    print("-" * 70)
    files = parse_generated_files(response)

    if not files:
        print("[ERROR] No files generated")
        return {"success": False, "error": "No files parsed from agent output"}

    print(f"\n[INFO] Parsed {len(files)} files")

    # Validate and fix common syntax errors
    print("\n[STEP] Validating and fixing syntax errors...")
    print("-" * 70)
    files = validate_and_fix_files(files)

    # Create project structure
    print("\n[STEP] Creating project structure...")
    print("-" * 70)
    output_dir = prd_folder / f"generated-website-query-{query_id}"
    created_files = create_project_structure(files, output_dir)

    # Generate README
    print("\n[STEP] Generating README...")
    print("-" * 70)
    generate_readme(output_dir, str(query_id), len(files))

    # Generate Feature Report
    print("\n[STEP] Generating feature report...")
    print("-" * 70)
    report_path = generate_feature_report(output_dir, str(query_id))

    print("\n" + "=" * 70)
    print("[SUCCESS] Website generation completed!")
    print("=" * 70)
    print(f"\n✓ Output directory: {output_dir}")
    print(f"✓ Total files created: {len(created_files)}")
    print(f"✓ Feature report: {report_path.name}")
    print(f"\n[NEXT STEPS]")
    print(f"1. cd {output_dir}/backend && npm install && npm start")
    print(f"2. cd {output_dir}/frontend && npm install && npm start")
    print("=" * 70 + "\n")

    return {
        "success": True,
        "query_id": query_id,
        "output_dir": str(output_dir),
        "files_created": len(created_files),
        "feature_report": str(report_path),
    }


def main():
    if not SDK_AVAILABLE:
        print(f"[ERROR] claude_agent_sdk not installed: {IMPORT_ERROR}")
        print("Install with: pip install claude-agent-sdk")
        sys.exit(1)

    args = parse_args()
    os.environ["ANTHROPIC_MODEL"] = args.model

    print("\n" + "=" * 70)
    print("WEBSITE GENERATOR (Agent SDK)")
    print("=" * 70 + "\n")

    try:
        prd_folder = find_project_folder(args.pname)
        print(f"[INFO] Using PRD folder: {prd_folder.name}\n")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        result = asyncio.run(
            generate_website_agent(
                prd_folder,
                args.query,
                heartbeat_secs=args.heartbeat_secs,
                overall_timeout=args.timeout_secs,
            )
        )

        if result["success"]:
            print(f"[SUMMARY]")
            print(f"✓ Project location: {result['output_dir']}")
            print(f"✓ Files created: {result['files_created']}")
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
