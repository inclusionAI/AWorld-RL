#!/usr/bin/env python3
"""
Semantic Conflict Perturbation Injector
Semantic conflict perturbation injector - changes interaction semantics

Changes all button click behavior to require double-click to trigger, testing Agent's adaptability to semantic changes.

Usage:
    python inject_semantic_conflict_perturbation.py --input <project_dir> [--react] [--mode double-click]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def get_semantic_conflict_script(mode="double-click", show_hints=False, hint_level="none"):
    """
    Generate semantic conflict injection script
    
    Args:
        mode: Conflict mode
            - "double-click": All buttons require double-click to trigger
            - "long-press": All buttons require long-press to trigger (not yet implemented)
            - "right-click": All buttons require right-click to trigger (not yet implemented)
        show_hints: Whether to show hints on page (deprecated, use hint_level)
        hint_level: Hint level
            - "none": Completely silent, no hints
            - "console": Only log in console
            - "subtle": Lightweight visual feedback (element highlight)
            - "explicit": Full hint text (popup + highlight)
    """
    
    if mode == "double-click":
        return f"""
<script>
(function() {{
    'use strict';
    
    const MODE = 'double-click';
    const HINT_LEVEL = '{hint_level}';  // none, console, subtle, explicit
    
    console.log(`[Semantic Conflict] Initialized with hint_level: ${{HINT_LEVEL}}`);
    const MAX_LOG_ENTRIES = 100;
    
    // Initialize localStorage logs
    if (!localStorage.getItem('semanticConflictLogs')) {{
        localStorage.setItem('semanticConflictLogs', JSON.stringify([]));
    }}
    
    // Log intercept events
    function logInterceptEvent(eventType, targetInfo, intercepted, reason) {{
        try {{
            let logs = JSON.parse(localStorage.getItem('semanticConflictLogs') || '[]');
            logs.push({{
                timestamp: new Date().toISOString(),
                mode: MODE,
                eventType: eventType,
                targetInfo: targetInfo,
                intercepted: intercepted,
                reason: reason
            }});
            
            if (logs.length > MAX_LOG_ENTRIES) {{
                logs = logs.slice(-MAX_LOG_ENTRIES);
            }}
            
            localStorage.setItem('semanticConflictLogs', JSON.stringify(logs));
        }} catch (e) {{
            console.error('[Semantic Conflict] Failed to log event:', e);
        }}
    }}
    
    // Get element information
    function getElementInfo(element) {{
        if (!element) return 'unknown';
        
        const info = [];
        if (element.id) info.push(`#${{element.id}}`);
        if (element.className) info.push(`.${{element.className.split(' ').join('.')}}`);
        if (element.tagName) info.push(element.tagName.toLowerCase());
        if (element.textContent) {{
            const text = element.textContent.trim().substring(0, 30);
            if (text) info.push(`"${{text}}"`);
        }}
        
        return info.join(' ') || 'unknown element';
    }}
    
    // Determine if element is clickable (button, link, etc.)
    function isClickableElement(element) {{
        if (!element) return false;
        
        const tagName = element.tagName.toLowerCase();
        const role = element.getAttribute('role');
        const type = element.getAttribute('type');
        
        // Standard clickable elements
        if (tagName === 'button') return true;
        if (tagName === 'a') return true;
        if (type === 'button' || type === 'submit') return true;
        if (role === 'button') return true;
        
        // Elements with click handlers
        const hasClickHandler = element.onclick || 
                               element.hasAttribute('onclick') ||
                               element.style.cursor === 'pointer';
        
        return hasClickHandler;
    }}
    
    // Show hint (based on hint_level)
    function showHint(element, message, type = 'blocked') {{
        // console level: only log messages
        if (HINT_LEVEL === 'console' || HINT_LEVEL === 'subtle' || HINT_LEVEL === 'explicit') {{
            console.warn(`[Semantic Conflict] ${{message}}`);
        }}
        
        // subtle level: brief element highlight + shake (on failure)
        if (HINT_LEVEL === 'subtle' && type === 'blocked') {{
            if (element && element.style) {{
                const originalOutline = element.style.outline;
                element.style.outline = '2px solid red';
                element.style.outlineOffset = '2px';
                
                // Add shake animation
                const originalTransform = element.style.transform;
                element.style.transition = 'transform 0.1s';
                element.style.transform = 'translateX(-3px)';
                setTimeout(() => {{
                    element.style.transform = 'translateX(3px)';
                    setTimeout(() => {{
                        element.style.transform = 'translateX(-3px)';
                        setTimeout(() => {{
                            element.style.transform = originalTransform;
                        }}, 50);
                    }}, 50);
                }}, 50);
                
                // Restore after 1 second
                setTimeout(() => {{
                    element.style.outline = originalOutline;
                    element.style.outlineOffset = '';
                    element.style.transition = '';
                }}, 1000);
            }}
        }}
        
        // explicit level: top banner hint
        if (HINT_LEVEL === 'explicit') {{
            showTopBanner(message);
        }}
    }}
    
    // Show top banner (explicit mode) - changed to persistent status bar
    let permanentStatusBar = null;
    
    function showTopBanner(message) {{
        // explicit mode: create or update persistent status bar
        if (!permanentStatusBar) {{
            createPermanentStatusBar();
        }}
        
        // Update status bar message
        const messageEl = permanentStatusBar.querySelector('.status-message');
        if (messageEl) {{
            messageEl.textContent = message;
            
            // Highlight effect
            permanentStatusBar.style.background = 'rgba(255, 152, 0, 0.95)';
            setTimeout(() => {{
                permanentStatusBar.style.background = 'rgba(33, 150, 243, 0.9)';
            }}, 1000);
        }}
    }}
    
    function createPermanentStatusBar() {{
        // Create persistent status bar
        const statusBar = document.createElement('div');
        statusBar.id = 'semantic-conflict-status-bar';
        statusBar.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 16px;">⚠️</span>
                    <span style="font-weight: bold;">Attention</span>
                    <span style="margin-left: 10px; opacity: 0.9;" class="status-message">Click to select, double-click to activate</span>
                </div>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 12px; opacity: 0.8;">Hint: Explicit</span>
                </div>
            </div>
        `;
        statusBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(33, 150, 243, 0.9);
            color: white;
            padding: 10px 20px;
            font-size: 13px;
            z-index: 999999;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: background 0.3s ease;
            backdrop-filter: blur(10px);
        `;
        
        // Add animation styles
        if (!document.querySelector('#semantic-conflict-status-style')) {{
            const style = document.createElement('style');
            style.id = 'semantic-conflict-status-style';
            style.textContent = `
                #semantic-conflict-status-bar {{
                    animation: slideDown 0.3s ease-out;
                }}
                @keyframes slideDown {{
                    from {{ transform: translateY(-100%); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                /* Adjust page content to avoid being covered by status bar */
                body {{
                    padding-top: 45px !important;
                }}
            `;
            document.head.appendChild(style);
        }}
        
        // Ensure added after body is ready
        function addToBody() {{
            if (document.body) {{
                document.body.appendChild(statusBar);
                permanentStatusBar = statusBar;
                console.log('[Semantic Conflict] Permanent status bar created');
            }} else {{
                setTimeout(addToBody, 50);
            }}
        }}
        
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', addToBody);
        }} else {{
            addToBody();
        }}
    }}
    
    // Selected state management
    const selectedElements = new WeakMap();
    
    // Show selected state
    function showSelected(element) {{
        if (!element || !element.style) return;
        
        element.style.outline = '3px solid #2196F3';
        element.style.outlineOffset = '2px';
        element.style.boxShadow = '0 0 10px rgba(33, 150, 243, 0.5)';
        selectedElements.set(element, {{
            originalOutline: element.style.outline,
            originalBoxShadow: element.style.boxShadow,
            selected: true
        }});
        
        if (HINT_LEVEL !== 'none') {{
            console.log('[Semantic Conflict] Element selected (click again to deselect, or double-click to activate)');
        }}
    }}
    
    // Clear selected state
    function clearSelected(element) {{
        if (!element || !element.style) return;
        
        const state = selectedElements.get(element);
        if (state && state.selected) {{
            element.style.outline = '';
            element.style.outlineOffset = '';
            element.style.boxShadow = '';
            selectedElements.set(element, {{ selected: false }});
            
            if (HINT_LEVEL !== 'none') {{
                console.log('[Semantic Conflict] Element deselected');
            }}
        }}
    }}
    
    // Check if selected
    function isSelected(element) {{
        const state = selectedElements.get(element);
        return state && state.selected;
    }}
    
    // Track click count and time for each element
    const clickTracker = new WeakMap();
    const DOUBLE_CLICK_THRESHOLD = 500; // Second click within 500ms counts as double-click
    
    // Find nearest clickable element (traverse up DOM tree)
    function findClickableElement(target) {{
        let element = target;
        let depth = 0;
        const maxDepth = 10; // Search up to 10 levels
        
        while (element && depth < maxDepth) {{
            if (isClickableElement(element)) {{
                return element;
            }}
            element = element.parentElement;
            depth++;
        }}
        
        return null;
    }}
    
    // Click interceptor - click to select/deselect, double-click to trigger
    const clickInterceptor = function(event) {{
        // Find actual clickable element (may be parent element)
        const element = findClickableElement(event.target);
        
        // No clickable element found, do not intercept
        if (!element) {{
            return;
        }}
        
        const now = Date.now();
        const tracker = clickTracker.get(element) || {{ count: 0, lastClickTime: 0 }};
        
        // Check if double-click (second click within threshold)
        const timeSinceLastClick = now - tracker.lastClickTime;
        const isDoubleClick = tracker.count === 1 && timeSinceLastClick < DOUBLE_CLICK_THRESHOLD;
        
        if (isDoubleClick) {{
            // Double-click: actually trigger action
            if (HINT_LEVEL !== 'none') {{
                console.log('[Semantic Conflict] ✓ Double-click detected, allowing action on:', getElementInfo(element));
            }}
            logInterceptEvent('click', getElementInfo(element), false, 'double-click allowed');
            
            // Clear selected state
            clearSelected(element);
            
            // Reset counter
            clickTracker.set(element, {{ count: 0, lastClickTime: now }});
            
            // Allow event to continue
            return;
        }} else {{
            // Single-click: toggle selected state
            if (isSelected(element)) {{
                // Already selected → deselect
                clearSelected(element);
                if (HINT_LEVEL !== 'none') {{
                    console.log('[Semantic Conflict] Element deselected');
                }}
            }} else {{
                // Not selected → select
                showSelected(element);
                if (HINT_LEVEL === 'explicit') {{
                    showTopBanner('Selected! Double-click to activate, or click again to deselect');
                }}
            }}
            
            if (HINT_LEVEL !== 'none') {{
                console.log('[Semantic Conflict] ✗ Single-click intercepted (select/deselect mode)');
            }}
            logInterceptEvent('click', getElementInfo(element), true, 'single-click - select/deselect');
            
            // Update counter
            clickTracker.set(element, {{ count: tracker.count + 1, lastClickTime: now }});
            
            // Block event
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            
            // Reset counter after 500ms (if no second click)
            setTimeout(() => {{
                const currentTracker = clickTracker.get(element);
                if (currentTracker && currentTracker.lastClickTime === now) {{
                    clickTracker.set(element, {{ count: 0, lastClickTime: 0 }});
                }}
            }}, DOUBLE_CLICK_THRESHOLD);
        }}
    }};
    
    // Intercept all click events (use capture phase, highest priority)
    document.addEventListener('click', clickInterceptor, true);
    
    // Intercept Enter key submission (optional, as it is also a "click" semantic)
    document.addEventListener('keypress', function(event) {{
        if (event.key === 'Enter') {{
            const element = event.target;
            if (isClickableElement(element)) {{
                if (HINT_LEVEL !== 'none') {{
                    console.log('[Semantic Conflict] Enter key blocked on button, requires double-click');
                }}
                logInterceptEvent('keypress-enter', getElementInfo(element), true, 'enter blocked');
                showHint(element, '⚠️ Use DOUBLE-CLICK instead of Enter');
                event.preventDefault();
                event.stopPropagation();
            }}
        }}
    }}, true);
    
    // Export view functions
    window.getSemanticConflictLogs = function() {{
        const logs = JSON.parse(localStorage.getItem('semanticConflictLogs') || '[]');
        console.table(logs);
        return logs;
    }};
    
    window.getSemanticConflictStats = function() {{
        const logs = JSON.parse(localStorage.getItem('semanticConflictLogs') || '[]');
        const stats = {{
            total: logs.length,
            intercepted: logs.filter(l => l.intercepted).length,
            allowed: logs.filter(l => !l.intercepted).length,
            byEventType: {{}},
            recentBlocked: logs.filter(l => l.intercepted).slice(-10)
        }};
        
        logs.forEach(log => {{
            if (!stats.byEventType[log.eventType]) {{
                stats.byEventType[log.eventType] = {{ total: 0, intercepted: 0, allowed: 0 }};
            }}
            stats.byEventType[log.eventType].total++;
            if (log.intercepted) {{
                stats.byEventType[log.eventType].intercepted++;
            }} else {{
                stats.byEventType[log.eventType].allowed++;
            }}
        }});
        
        console.log('=== Semantic Conflict Statistics ===');
        console.log(`Mode: ${{MODE}}`);
        console.log(`Total events: ${{stats.total}}`);
        console.log(`Intercepted (blocked): ${{stats.intercepted}}`);
        console.log(`Allowed: ${{stats.allowed}}`);
        console.log('\\nBy Event Type:');
        console.table(stats.byEventType);
        console.log('\\nRecent Blocked Events:');
        console.table(stats.recentBlocked);
        return stats;
    }};
    
    window.clearSemanticConflictLogs = function() {{
        localStorage.removeItem('semanticConflictLogs');
        console.log('[Semantic Conflict] Logs cleared');
    }};
    
    console.log('🎭 Semantic Conflict Perturbation Injected');
    console.log(`   Mode: ${{MODE}}`);
    console.log(`   Effect: All buttons require DOUBLE-CLICK to activate`);
    console.log('   💡 Use getSemanticConflictStats() to view statistics');
    console.log('   💡 Use getSemanticConflictLogs() to view detailed logs');
}})();
</script>
"""
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def inject_to_html(html_path, script_content):
    """Inject script into HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already injected
        if 'Semantic Conflict' in content:
            print(f"⚠️  Already injected: {html_path}")
            return False
        
        # Inject before </head> tag
        if '</head>' in content:
            content = content.replace('</head>', f'{script_content}\n</head>')
        elif '<body>' in content or '<body ' in content:
            import re
            content = re.sub(r'(<body[^>]*>)', f'{script_content}\n\\1', content, count=1)
        else:
            content = script_content + '\n' + content
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"❌ Error injecting to {html_path}: {e}")
        return False


def inject_semantic_conflict(
    input_dir,
    output_dir=None,
    is_react=False,
    mode="double-click",
    show_hints=True,
    hint_level="none"
):
    """
    Inject semantic conflict perturbation into web project
    
    Args:
        input_dir: Input project directory
        output_dir: Output directory (if None, add _semantic suffix to original directory)
        is_react: Whether it's a React project
        mode: Conflict mode
        show_hints: Whether to show hints (deprecated, use hint_level)
        hint_level: Hint level (none/console/subtle/explicit)
    """
    input_path = Path(input_dir).resolve()
    
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {input_path}")
        return False
    
    # Determine output directory
    if output_dir is None:
        output_path = Path(str(input_path) + f'_semantic_{hint_level}')
    else:
        output_path = Path(output_dir).resolve()
    
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"🎭 Mode: {mode}")
    print(f"💡 Hint level: {hint_level}")
    
    # If output directory already exists, delete it first
    if output_path.exists():
        print(f"🗑️  Removing existing output directory...")
        shutil.rmtree(output_path)
    
    # Copy entire project (excluding node_modules)
    print("📋 Copying project files...")
    try:
        shutil.copytree(
            input_path,
            output_path,
            ignore=shutil.ignore_patterns('node_modules', '__pycache__', '*.pyc', '.git')
        )
    except Exception as e:
        print(f"❌ Error copying project: {e}")
        return False
    
    # Generate injection script
    script_content = get_semantic_conflict_script(mode=mode, show_hints=show_hints, hint_level=hint_level)
    
    # Find and inject HTML files
    injected_count = 0
    
    if is_react:
        # React project: inject into frontend/public/index.html
        html_paths = [
            output_path / 'frontend' / 'public' / 'index.html',
            output_path / 'public' / 'index.html'
        ]
    else:
        # Regular HTML project: find all HTML files
        html_paths = list(output_path.rglob('*.html'))
    
    for html_path in html_paths:
        if html_path.exists() and html_path.is_file():
            print(f"💉 Injecting to: {html_path.relative_to(output_path)}")
            if inject_to_html(html_path, script_content):
                injected_count += 1
    
    if injected_count == 0:
        print("⚠️  Warning: No HTML files were injected!")
        return False
    
    print(f"\n{'='*80}")
    print(f"✅ Semantic conflict perturbation injection completed!")
    print(f"   Injected to {injected_count} HTML file(s)")
    print(f"   Mode: {mode}")
    print(f"   Effect: All buttons require DOUBLE-CLICK to activate")
    print(f"   Output: {output_path}")
    print(f"{'='*80}")
    
    print(f"\n📊 To view statistics in browser console:")
    print(f"   window.getSemanticConflictStats()")
    print(f"   window.getSemanticConflictLogs()")
    print(f"   window.clearSemanticConflictLogs()")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Inject semantic conflict perturbations into web projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # React project with subtle hints (recommended)
  python inject_semantic_conflict_perturbation.py --input ./generated-website --react --hint-level subtle
  
  # No hints at all (silent mode)
  python inject_semantic_conflict_perturbation.py --input ./website --hint-level none
  
  # Console logging only
  python inject_semantic_conflict_perturbation.py --input ./site --hint-level console
  
  # Explicit hints (full visual feedback)
  python inject_semantic_conflict_perturbation.py --input ./site --hint-level explicit
  
  # Custom output directory
  python inject_semantic_conflict_perturbation.py --input ./site --output ./site_semantic
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input project directory path'
    )
    parser.add_argument(
        '--output',
        help='Output directory path (default: input_dir + "_semantic")'
    )
    parser.add_argument(
        '--react',
        action='store_true',
        default=True,
        help='Treat as React project (inject to frontend/public/index.html)'
    )
    parser.add_argument(
        '--mode',
        default='double-click',
        choices=['double-click'],
        help='Semantic conflict mode (default: double-click)'
    )
    parser.add_argument(
        '--hint-level',
        default='explicit',
        choices=['none', 'console', 'subtle', 'explicit'],
        help='Hint level: none=silent, console=logs only, subtle=element highlight, explicit=full hints (default: none)'
    )
    parser.add_argument(
        '--no-hints',
        action='store_true',
        help='Deprecated: use --hint-level none instead'
    )
    
    args = parser.parse_args()
    
    # Handle deprecated --no-hints flag
    hint_level = args.hint_level
    if args.no_hints and hint_level == 'none':
        hint_level = 'none'
    
    success = inject_semantic_conflict(
        args.input,
        args.output,
        args.react,
        args.mode,
        show_hints=not args.no_hints,
        hint_level=hint_level
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
