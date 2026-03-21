#!/usr/bin/env python3
"""
Layout Chaos Perturbation Injector
Layout chaos perturbation injector - disrupts page layout without covering information

Randomly changes element position, size, spacing, alignment, etc., to test Agent's adaptability to visual layout changes.

Usage:
    python inject_layout_chaos_perturbation.py --input <project_dir> [--react] [--intensity medium]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def get_layout_chaos_script(intensity="medium", seed=None):
    """
    Generate layout chaos injection script - using CSS injection for better efficiency

    Args:
        intensity: Chaos level
            - "mild": Light chaos, minor adjustments
            - "medium": Medium chaos, noticeable but usable
            - "extreme": Extreme chaos, major changes
        seed: Random seed for reproducible chaos effects
    """

    # Set parameters based on intensity
    intensity_configs = {
        "mild": {
            "scale_range": "0.75, 1.05",
            "rotate_range": "-3, 3",
            "translate_range": "-8, 8",
            "skew_range": "-2, 2",
            "margin_extra": "8",
            "padding_extra": "5",
            "font_scale": "0.88, 1.03",
            "letter_spacing": "-0.5, 1",
            "line_height_range": "1.2, 1.6",
            "border_radius_range": "0, 12",
        },
        "medium": {
            "scale_range": "0.65, 1.2",
            "rotate_range": "-8, 8",
            "translate_range": "-20, 20",
            "skew_range": "-4, 4",
            "margin_extra": "15",
            "padding_extra": "12",
            "font_scale": "0.8, 1.1",
            "letter_spacing": "-1, 2",
            "line_height_range": "1.0, 1.8",
            "border_radius_range": "0, 20",
        },
        "extreme": {
            "scale_range": "0.4, 1.4",
            "rotate_range": "-15, 15",
            "translate_range": "-35, 35",
            "skew_range": "-8, 8",
            "margin_extra": "30",
            "padding_extra": "20",
            "font_scale": "0.6, 1.5",
            "letter_spacing": "-2, 4",
            "line_height_range": "0.9, 2.2",
            "border_radius_range": "0, 30",
        }
    }
    
    config = intensity_configs.get(intensity, intensity_configs["medium"])
    seed_value = seed if seed is not None else "null"
    
    return f"""
<style id="layout-chaos-base-styles">
/* Layout Chaos Perturbation - Base Styles */

/* Ensure transformations don't cause content to be clipped */
body {{
    overflow-x: hidden;
}}
</style>

<script>
(function() {{
    'use strict';

    const INTENSITY = '{intensity}';
    const SEED = {seed_value};
    const CONFIG = {{
        scaleRange: [{config['scale_range']}],
        rotateRange: [{config['rotate_range']}],
        translateRange: [{config['translate_range']}],
        skewRange: [{config['skew_range']}],
        marginExtra: {config['margin_extra']},
        paddingExtra: {config['padding_extra']},
        fontScale: [{config['font_scale']}],
        letterSpacing: [{config['letter_spacing']}],
        lineHeightRange: [{config['line_height_range']}],
        borderRadiusRange: [{config['border_radius_range']}]
    }};

    // Minimum size protection (pixels)
    const MIN_SIZES = {{
        button: 28,
        img: 20,
        svg: 14,
        input: 24,
        a: 20,
        i: 14,
        span: 12
    }};
    
    // Pseudo-random number generator with fixed seed (based on element features, ensures consistent results each time)
    function hashCode(str) {{
        let hash = 5381;
        for (let i = 0; i < str.length; i++) {{
            const char = str.charCodeAt(i);
            hash = ((hash << 5) + hash) ^ char; // hash * 33 ^ char
        }}
        return hash;
    }}

    // Generate stable random number based on element
    function getElementHash(element) {{
        const tag = element.tagName || '';
        const id = element.id || '';
        const className = (element.className || '').toString();
        const text = (element.textContent || '').substring(0, 20);
        const rect = element.getBoundingClientRect();
        // Add more variation factors
        const extra = Math.floor(rect.width) + '_' + Math.floor(rect.height);
        return hashCode(tag + id + className + text + Math.floor(rect.left) + Math.floor(rect.top) + extra);
    }}

    // Seed-based random number - using better mixing algorithm
    function seededRandomFromHash(hash, index) {{
        // Using xorshift algorithm variant for better distribution
        let x = Math.abs(hash) ^ (index * 2654435769);
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return (Math.abs(x) % 10000) / 10000;
    }}

    function randomInRangeWithHash(hash, index, min, max) {{
        const rand = seededRandomFromHash(hash, index);
        return min + rand * (max - min);
    }}

    // Check if near right edge (may be clipped)
    function isNearRightEdge(element) {{
        const rect = element.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        return rect.right > viewportWidth - 80;
    }}

    // Elements that should be skipped
    function shouldSkip(element) {{
        if (!element || !element.tagName) return true;

        const tagName = element.tagName.toLowerCase();
        if (['html', 'body', 'head', 'script', 'style', 'meta', 'link', 'title', 'noscript', 'br', 'hr'].includes(tagName)) {{
            return true;
        }}

        // Skip invisible elements
        try {{
            const style = window.getComputedStyle(element);
            if (style.display === 'none' || style.visibility === 'hidden') {{
                return true;
            }}
        }} catch (e) {{
            return true;
        }}

        // Skip too small elements
        const rect = element.getBoundingClientRect();
        if (rect.width < 3 || rect.height < 3) {{
            return true;
        }}

        // Skip already processed elements
        if (element.dataset.chaosApplied === 'true') {{
            return true;
        }}

        return false;
    }}
    
    // Generate chaos transformations
    function generateChaosStyles(element) {{
        const hash = getElementHash(element);
        const tagName = element.tagName.toLowerCase();
        const rect = element.getBoundingClientRect();
        const className = (element.className || '').toString().toLowerCase();

        // Check if near right edge
        const nearRightEdge = isNearRightEdge(element);

        // Calculate safe scaling range
        const minSize = MIN_SIZES[tagName] || 12;
        let scaleMin = CONFIG.scaleRange[0];
        let scaleMax = CONFIG.scaleRange[1];

        if (rect.width > 0 && rect.width * scaleMin < minSize) {{
            scaleMin = Math.max(scaleMin, minSize / rect.width);
        }}
        if (rect.height > 0 && rect.height * scaleMin < minSize) {{
            scaleMin = Math.max(scaleMin, minSize / rect.height);
        }}

        // Limit rightward translation for elements near right edge
        let translateXMin = CONFIG.translateRange[0];
        let translateXMax = CONFIG.translateRange[1];
        if (nearRightEdge) {{
            translateXMax = Math.min(translateXMax, 5);
        }}

        const styles = {{}};
        let idx = 0;

        // 1. Scale (70% probability)
        if (seededRandomFromHash(hash, idx++) > 0.3) {{
            const scale = randomInRangeWithHash(hash, idx++, scaleMin, scaleMax);
            styles.scale = scale;
        }}

        // 2. Rotate (60% probability)
        if (seededRandomFromHash(hash, idx++) > 0.4) {{
            const rotate = randomInRangeWithHash(hash, idx++, CONFIG.rotateRange[0], CONFIG.rotateRange[1]);
            styles.rotate = rotate;
        }}

        // 3. Translate (50% probability)
        if (seededRandomFromHash(hash, idx++) > 0.5) {{
            const translateX = randomInRangeWithHash(hash, idx++, translateXMin, translateXMax);
            const translateY = randomInRangeWithHash(hash, idx++, CONFIG.translateRange[0], CONFIG.translateRange[1]);
            styles.translateX = translateX;
            styles.translateY = translateY;
        }}

        // 4. Skew (40% probability) - Random for both X and Y directions
        if (seededRandomFromHash(hash, idx++) > 0.6) {{
            const skewX = randomInRangeWithHash(hash, idx++, CONFIG.skewRange[0], CONFIG.skewRange[1]);
            // Randomly decide whether to also apply Y direction skew
            if (seededRandomFromHash(hash, idx++) > 0.5) {{
                const skewY = randomInRangeWithHash(hash, idx++, CONFIG.skewRange[0], CONFIG.skewRange[1]);
                styles.skewX = skewX;
                styles.skewY = skewY;
            }} else {{
                // Randomly choose to use only X or only Y
                if (seededRandomFromHash(hash, idx++) > 0.5) {{
                    styles.skewX = skewX;
                }} else {{
                    styles.skewY = skewX; // Use same value but apply to Y
                }}
            }}
        }}

        // 5. Margin increase (50% probability)
        if (seededRandomFromHash(hash, idx++) > 0.5) {{
            const marginSide = ['Top', 'Right', 'Bottom', 'Left'][Math.floor(seededRandomFromHash(hash, idx++) * 4)];
            const marginValue = Math.floor(randomInRangeWithHash(hash, idx++, 0, CONFIG.marginExtra));
            styles.marginSide = marginSide;
            styles.marginValue = marginValue;
        }}

        // 6. Font changes (for text elements, 60% probability)
        if (['p', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'button', 'label', 'li', 'div'].includes(tagName)) {{
            if (seededRandomFromHash(hash, idx++) > 0.4) {{
                const fontScale = randomInRangeWithHash(hash, idx++, CONFIG.fontScale[0], CONFIG.fontScale[1]);
                styles.fontScale = fontScale;
            }}
            if (seededRandomFromHash(hash, idx++) > 0.6) {{
                const letterSpacing = randomInRangeWithHash(hash, idx++, CONFIG.letterSpacing[0], CONFIG.letterSpacing[1]);
                styles.letterSpacing = letterSpacing;
            }}
        }}

        // 7. Border radius (40% probability)
        if (seededRandomFromHash(hash, idx++) > 0.6) {{
            const borderRadius = Math.floor(randomInRangeWithHash(hash, idx++, CONFIG.borderRadiusRange[0], CONFIG.borderRadiusRange[1]));
            styles.borderRadius = borderRadius;
        }}

        // 8. Shadow (25% probability)
        if (seededRandomFromHash(hash, idx++) > 0.75) {{
            const shadowBlur = Math.floor(randomInRangeWithHash(hash, idx++, 3, 15));
            const shadowY = Math.floor(randomInRangeWithHash(hash, idx++, 2, 10));
            const shadowAlpha = randomInRangeWithHash(hash, idx++, 0.1, 0.3);
            styles.shadow = {{ blur: shadowBlur, y: shadowY, alpha: shadowAlpha }};
        }}

        return styles;
    }}
    
    // Apply chaos styles to element
    function applyChaosToElement(element) {{
        if (shouldSkip(element)) return false;

        const styles = generateChaosStyles(element);

        // Build transform string
        let transformParts = [];
        if (styles.scale !== undefined) {{
            transformParts.push(`scale(${{styles.scale.toFixed(3)}})`);
        }}
        if (styles.rotate !== undefined) {{
            transformParts.push(`rotate(${{styles.rotate.toFixed(2)}}deg)`);
        }}
        if (styles.translateX !== undefined || styles.translateY !== undefined) {{
            const tx = (styles.translateX || 0).toFixed(1);
            const ty = (styles.translateY || 0).toFixed(1);
            transformParts.push(`translate(${{tx}}px, ${{ty}}px)`);
        }}
        if (styles.skewX !== undefined) {{
            transformParts.push(`skewX(${{styles.skewX.toFixed(2)}}deg)`);
        }}
        if (styles.skewY !== undefined) {{
            transformParts.push(`skewY(${{styles.skewY.toFixed(2)}}deg)`);
        }}

        // Apply styles
        if (transformParts.length > 0) {{
            element.style.transform = transformParts.join(' ');
            element.style.transformOrigin = 'center center';
        }}

        if (styles.marginSide && styles.marginValue) {{
            const currentMargin = parseInt(window.getComputedStyle(element)[`margin${{styles.marginSide}}`]) || 0;
            element.style[`margin${{styles.marginSide}}`] = (currentMargin + styles.marginValue) + 'px';
        }}

        if (styles.fontScale !== undefined) {{
            const currentSize = parseFloat(window.getComputedStyle(element).fontSize) || 16;
            element.style.fontSize = Math.max(9, currentSize * styles.fontScale).toFixed(1) + 'px';
        }}

        if (styles.letterSpacing !== undefined) {{
            element.style.letterSpacing = styles.letterSpacing.toFixed(1) + 'px';
        }}

        if (styles.borderRadius !== undefined) {{
            element.style.borderRadius = styles.borderRadius + 'px';
        }}

        if (styles.shadow) {{
            element.style.boxShadow = `0 ${{styles.shadow.y}}px ${{styles.shadow.blur}}px rgba(0,0,0,${{styles.shadow.alpha.toFixed(2)}})`;
        }}

        element.dataset.chaosApplied = 'true';
        return true;
    }}

    // Process all elements on the page
    function processAllElements() {{
        console.log('[Layout Chaos] 🌀 Processing elements...');

        // Broader selectors covering more elements + calendar-specific selectors
        const selectors = [
            // Buttons and links
            'button', 'a', '[role="button"]',
            // Images and icons
            'img', 'svg', 'i[class*="fa"]', 'i[class*="icon"]', '[class*="icon"]', '[class*="Icon"]',
            // Cards and containers
            '[class*="card"]', '[class*="Card"]', '[class*="item"]', '[class*="Item"]',
            '[class*="product"]', '[class*="Product"]', '[class*="restaurant"]', '[class*="Restaurant"]',
            // Headings and text
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'label',
            // Lists
            'li', 'ul > *', 'ol > *',
            // Navigation and menus
            'nav *', 'header *', '[class*="nav"] *', '[class*="menu"] *', '[class*="Menu"] *',
            // Form elements
            'input', 'select', 'textarea',
            // Generic containers (limited levels)
            'main > div', 'main > section', 'section > div',
            'article', 'aside', 'footer *',
            // Calendar-specific selectors
            '[class*="calendar"]', '[class*="Calendar"]',
            '[class*="event"]', '[class*="Event"]',
            '[class*="day"]', '[class*="Day"]',
            '[class*="week"]', '[class*="Week"]',
            '[class*="month"]', '[class*="Month"]',
            '[class*="time"]', '[class*="Time"]',
            '[class*="date"]', '[class*="Date"]',
            '[class*="mini"]', '[class*="Mini"]',
            '[class*="grid"]', '[class*="Grid"]',
            '[class*="slot"]', '[class*="Slot"]',
            '[class*="header"]', '[class*="Header"]',
            '[class*="cell"]', '[class*="Cell"]',
            '[class*="filter"]', '[class*="Filter"]',
            '[class*="category"]', '[class*="Category"]'
        ];

        let processedCount = 0;
        const processed = new Set();

        selectors.forEach(selector => {{
            try {{
                document.querySelectorAll(selector).forEach(el => {{
                    // Avoid duplicate processing
                    if (processed.has(el)) return;
                    processed.add(el);

                    if (applyChaosToElement(el)) {{
                        processedCount++;
                    }}
                }});
            }} catch (e) {{
                // Ignore selector errors
            }}
        }});

        console.log('[Layout Chaos] ✅ Processed', processedCount, 'elements');
        return processedCount;
    }}

    // Clear all chaos effects
    function clearChaos() {{
        document.querySelectorAll('[data-chaos-applied]').forEach(el => {{
            el.style.transform = '';
            el.style.transformOrigin = '';
            el.style.boxShadow = '';
            el.style.borderRadius = '';
            el.style.fontSize = '';
            el.style.letterSpacing = '';
            // Don't clear margin as it may affect layout
            delete el.dataset.chaosApplied;
        }});
    }}

    // Wait for content to load
    function waitForContent() {{
        return new Promise(resolve => {{
            let attempts = 0;
            const maxAttempts = 25;
            
            function check() {{
                attempts++;
                const root = document.getElementById('root');
                
                if (root && root.children.length > 0) {{
                    const text = (root.textContent || '').toLowerCase();
                    if (!text.includes('loading') || attempts > 15) {{
                        resolve();
                        return;
                    }}
                }}
                
                if (attempts >= maxAttempts) {{
                    resolve();
                    return;
                }}
                
                setTimeout(check, 150);
            }}
            
            check();
        }});
    }}

    // Main initialization function
    async function init() {{
        await waitForContent();

        // First application
        setTimeout(() => {{
            processAllElements();
        }}, 500);

        // Listen for URL changes (React Router) - use multiple methods to ensure capture
        let lastUrl = location.href;
        let lastPathname = location.pathname;

        // Method 1: MutationObserver to monitor DOM changes
        const urlObserver = new MutationObserver(() => {{
            if (location.href !== lastUrl || location.pathname !== lastPathname) {{
                lastUrl = location.href;
                lastPathname = location.pathname;
                console.log('[Layout Chaos] 🔄 URL changed to:', location.pathname);
                // Clear old marks and reapply
                document.querySelectorAll('[data-chaos-applied]').forEach(el => {{
                    delete el.dataset.chaosApplied;
                }});
                setTimeout(processAllElements, 500);
            }}
        }});

        urlObserver.observe(document.body, {{ childList: true, subtree: true }});

        // Method 2: Listen for popstate event (browser forward/back)
        window.addEventListener('popstate', () => {{
            console.log('[Layout Chaos] 🔄 Navigation detected (popstate)');
            document.querySelectorAll('[data-chaos-applied]').forEach(el => {{
                delete el.dataset.chaosApplied;
            }});
            setTimeout(processAllElements, 500);
        }});

        // Method 3: Periodically check for URL changes (fallback)
        setInterval(() => {{
            if (location.href !== lastUrl || location.pathname !== lastPathname) {{
                lastUrl = location.href;
                lastPathname = location.pathname;
                console.log('[Layout Chaos] 🔄 URL changed (polling):', location.pathname);
                document.querySelectorAll('[data-chaos-applied]').forEach(el => {{
                    delete el.dataset.chaosApplied;
                }});
                setTimeout(processAllElements, 300);
            }}
        }}, 500);

        // Method 4: Monitor main container content changes (React re-renders)
        const rootObserver = new MutationObserver((mutations) => {{
            // Check if there are significant DOM changes (possible view switching)
            let significantChange = false;
            for (const mutation of mutations) {{
                if (mutation.addedNodes.length > 5 || mutation.removedNodes.length > 5) {{
                    significantChange = true;
                    break;
                }}
            }}

            if (significantChange) {{
                console.log('[Layout Chaos] 🔄 Significant DOM change detected');
                // Clear old marks and reapply
                setTimeout(() => {{
                    document.querySelectorAll('[data-chaos-applied]').forEach(el => {{
                        delete el.dataset.chaosApplied;
                    }});
                    processAllElements();
                }}, 400);
            }}
        }});

        // Observe root or app container
        const rootElement = document.getElementById('root') || document.getElementById('app') || document.body;
        if (rootElement) {{
            rootObserver.observe(rootElement, {{
                childList: true,
                subtree: true,
                attributes: false // Only care about structure changes, not attributes
            }});
        }}

        // Method 5: Periodically check for new elements (handle dynamic loading)
        setInterval(() => {{
            const unprocessed = document.querySelectorAll('button:not([data-chaos-applied]), [class*=\"event\"]:not([data-chaos-applied]), [class*=\"day\"]:not([data-chaos-applied])');
            if (unprocessed.length > 10) {{
                console.log('[Layout Chaos] 🔄 Found', unprocessed.length, 'unprocessed elements');
                processAllElements();
            }}
        }}, 1500);
    }}
    
    // Export control functions
    window.getLayoutChaosStats = function() {{
        const count = document.querySelectorAll('[data-chaos-applied]').length;
        console.log('=== Layout Chaos Statistics ===');
        console.log('Intensity:', INTENSITY);
        console.log('Elements with chaos:', count);
        console.log('Config:', CONFIG);
        return {{ intensity: INTENSITY, count, config: CONFIG }};
    }};
    
    window.resetLayoutChaos = function() {{
        clearChaos();
        console.log('[Layout Chaos] Reset complete');
    }};
    
    window.reapplyLayoutChaos = function() {{
        clearChaos();
        setTimeout(processAllElements, 100);
        console.log('[Layout Chaos] Reapplied');
    }};
    
    // Startup
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', init);
    }} else {{
        init();
    }}
    
    console.log('🌀 Layout Chaos Perturbation Loaded');
    console.log('   Intensity:', INTENSITY);
    console.log('   Use: getLayoutChaosStats(), resetLayoutChaos(), reapplyLayoutChaos()');
}})();
</script>
"""


def inject_to_html(html_path, script_content):
    """Inject script into HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already injected
        if 'Layout Chaos' in content:
            print(f"⚠️  Already injected, skipping: {html_path}")
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


def inject_layout_chaos(
    input_dir,
    output_dir=None,
    is_react=False,
    intensity="medium",
    seed=None
):
    """
    Inject layout chaos perturbation into web project
    
    Args:
        input_dir: Input project directory
        output_dir: Output directory (if None, add _chaos suffix to original directory)
        is_react: Whether it's a React project
        intensity: Chaos level (mild/medium/extreme)
        seed: Random seed
    """
    input_path = Path(input_dir).resolve()
    
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {input_path}")
        return False
    
    # Determine output directory
    if output_dir is None:
        output_path = Path(str(input_path) + f'_chaos_{intensity}')
    else:
        output_path = Path(output_dir).resolve()
    
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"🌀 Intensity: {intensity}")
    if seed:
        print(f"🎲 Seed: {seed}")
    
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
    script_content = get_layout_chaos_script(intensity=intensity, seed=seed)
    
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
            print(f"💉 Injecting layout chaos to: {html_path.relative_to(output_path)}")
            if inject_to_html(html_path, script_content):
                injected_count += 1
    
    if injected_count == 0:
        print("⚠️  Warning: No HTML files were injected!")
        return False
    
    print(f"\n{'='*80}")
    print(f"✅ Layout chaos perturbation injection completed!")
    print(f"   Injected to {injected_count} HTML file(s)")
    print(f"   Intensity: {intensity}")
    print(f"   Effects:")
    print(f"     • Random element scaling (min size preserved for buttons/images)")
    print(f"     • Random rotation and translation")
    print(f"     • Random margin/padding/gap changes")
    print(f"     • Random font size variations")
    print(f"     • Random border and shadow effects")
    print(f"     • Flex/Grid layout alignment changes")
    print(f"     • Element order shuffling")
    print(f"   Output: {output_path}")
    print(f"{'='*80}")
    
    print(f"\n📊 To interact with chaos in browser console:")
    print(f"   window.getLayoutChaosStats()   - View statistics")
    print(f"   window.reapplyLayoutChaos()    - Reapply with new random values")
    print(f"   window.resetLayoutChaos()      - Restore original styles")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Inject layout chaos perturbations into web projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mild chaos (slight visual changes)
  python inject_layout_chaos_perturbation.py --input ./generated-website --intensity mild
  
  # Medium chaos (noticeable but usable)
  python inject_layout_chaos_perturbation.py --input ./website --react --intensity medium
  
  # Extreme chaos (significant visual disruption)
  python inject_layout_chaos_perturbation.py --input ./site --intensity extreme
  
  # With specific seed for reproducible results
  python inject_layout_chaos_perturbation.py --input ./site --intensity medium --seed 12345
  
  # Custom output directory
  python inject_layout_chaos_perturbation.py --input ./site --output ./site_chaos
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input project directory path'
    )
    parser.add_argument(
        '--output',
        help='Output directory path (default: input_dir + "_chaos")'
    )
    parser.add_argument(
        '--react',
        action='store_true',
        default=True,
        help='Treat as React project (inject to frontend/public/index.html)'
    )
    parser.add_argument(
        '--intensity',
        default='medium',
        choices=['mild', 'medium', 'extreme'],
        help='Chaos intensity level (default: medium)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible chaos (default: random)'
    )
    
    args = parser.parse_args()
    
    success = inject_layout_chaos(
        args.input,
        args.output,
        args.react,
        args.intensity,
        args.seed
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
