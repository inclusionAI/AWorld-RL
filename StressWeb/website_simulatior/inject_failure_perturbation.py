#!/usr/bin/env python3
"""
Failure Perturbation Injector
Inject operation failure perturbation - make web operations randomly fail

Usage:
    python inject_failure_perturbation.py --input <input_project_dir> [--output <output_dir>] [--react] [--probability 0.3]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def get_failure_injection_script(probability=0.3):
    """
    Generate operation failure injection script
    Intercept user interactions and randomly block or corrupt operations
    """
    return f"""
<script>
(function() {{
    'use strict';

    const FAILURE_PROBABILITY = {probability};
    const MAX_LOG_ENTRIES = 100;

    if (!localStorage.getItem('failurePerturbations')) {{
        localStorage.setItem('failurePerturbations', JSON.stringify([]));
    }}

    function logFailureEvent(eventType, targetInfo, failureType) {{
        try {{
            let logs = JSON.parse(localStorage.getItem('failurePerturbations') || '[]');
            logs.push({{
                timestamp: new Date().toISOString(),
                eventType: eventType,
                targetInfo: targetInfo,
                failureType: failureType
            }});

            if (logs.length > MAX_LOG_ENTRIES) {{
                logs = logs.slice(-MAX_LOG_ENTRIES);
            }}

            localStorage.setItem('failurePerturbations', JSON.stringify(logs));
        }} catch (e) {{
            console.error('Failed to log failure event:', e);
        }}
    }}

    function shouldFail() {{
        return Math.random() < FAILURE_PROBABILITY;
    }}

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

    function isTextInputTarget(target) {{
        if (!target) return false;
        const tag = (target.tagName || '').toLowerCase();
        if (tag === 'textarea') return true;
        if (tag !== 'input') return false;
        const type = (target.type || 'text').toLowerCase();
        return ['text', 'search', 'email', 'url', 'tel', 'password', 'number'].includes(type);
    }}

    function isContentEditableTarget(target) {{
        if (!target) return false;
        return Boolean(target.isContentEditable || target.getAttribute?.('contenteditable') === 'true');
    }}

    function getCaretOffsetInElement(element) {{
        const selection = window.getSelection?.();
        if (!selection || selection.rangeCount === 0) return 0;
        const range = selection.getRangeAt(0);
        if (!element.contains(range.startContainer)) return 0;

        const preRange = range.cloneRange();
        preRange.selectNodeContents(element);
        preRange.setEnd(range.startContainer, range.startOffset);
        return preRange.toString().length;
    }}

    function setCaretOffsetInElement(element, offset) {{
        const selection = window.getSelection?.();
        if (!selection) return;

        const range = document.createRange();
        let currentOffset = 0;
        let found = false;

        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null);
        while (walker.nextNode()) {{
            const node = walker.currentNode;
            const nextOffset = currentOffset + node.textContent.length;
            if (offset <= nextOffset) {{
                range.setStart(node, Math.max(0, offset - currentOffset));
                range.collapse(true);
                found = true;
                break;
            }}
            currentOffset = nextOffset;
        }}

        if (!found) {{
            range.selectNodeContents(element);
            range.collapse(false);
        }}

        selection.removeAllRanges();
        selection.addRange(range);
    }}

    const FailureTypes = {{
        BLOCK: 'block',
        DELAY: 'delay',
        WRONG_TARGET: 'wrong',
        PARTIAL: 'partial',
        ERROR: 'error'
    }};

    function getRandomFailureType(eventType) {{
        const types = [FailureTypes.BLOCK, FailureTypes.DELAY];

        if (eventType === 'input' || eventType === 'change') {{
            types.push(FailureTypes.PARTIAL);
        }}

        return types[Math.floor(Math.random() * types.length)];
    }}

    const clickInterceptor = async function(event) {{
        if (!shouldFail()) return;

        const failureType = getRandomFailureType('click');
        const elementInfo = getElementInfo(event.target);

        console.log(`[Failure Perturbation] Click failure injected: ${{failureType}} on ${{elementInfo}}`);
        logFailureEvent('click', elementInfo, failureType);

        switch (failureType) {{
            case FailureTypes.BLOCK:
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                break;

            case FailureTypes.DELAY:
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();

                setTimeout(() => {{
                    console.log('[Failure Perturbation] Delayed click executed');
                    document.removeEventListener('click', clickInterceptor, true);
                    event.target.click();
                    setTimeout(() => {{
                        document.addEventListener('click', clickInterceptor, true);
                    }}, 100);
                }}, 3000);
                break;
        }}
    }};

    const inputInterceptor = function(event) {{
        if (!shouldFail()) return;

        const failureType = getRandomFailureType('input');
        const elementInfo = getElementInfo(event.target);

        console.log(`[Failure Perturbation] Input failure injected: ${{failureType}} on ${{elementInfo}}`);
        logFailureEvent('input', elementInfo, failureType);

        switch (failureType) {{
            case FailureTypes.BLOCK:
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                break;

            case FailureTypes.PARTIAL:
                setTimeout(() => {{
                    const target = event.target;
                    if (isTextInputTarget(target) && typeof target.value === 'string') {{
                        const start = target.selectionStart ?? target.value.length;
                        const end = target.selectionEnd ?? target.value.length;
                        const original = target.value;
                        const corrupted = original.split('').filter(() => Math.random() > 0.3).join('');
                        target.value = corrupted;
                        const safeStart = Math.min(start, corrupted.length);
                        const safeEnd = Math.min(end, corrupted.length);
                        target.setSelectionRange?.(safeStart, safeEnd);
                        target.focus?.();
                        console.log(`[Failure Perturbation] Input corrupted: "${{original}}" -> "${{corrupted}}"`);
                    }} else if (isContentEditableTarget(target)) {{
                        const caretOffset = getCaretOffsetInElement(target);
                        const original = target.innerText || target.textContent || '';
                        if (!original) return;
                        const corrupted = original.split('').filter(() => Math.random() > 0.3).join('');
                        target.textContent = corrupted;
                        const safeOffset = Math.min(caretOffset, corrupted.length);
                        setCaretOffsetInElement(target, safeOffset);
                        console.log(`[Failure Perturbation] Editable corrupted: "${{original}}" -> "${{corrupted}}"`);
                    }}
                }}, 100);
                break;
        }}
    }};

    const beforeInputInterceptor = function(event) {{
        const target = event.target;
        if (!isContentEditableTarget(target)) return;
        if (!shouldFail()) return;

        const failureType = getRandomFailureType('input');
        const elementInfo = getElementInfo(target);
        console.log(`[Failure Perturbation] BeforeInput failure injected: ${{failureType}} on ${{elementInfo}}`);
        logFailureEvent('beforeinput', elementInfo, failureType);

        if (failureType === FailureTypes.BLOCK) {{
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
        }} else if (failureType === FailureTypes.PARTIAL) {{
            setTimeout(() => {{
                const caretOffset = getCaretOffsetInElement(target);
                const original = target.innerText || target.textContent || '';
                if (!original) return;
                const corrupted = original.split('').filter(() => Math.random() > 0.3).join('');
                target.textContent = corrupted;
                const safeOffset = Math.min(caretOffset, corrupted.length);
                setCaretOffsetInElement(target, safeOffset);
                console.log(`[Failure Perturbation] Editable corrupted: "${{original}}" -> "${{corrupted}}"`);
            }}, 50);
        }}
    }};

    const submitInterceptor = function(event) {{
        if (!shouldFail()) return;

        const failureType = FailureTypes.BLOCK;
        const elementInfo = getElementInfo(event.target);

        console.log(`[Failure Perturbation] Submit failure injected: ${{failureType}} on ${{elementInfo}}`);
        logFailureEvent('submit', elementInfo, failureType);

        event.preventDefault();
        event.stopPropagation();

        const errorMsg = document.createElement('div');
        errorMsg.textContent = '⚠️ Operation failed. Please try again.';
        errorMsg.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #ff4444;
            color: white;
            padding: 20px 40px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            z-index: 999999;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        document.body.appendChild(errorMsg);

        setTimeout(() => {{
            errorMsg.remove();
        }}, 2000);
    }};

    const buttonInterceptor = function(event) {{
        const target = event.target;
        const isButton = target.tagName === 'BUTTON' ||
            target.type === 'submit' ||
            target.type === 'button' ||
            target.getAttribute('role') === 'button';

        if (!isButton || !shouldFail()) return;

        const failureType = getRandomFailureType('button');
        const elementInfo = getElementInfo(target);

        console.log(`[Failure Perturbation] Button failure injected: ${{failureType}} on ${{elementInfo}}`);
        logFailureEvent('button', elementInfo, failureType);

        if (failureType === FailureTypes.BLOCK) {{
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
        }}
    }};

    const keyInterceptor = function(event) {{
        if (event.key === 'Enter' && shouldFail()) {{
            const failureType = FailureTypes.BLOCK;
            const elementInfo = getElementInfo(event.target);

            console.log(`[Failure Perturbation] Enter key failure injected on ${{elementInfo}}`);
            logFailureEvent('keypress', elementInfo, failureType);

            event.preventDefault();
            event.stopPropagation();
        }}
    }};

    const changeInterceptor = function(event) {{
        if (!shouldFail()) return;

        const failureType = getRandomFailureType('change');
        const elementInfo = getElementInfo(event.target);

        console.log(`[Failure Perturbation] Change failure injected: ${{failureType}} on ${{elementInfo}}`);
        logFailureEvent('change', elementInfo, failureType);

        if (failureType === FailureTypes.BLOCK) {{
            setTimeout(() => {{
                if (event.target.type === 'checkbox' || event.target.type === 'radio') {{
                    event.target.checked = !event.target.checked;
                }} else if (event.target.tagName === 'SELECT') {{
                    event.target.selectedIndex = 0;
                }}
            }}, 0);
        }}
    }};

    document.addEventListener('click', clickInterceptor, true);
    document.addEventListener('beforeinput', beforeInputInterceptor, true);
    document.addEventListener('input', inputInterceptor, true);
    document.addEventListener('submit', submitInterceptor, true);
    document.addEventListener('click', buttonInterceptor, true);
    document.addEventListener('keypress', keyInterceptor, true);
    document.addEventListener('change', changeInterceptor, true);

    window.getFailureStats = function() {{
        const logs = JSON.parse(localStorage.getItem('failurePerturbations') || '[]');
        const stats = {{
            total: logs.length,
            byType: {{}},
            byFailureType: {{}},
            recent: logs.slice(-10)
        }};

        logs.forEach(log => {{
            stats.byType[log.eventType] = (stats.byType[log.eventType] || 0) + 1;
            stats.byFailureType[log.failureType] = (stats.byFailureType[log.failureType] || 0) + 1;
        }});

        console.log('Failure Perturbation Statistics:', stats);
        return stats;
    }};

    console.log(`[Failure Perturbation] Injected with ${{FAILURE_PROBABILITY * 100}}% failure rate`);
    console.log('[Failure Perturbation] Use window.getFailureStats() to view statistics');
}})();
</script>
"""


def inject_to_html(html_path, script_content):
    """Inject script into HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Inject script before </head> tag
        if '</head>' in content:
            content = content.replace('</head>', f'{script_content}\n</head>')
        # If no </head> tag, inject before <body> tag
        elif '<body>' in content or '<body ' in content:
            import re
            content = re.sub(r'(<body[^>]*>)', f'{script_content}\n\\1', content, count=1)
        else:
            # If neither exists, add to beginning of file
            content = script_content + '\n' + content
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Error injecting to {html_path}: {e}")
        return False


def inject_failure_perturbation(input_dir, output_dir=None, is_react=False, probability=0.3):
    """
    Inject operation failure perturbation into web project

    Args:
        input_dir: Input project directory
        output_dir: Output directory (if None, add _failed suffix to original directory)
        is_react: Whether it's a React project
        probability: Failure probability (between 0-1)
    """
    input_path = Path(input_dir).resolve()

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return False

    # Determine output directory
    if output_dir is None:
        output_path = Path(str(input_path) + '_failed')
    else:
        output_path = Path(output_dir).resolve()

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Failure probability: {probability * 100}%")

    # If output directory already exists, delete it first
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    # Copy entire project (excluding node_modules)
    print("Copying project files...")
    try:
        shutil.copytree(
            input_path,
            output_path,
            ignore=shutil.ignore_patterns('node_modules', '__pycache__', '*.pyc', '.git')
        )
    except Exception as e:
        print(f"Error copying project: {e}")
        return False

    # Generate injection script
    script_content = get_failure_injection_script(probability)

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
            print(f"Injecting failure perturbation to: {html_path.relative_to(output_path)}")
            if inject_to_html(html_path, script_content):
                injected_count += 1
    
    if injected_count == 0:
        print("Warning: No HTML files were injected!")
        return False
    
    print(f"\n✅ Failure perturbation injection completed!")
    print(f"   Injected to {injected_count} HTML file(s)")
    print(f"   Failure rate: {probability * 100}%")
    print(f"   Output directory: {output_path}")
    print(f"\n📊 To view failure statistics in browser console:")
    print(f"   window.getFailureStats()")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Inject operation failure perturbations into web projects'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input project directory path'
    )
    parser.add_argument(
        '--output',
        help='Output directory path (default: input_dir + "_failed")'
    )
    parser.add_argument(
        '--react',
        action='store_true',
        default=True,
        help='Treat as React project (inject to frontend/public/index.html)'
    )
    parser.add_argument(
        '--probability',
        type=float,
        default=0.35,
        help='Failure probability (0-1, default: 0.35 for 35%%)'
    )
    
    args = parser.parse_args()
    
    # Validate probability parameter
    if not 0 <= args.probability <= 1:
        print("Error: Probability must be between 0 and 1")
        sys.exit(1)
    
    success = inject_failure_perturbation(
        args.input,
        args.output,
        args.react,
        args.probability
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
