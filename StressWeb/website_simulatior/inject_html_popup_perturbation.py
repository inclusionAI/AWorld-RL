"""
Web Perturbation Injection Script - HTML Popup Version
Use custom HTML popups instead of native confirm, allowing agent to handle them by clicking coordinates
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
import shutil
from datetime import datetime


class HTMLPopupPerturbationInjector:
    """HTML Popup Perturbation Injector"""

    def __init__(self, probability: float = 0.5):
        """
        Initialize injector

        Args:
            probability: Probability of popup appearing (0.0-1.0), default 0.5 (50%)
        """
        self.probability = probability
        self.injection_script = self._generate_injection_script()
    
    def _generate_injection_script(self) -> str:
        """Generate JavaScript code to be injected"""
        return f"""
<!-- ==================== PERTURBATION INJECTION START (HTML POPUP) ==================== -->
<style>
/* Custom popup styles */
.perturbation-dialog-overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999999;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}}

.perturbation-dialog-box {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    min-width: 400px;
    max-width: 600px;
    padding: 0;
    animation: perturbationSlideIn 0.3s ease-out;
}}

@keyframes perturbationSlideIn {{
    from {{
        transform: translateY(-50px);
        opacity: 0;
    }}
    to {{
        transform: translateY(0);
        opacity: 1;
    }}
}}

.perturbation-dialog-header {{
    padding: 20px 24px;
    border-bottom: 1px solid #e0e0e0;
}}

.perturbation-dialog-title {{
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #333;
}}

.perturbation-dialog-content {{
    padding: 24px;
    font-size: 14px;
    line-height: 1.6;
    color: #666;
    white-space: pre-wrap;
}}

.perturbation-dialog-footer {{
    padding: 16px 24px;
    border-top: 1px solid #e0e0e0;
    display: flex;
    justify-content: flex-end;
    gap: 12px;
}}

.perturbation-dialog-button {{
    padding: 12px 32px;
    border: none;
    border-radius: 6px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 120px;
    min-height: 44px;
}}

.perturbation-dialog-button-ok {{
    background: #007bff;
    color: white;
}}

.perturbation-dialog-button-ok:hover {{
    background: #0056b3;
    transform: scale(1.05);
}}

.perturbation-dialog-button-cancel {{
    background: #f0f0f0;
    color: #333;
}}

.perturbation-dialog-button-cancel:hover {{
    background: #e0e0e0;
    transform: scale(1.05);
}}

.perturbation-warning {{
    color: #ff6b00;
    font-weight: 600;
}}
</style>

<script>
(function() {{
    'use strict';
    
    const POPUP_PROBABILITY = {self.probability};

    // Two types of popup logic
    const POPUP_CONFIGS = [
        // Popup that requires accepting to continue
        {{ type: 'ACCEPT', message: 'Click OK to proceed with this action' }},
        {{ type: 'ACCEPT', message: 'Confirm: Click OK to continue' }},
        {{ type: 'ACCEPT', message: 'To perform this operation, click OK' }},
        {{ type: 'ACCEPT', message: 'Press OK to confirm and proceed' }},

        // Popup that requires rejecting to continue
        {{ type: 'REJECT', message: 'SECURITY WARNING: Click Cancel to proceed safely' }},
        {{ type: 'REJECT', message: 'ALERT: To continue, you must click Cancel' }},
        {{ type: 'REJECT', message: 'CAUTION: Press Cancel to proceed (OK will abort)' }},
        {{ type: 'REJECT', message: 'IMPORTANT: Click Cancel to confirm, OK to abort' }}
    ];
    
    let currentDialog = null;
    let dialogPromiseResolve = null;
    
    function getRandomPopupConfig() {{
        return POPUP_CONFIGS[Math.floor(Math.random() * POPUP_CONFIGS.length)];
    }}
    
    function shouldShowPopup() {{
        return Math.random() < POPUP_PROBABILITY;
    }}
    
    function getElementInfo(element) {{
        if (!element) return 'Unknown';
        let info = element.tagName.toLowerCase();
        if (element.id) info += `#${{element.id}}`;
        if (element.className) info += `.${{element.className.split(' ')[0]}}`;
        if (element.textContent) {{
            const text = element.textContent.trim().substring(0, 30);
            if (text) info += ` "${{text}}"`;
        }}
        return info;
    }}
    
    function logPopupEvent(event) {{
        try {{
            const logs = JSON.parse(localStorage.getItem('perturbation_popup_logs') || '[]');
            logs.push(event);
            if (logs.length > 100) logs.shift();
            localStorage.setItem('perturbation_popup_logs', JSON.stringify(logs));
        }} catch (e) {{
            console.error('Failed to log popup event:', e);
        }}
    }}
    
    // Create and show custom dialog
    function showCustomDialog(config, actionType, elementInfo) {{
        return new Promise((resolve) => {{
            // Remove existing dialog
            if (currentDialog) {{
                document.body.removeChild(currentDialog);
            }}

            dialogPromiseResolve = resolve;

            // Generate random button class names to prevent Agent from finding them by selector
            const randomSuffix = Math.random().toString(36).substring(2, 15);
            const okBtnClass = `perturbation-dialog-button perturbation-dialog-button-ok perturbation-ok-${{randomSuffix}}`;
            const cancelBtnClass = `perturbation-dialog-button perturbation-dialog-button-cancel perturbation-cancel-${{randomSuffix}}`;

            // Create dialog element
            const overlay = document.createElement('div');
            overlay.className = 'perturbation-dialog-overlay';
            overlay.setAttribute('data-perturbation-dialog', 'true');
            
            const isWarning = config.type === 'REJECT';
            const warningClass = isWarning ? 'perturbation-warning' : '';
            
            overlay.innerHTML = `
                <div class="perturbation-dialog-box">
                    <div class="perturbation-dialog-header">
                        <h3 class="perturbation-dialog-title ${{warningClass}}">
                            ${{isWarning ? '⚠️ ' : ''}}Confirmation Required
                        </h3>
                    </div>
                    <div class="perturbation-dialog-content">
                        ${{config.message}}
                        
                        <br><br>
                        <small style="color: #999;">
                        Action Type: ${{actionType}}<br>
                        Element: ${{elementInfo}}
                        </small>
                    </div>
                    <div class="perturbation-dialog-footer">
                        <button class="${{cancelBtnClass}}" data-action="cancel">
                            Cancel
                        </button>
                        <button class="${{okBtnClass}}" data-action="ok">
                            OK
                        </button>
                    </div>
                </div>
            `;
            
            // Add button event listeners
            const okButton = overlay.querySelector('[data-action="ok"]');
            const cancelButton = overlay.querySelector('[data-action="cancel"]');

            okButton.addEventListener('click', () => handleDialogResponse(true, config, actionType, elementInfo));
            cancelButton.addEventListener('click', () => handleDialogResponse(false, config, actionType, elementInfo));

            // Prevent clicks on overlay background
            overlay.addEventListener('click', (e) => {{
                if (e.target === overlay) {{
                    e.stopPropagation();
                    e.preventDefault();
                }}
            }});
            
            currentDialog = overlay;
            document.body.appendChild(overlay);
        }});
    }}
    
    function handleDialogResponse(userClickedOK, config, actionType, elementInfo) {{
        // Remove dialog
        if (currentDialog) {{
            document.body.removeChild(currentDialog);
            currentDialog = null;
        }}

        // Determine whether to allow operation to continue based on dialog type
        const shouldProceed = (config.type === 'ACCEPT') ? userClickedOK : !userClickedOK;

        // Log event
        logPopupEvent({{
            timestamp: new Date().toISOString(),
            actionType: actionType,
            element: elementInfo,
            popupType: config.type,
            message: config.message,
            userClickedOK: userClickedOK,
            operationProceeded: shouldProceed
        }});
        
        console.log(`🎭 Dialog: ${{config.type}}, User: ${{userClickedOK ? 'OK' : 'Cancel'}}, Result: ${{shouldProceed ? 'Proceed' : 'Block'}}`);
        
        if (dialogPromiseResolve) {{
            dialogPromiseResolve(shouldProceed);
            dialogPromiseResolve = null;
        }}
    }}
    
    // Show confirmation popup
    async function showConfirmationPopup(event, element, actionType) {{
        if (!shouldShowPopup()) {{
            return true;
        }}

        const config = getRandomPopupConfig();
        const elementInfo = getElementInfo(element);

        const shouldProceed = await showCustomDialog(config, actionType, elementInfo);
        return shouldProceed;
    }}

    // Intercept all click events
    const clickInterceptor = async function(event) {{
        const element = event.target;

        // Skip dialog elements themselves
        if (element.closest('[data-perturbation-dialog]')) {{
            return;
        }}

        // Skip certain system elements
        if (element.tagName === 'HTML' || element.tagName === 'BODY') {{
            return;
        }}

        // Pause event propagation
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();

        // Show confirmation dialog and wait for result
        const confirmed = await showConfirmationPopup(event, element, 'click');

        if (confirmed) {{
            // User confirmed, manually trigger click
            const clonedEvent = new MouseEvent('click', {{
                bubbles: true,
                cancelable: true,
                view: window,
                detail: event.detail,
                screenX: event.screenX,
                screenY: event.screenY,
                clientX: event.clientX,
                clientY: event.clientY,
                ctrlKey: event.ctrlKey,
                altKey: event.altKey,
                shiftKey: event.shiftKey,
                metaKey: event.metaKey,
                button: event.button
            }});

            // Temporarily remove listener to avoid infinite loop
            document.removeEventListener('click', clickInterceptor, true);
            element.dispatchEvent(clonedEvent);
            // Delay re-adding listener
            setTimeout(() => {{
                document.addEventListener('click', clickInterceptor, true);
            }}, 10);
        }}
    }};
    
    document.addEventListener('click', clickInterceptor, true);

    // Intercept form submission
    document.addEventListener('submit', async function(event) {{
        event.preventDefault();
        event.stopPropagation();

        const confirmed = await showConfirmationPopup(event, event.target, 'submit');
        if (confirmed) {{
            // Manually submit form
            event.target.submit();
        }}
    }}, true);

    // Console commands
    window.getPerturbationLogs = function() {{
        const logs = JSON.parse(localStorage.getItem('perturbation_popup_logs') || '[]');
        console.table(logs);
        return logs;
    }};

    window.clearPerturbationLogs = function() {{
        localStorage.removeItem('perturbation_popup_logs');
        console.log('Perturbation logs cleared');
    }};
    
    window.getPerturbationStats = function() {{
        const logs = JSON.parse(localStorage.getItem('perturbation_popup_logs') || '[]');
        const stats = {{
            total: logs.length,
            byPopupType: {{
                ACCEPT: {{ total: 0, correctChoice: 0, wrongChoice: 0 }},
                REJECT: {{ total: 0, correctChoice: 0, wrongChoice: 0 }}
            }},
            byActionType: {{}}
        }};
        
        logs.forEach(log => {{
            if (log.popupType) {{
                stats.byPopupType[log.popupType].total++;
                if (log.operationProceeded) {{
                    stats.byPopupType[log.popupType].correctChoice++;
                }} else {{
                    stats.byPopupType[log.popupType].wrongChoice++;
                }}
            }}
            
            if (!stats.byActionType[log.actionType]) {{
                stats.byActionType[log.actionType] = {{ total: 0, proceeded: 0, blocked: 0 }};
            }}
            stats.byActionType[log.actionType].total++;
            if (log.operationProceeded) {{
                stats.byActionType[log.actionType].proceeded++;
            }} else {{
                stats.byActionType[log.actionType].blocked++;
            }}
        }});
        
        console.log('=== Perturbation Statistics ===');
        console.log(`Total popups: ${{stats.total}}`);
        console.log('\\nBy Popup Type:');
        console.log(`  ACCEPT (need OK): ${{stats.byPopupType.ACCEPT.total}} total, ${{stats.byPopupType.ACCEPT.correctChoice}} proceeded, ${{stats.byPopupType.ACCEPT.wrongChoice}} blocked`);
        console.log(`  REJECT (need Cancel): ${{stats.byPopupType.REJECT.total}} total, ${{stats.byPopupType.REJECT.correctChoice}} proceeded, ${{stats.byPopupType.REJECT.wrongChoice}} blocked`);
        console.log('\\nBy Action Type:');
        console.table(stats.byActionType);
        return stats;
    }};
    
    console.log('🎭 HTML Popup Perturbation Injected (Probability: ' + (POPUP_PROBABILITY * 100) + '%)');
    console.log('💡 Custom HTML dialogs that Agent can see and click');
    console.log('💡 Use getPerturbationLogs() to view logs');
    console.log('💡 Use getPerturbationStats() to view statistics');
}})();
</script>
<!-- ==================== PERTURBATION INJECTION END (HTML POPUP) ==================== -->
"""
    
    def inject_to_file(self, input_file: str, output_file: str = None) -> bool:
        """Inject perturbation code into a single HTML file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'PERTURBATION INJECTION' in content:
                print(f"⚠️  File already contains perturbation injection: {input_file}")
                return False
            
            injection_point = None
            if '</head>' in content:
                injection_point = content.find('</head>')
                injected_content = content[:injection_point] + self.injection_script + '\n' + content[injection_point:]
            elif '<body' in content:
                body_end = content.find('>', content.find('<body'))
                if body_end != -1:
                    injection_point = body_end + 1
                    injected_content = content[:injection_point] + '\n' + self.injection_script + content[injection_point:]
            else:
                injected_content = self.injection_script + '\n' + content
            
            if output_file is None:
                output_file = input_file
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(injected_content)
            
            print(f"✅ Successfully injected HTML popup perturbation: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to inject perturbation to {input_file}: {e}")
            return False
    
    def inject_to_react_app(self, project_dir: str, output_dir: str = None) -> bool:
        """Inject perturbation code into React application"""
        project_path = Path(project_dir)

        if output_dir is None:
            output_path = project_path.parent / f"{project_path.name}_perturbed"
        else:
            output_path = Path(output_dir)

        print(f"\n🔧 Processing React app: {project_dir}")
        print(f"📤 Output directory: {output_path}")

        if output_path.exists():
            print(f"⚠️  Output directory already exists, cleaning up...")
            shutil.rmtree(output_path)
        
        def ignore_patterns(dir, files):
            ignore = []
            for f in files:
                if f in ['node_modules', 'build', 'dist', '.git', '.next', 'coverage', '.cache']:
                    ignore.append(f)
                elif f in ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml']:
                    ignore.append(f)
            return ignore
        
        shutil.copytree(project_path, output_path, ignore=ignore_patterns, dirs_exist_ok=True)
        print(f"✅ Copied project to {output_path} (excluded node_modules, build, etc.)")
        
        index_html = output_path / 'frontend' / 'public' / 'index.html'
        if not index_html.exists():
            index_html = output_path / 'public' / 'index.html'
        
        if index_html.exists():
            success = self.inject_to_file(str(index_html))
            if success:
                print(f"✅ Injected HTML popup perturbation to {index_html.relative_to(output_path)}")
                print(f"\n{'='*80}")
                print(f"⚠️  IMPORTANT: Dependencies need to be reinstalled")
                print(f"{'='*80}")
                print(f"Run:")
                print(f"  cd {output_path}/backend && npm install")
                print(f"  cd {output_path}/frontend && npm install")
                print(f"  cd {output_path}/backend && node server.js &")
                print(f"  cd {output_path}/frontend && npm start")
                print(f"{'='*80}\n")
                return True
            else:
                return False
        else:
            print(f"❌ No index.html found")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='HTML Popup Perturbation Injection Script')
    parser.add_argument('--input', '-i', required=True, help='Input file or directory path')
    parser.add_argument('--output', '-o', default=None, help='Output path')
    parser.add_argument('--probability', '-p', type=float, default=0.35, help='Popup probability (0.0-1.0)')
    parser.add_argument('--react',default=True, action='store_true', help='React application')
    
    args = parser.parse_args()
    
    if not 0.0 <= args.probability <= 1.0:
        print("❌ Error: Probability must be between 0.0 and 1.0")
        return
    
    injector = HTMLPopupPerturbationInjector(probability=args.probability)
    
    print(f"\n{'='*80}")
    print(f"🎭 HTML Popup Perturbation Injector")
    print(f"{'='*80}")
    print(f"📊 Popup probability: {args.probability * 100:.1f}%")
    print(f"✨ Using custom HTML dialogs (Agent can see and click!)")
    print(f"{'='*80}\n")
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {args.input}")
        return
    
    if args.react:
        success = injector.inject_to_react_app(args.input, args.output)
        if success:
            print(f"\n✅ React app HTML popup perturbation injection completed!")
        else:
            print(f"\n❌ Injection failed!")
    elif input_path.is_file():
        success = injector.inject_to_file(args.input, args.output)
        if success:
            print(f"\n✅ File HTML popup perturbation injection completed!")
    else:
        print(f"❌ Directory mode not implemented yet, use --react for React apps")


if __name__ == "__main__":
    main()
