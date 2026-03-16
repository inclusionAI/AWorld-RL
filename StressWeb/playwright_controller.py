

import asyncio
import base64
import json
import logging
import random
import time
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from PIL import Image
from io import BytesIO

from web_actions import ACTION_SPACE, VIEWPORT_WIDTH, VIEWPORT_HEIGHT

logger = logging.getLogger("web_env.playwright_controller")


class PlaywrightController:
    
    
    def __init__(
        self,
        headless: bool = False,
        viewport_width: int = VIEWPORT_WIDTH,
        viewport_height: int = VIEWPORT_HEIGHT,
        slow_mo: int = 0,
        record_video: bool = False,
        video_path: Optional[str] = None,
        dom_filter_selectors: Optional[List[str]] = None,
        dom_perturbation_modes: Optional[List[str]] = None,
        action_max_retries: int = 3,
        action_retry_min_delay: float = 1.0,
        action_retry_max_delay: float = 3.0,
    ):
        
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.slow_mo = slow_mo
        self.record_video = record_video
        self.video_path = video_path
        self.dom_filter_selectors = dom_filter_selectors or []  
        self.dom_perturbation_modes = dom_perturbation_modes or []  
        
        
        self.action_max_retries = action_max_retries  
        self.action_retry_min_delay = action_retry_min_delay  
        self.action_retry_max_delay = action_retry_max_delay  
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        self._is_initialized = False
        
        
        self.dialog_logs = []  
        self.auto_accept_dialogs = True  
        self.dialog_acceptance_rate = 1.0  
            
    async def initialize(self):
        
        if self._is_initialized:
            logger.warning("Controller already initialized")
            return
            
        logger.info("Initializing Playwright Controller...")
        self.playwright = await async_playwright().start()
        
        launch_options = {
            "headless": self.headless,
            "slow_mo": self.slow_mo,
        }
        
        self.browser = await self.playwright.chromium.launch(**launch_options)
        
        context_options = {
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height
            }
        }
        
        if self.record_video and self.video_path:
            context_options["record_video_dir"] = self.video_path
            
        self.context = await self.browser.new_context(**context_options)
        self.page = await self.context.new_page()
        
        self._is_initialized = True
        logger.info("Playwright Controller initialized successfully")
        
        
        if self.dom_filter_selectors:
            logger.info(f"🔍 DOM filter active in controller: {self.dom_filter_selectors}")
            selectors_json = json.dumps(self.dom_filter_selectors)
            await self.page.add_init_script(f"""
                const filterSelectors = {selectors_json};

                function removeFilteredElements() {{{{
                    for (const selector of filterSelectors) {{{{
                        try {{{{
                            const elements = document.querySelectorAll(selector);
                            elements.forEach(el => el.remove());
                        }}}}  catch (e) {{{{
                        }}}}
                    }}}}
                }}}}

                removeFilteredElements();

                const observer = new MutationObserver(() => {{{{
                    removeFilteredElements();
                }}}} );

                observer.observe(document.body, {{{{
                    childList: true,
                    subtree: true,
                    attributes: false
                }}}} );
            """)
            logger.info(f"✅ DOM filter injected: {self.dom_filter_selectors}")
        
        if self.dom_perturbation_modes:
            logger.info(f"🌀 DOM perturbations enabled: {self.dom_perturbation_modes}")
        
    async def clear_storage(self):
        
        if not self._is_initialized or not self.page:
            logger.warning("Cannot clear storage: controller not initialized")
            return False
            
        try:
            logger.info("Clearing browser storage...")
            
            
            await self.page.evaluate("""
                () => {
                    localStorage.clear();
                    sessionStorage.clear();
                }
            """)
            
            
            await self.context.clear_cookies()
            
            logger.info("✅ Browser storage cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear storage: {e}")
            return False
    
    async def close(self):
        
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._is_initialized = False
        logger.info("Playwright Controller closed")
        
    def _is_retriable_error(self, error_str: str) -> bool:
        
        retriable_keywords = [
            "502",
            "503",
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "connection aborted",
            "ERR_CONNECTION_REFUSED",
            "ERR_CONNECTION_RESET",
            "ERR_CONNECTION_ABORTED",
            "ERR_NETWORK",
            "ERR_INTERNET_DISCONNECTED",
            "net::ERR",
        ]
        error_lower = error_str.lower()
        return any(keyword.lower() in error_lower for keyword in retriable_keywords)
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        
        if not self._is_initialized:
            return {"success": False, "error": "Controller not initialized"}

        action_type = action.get("action_type")
        parameters = action.get("parameters", {})

        logger.info(f"Executing action: {action_type}")

        
        for attempt in range(self.action_max_retries + 1):
            try:
                result = await self._execute_with_playwright(action_type, parameters)
                
                
                if result.get("success"):
                    return result
                
                error = result.get("error", "")
                if self._is_retriable_error(error) and attempt < self.action_max_retries:
                    
                    delay = random.uniform(self.action_retry_min_delay, self.action_retry_max_delay)
                    logger.warning(
                        f"⚠️ Retriable error in action {action_type}: {error}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.action_max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                return result
                
            except Exception as e:
                error_str = str(e)
                if self._is_retriable_error(error_str) and attempt < self.action_max_retries:
                    
                    delay = random.uniform(self.action_retry_min_delay, self.action_retry_max_delay)
                    logger.warning(
                        f"⚠️ Retriable exception in action {action_type}: {error_str}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.action_max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                logger.error(f"❌ Error executing action {action_type}: {error_str}")
                return {"success": False, "error": error_str}
        
        
        return {
            "success": False,
            "error": f"Max retries ({self.action_max_retries}) exceeded for action {action_type}"
        }
            
    async def _execute_with_playwright(self, action_type: str, parameters: Dict) -> Dict:
        
        if action_type == "NAVIGATE":
            return await self._navigate(parameters)
        elif action_type == "CLICK":
            return await self._click_playwright(parameters)
        elif action_type == "TYPE":
            return await self._type_playwright(parameters)
        elif action_type == "PRESS":
            return await self._press_playwright(parameters)
        elif action_type == "HOTKEY":
            return await self._hotkey_playwright(parameters)
        elif action_type == "SCROLL":
            return await self._scroll(parameters)
        elif action_type == "HOVER":
            return await self._hover(parameters)
        elif action_type == "SELECT":
            return await self._select(parameters)
        elif action_type == "FILL":
            return await self._fill(parameters)
        elif action_type == "CHECK":
            return await self._check(parameters)
        elif action_type == "UNCHECK":
            return await self._uncheck(parameters)
        elif action_type == "WAIT":
            return await self._wait(parameters)
        elif action_type == "GO_BACK":
            await self.page.go_back()
            return {"success": True}
        elif action_type == "GO_FORWARD":
            await self.page.go_forward()
            return {"success": True}
        elif action_type == "RELOAD":
            await self.page.reload()
            return {"success": True}
        elif action_type in ["DONE", "FAIL"]:
            return {"success": True, "message": action_type}
        else:
            return {"success": False, "error": f"Unknown action type: {action_type}"}
        
    
    
    async def _ensure_coordinates_visible(self, x: int, y: int) -> Dict[str, int]:
        
        viewport_info = await self.get_viewport_info()
        vp_width = viewport_info["width"]
        vp_height = viewport_info["height"]
        scroll_x = viewport_info["scrollX"]
        scroll_y = viewport_info["scrollY"]
        
        
        abs_x = x + scroll_x
        abs_y = y + scroll_y
        
        
        if 0 <= x <= vp_width and 0 <= y <= vp_height:
            return {"scrollX": scroll_x, "scrollY": scroll_y}
        
        
        target_scroll_x = max(0, abs_x - vp_width // 2)
        target_scroll_y = max(0, abs_y - vp_height // 2)
        
        logger.info(f"📜 Scrolling to make ({x}, {y}) visible: scroll to ({target_scroll_x}, {target_scroll_y})")
        await self.page.evaluate(f"window.scrollTo({target_scroll_x}, {target_scroll_y})")
        await asyncio.sleep(0.3)  
        
        return {"scrollX": target_scroll_x, "scrollY": target_scroll_y}

    async def _selector_is_filtered(self, selector: str) -> bool:
        
        if not self.dom_filter_selectors:
            return False

        try:
            blocked = await self.page.locator(selector).evaluate_all(
                """
                (nodes, filters) => {
                    if (!Array.isArray(nodes) || nodes.length === 0) return false;
                    return nodes.some(n => filters.some(f => n.matches?.(f) || n.closest?.(f)));
                }
                """,
                self.dom_filter_selectors,
            )
            if blocked:
                logger.info(f"🚫 Selector blocked by DOM filter: {selector}")
            return blocked
        except Exception:
            
            return False
    
    async def _navigate(self, params: Dict) -> Dict:
        
        url = params.get("url")
        if not url:
            return {"success": False, "error": "URL is required"}
        try:
            await self.page.goto(url, wait_until="load", timeout=30000)
            await asyncio.sleep(0.5)
            return {"success": True, "url": url}
        except Exception as e:
            error_msg = str(e)
            
            return {"success": False, "error": f"Navigation failed: {error_msg}"}
        
    async def _click_playwright(self, params: Dict) -> Dict:
        
        selector = params.get("selector")
        x = params.get("x")
        y = params.get("y")
        button = params.get("button", "left")
        click_count = params.get("click_count", 1)

        
        if selector:
            try:
                
                if await self._selector_is_filtered(selector):
                    raise Exception(f"Selector matches filtered element: {selector}")

                
                await self.page.click(selector, button=button, click_count=click_count, timeout=10000)
                logger.info(f"✓ Clicked using selector: {selector}")
                return {"success": True, "selector": selector, "method": "playwright"}
            except Exception as selector_error:
                logger.warning(f"Selector click failed: {selector_error}")
                
                if x is not None and y is not None:
                    logger.info(f"Falling back to coordinates: ({x}, {y})")
                    try:
                        
                        viewport_info = await self.get_viewport_info()
                        scroll_x = viewport_info["scrollX"]
                        scroll_y = viewport_info["scrollY"]
                        
                        
                        page_x = x + scroll_x
                        page_y = y + scroll_y
                        
                        if scroll_x != 0 or scroll_y != 0:
                            logger.warning(f"⚠️  Page scroll: ({scroll_x}, {scroll_y})")
                        
                        logger.info(f"📍 Fallback coordinate click: viewport ({x}, {y}) + scroll ({scroll_x}, {scroll_y}) = page ({page_x}, {page_y})")
                        
                        await self.page.mouse.click(page_x, page_y, button=button, click_count=click_count)
                        logger.info(f"✓ Clicked using coordinates: ({x}, {y})")
                        return {"success": True, "x": x, "y": y, "method": "playwright", "scroll": {"x": scroll_x, "y": scroll_y}}
                    except Exception as coord_error:
                        logger.error(f"Coordinate click also failed: {coord_error}")
                        raise coord_error
                raise selector_error

        
        elif x is not None and y is not None:
            try:
                
                viewport_info = await self.get_viewport_info()
                scroll_x = viewport_info["scrollX"]
                scroll_y = viewport_info["scrollY"]
                
                
                
                page_x = x + scroll_x
                page_y = y + scroll_y
                
                
                vp_width = viewport_info["width"]
                vp_height = viewport_info["height"]
                if x < 0 or y < 0 or x > vp_width or y > vp_height:
                    logger.warning(f"⚠️  Viewport coordinates ({x}, {y}) outside viewport ({vp_width}x{vp_height})")
                
                logger.info(f"📍 Coordinate click: viewport ({x}, {y}) + scroll ({scroll_x}, {scroll_y}) = page ({page_x}, {page_y})")
                
                
                await self.page.mouse.click(page_x, page_y, button=button, click_count=click_count)
                logger.info(f"✓ Clicked using page coordinates: ({page_x}, {page_y})")
                return {"success": True, "x": x, "y": y, "method": "playwright", "scroll": {"x": scroll_x, "y": scroll_y}}
            except Exception as e:
                logger.error(f"Playwright coordinate click failed: {e}")
                raise
        else:
            return {"success": False, "error": "Either selector or (x, y) is required"}
            
    async def _type_playwright(self, params: Dict) -> Dict:
        
        text = params.get("text")
        selector = params.get("selector")
        delay = params.get("delay", 0)
        
        if not text:
            return {"success": False, "error": "Text is required"}
        
        if selector:
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            await self.page.click(selector)
            await asyncio.sleep(0.1)
            await self.page.type(selector, text, delay=delay)
        else:
            await self.page.keyboard.type(text, delay=delay)
        return {"success": True, "text": text, "method": "playwright"}
            
    async def _press_playwright(self, params: Dict) -> Dict:
        
        key = params.get("key")
        if not key:
            return {"success": False, "error": "Key is required"}
        await self.page.keyboard.press(key)
        return {"success": True, "key": key, "method": "playwright"}
        
    async def _hotkey_playwright(self, params: Dict) -> Dict:
        
        keys = params.get("keys")
        if not keys:
            return {"success": False, "error": "Keys combination is required"}
        
        key_list = keys.split("+")
        for key in key_list[:-1]:
            await self.page.keyboard.down(key)
        await self.page.keyboard.press(key_list[-1])
        for key in reversed(key_list[:-1]):
            await self.page.keyboard.up(key)
            
        return {"success": True, "keys": keys, "method": "playwright"}
        
    async def _fill(self, params: Dict) -> Dict:
        
        selector = params.get("selector")
        text = params.get("text")
        
        if not selector or text is None:
            return {"success": False, "error": "Selector and text are required"}
        
        try:
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            
            is_readonly = await self.page.eval_on_selector(
                selector, "el => el.hasAttribute('readonly') || el.readOnly"
            )
            
            if is_readonly:
                logger.info("Detected readonly input, using JavaScript")
                await self.page.eval_on_selector(
                    selector,
                    """(el, text) => {
                        el.value = text;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }""",
                    text
                )
            else:
                await self.page.click(selector)
                await asyncio.sleep(0.1)
                await self.page.fill(selector, text)
            
            return {"success": True, "selector": selector, "text": text, "method": "playwright"}
        except Exception as e:
            logger.error(f"Fill failed: {e}")
            return {"success": False, "error": str(e)}
            
    
    
    async def _scroll(self, params: Dict) -> Dict:
        
        direction = params.get("direction")
        amount = params.get("amount", 100)
        
        if not direction:
            return {"success": False, "error": "Direction is required"}
            
        delta_x, delta_y = 0, 0
        if direction == "down":
            delta_y = amount
        elif direction == "up":
            delta_y = -amount
        elif direction == "right":
            delta_x = amount
        elif direction == "left":
            delta_x = -amount
            
        await self.page.evaluate(f"window.scrollBy({delta_x}, {delta_y})")
        return {"success": True, "direction": direction, "amount": amount}
        
    async def _hover(self, params: Dict) -> Dict:
        
        selector = params.get("selector")
        x = params.get("x")
        y = params.get("y")
        
        if selector:
            await self.page.hover(selector)
            return {"success": True, "selector": selector}
        elif x is not None and y is not None:
            await self.page.mouse.move(x, y)
            return {"success": True, "x": x, "y": y}
        else:
            return {"success": False, "error": "Either selector or (x, y) is required"}
            
    async def _select(self, params: Dict) -> Dict:
        
        selector = params.get("selector")
        value = params.get("value")
        label = params.get("label")
        index = params.get("index")
        
        if not selector:
            return {"success": False, "error": "Selector is required"}
            
        if value is not None:
            await self.page.select_option(selector, value=value)
        elif label is not None:
            await self.page.select_option(selector, label=label)
        elif index is not None:
            await self.page.select_option(selector, index=index)
        else:
            return {"success": False, "error": "Either value, label, or index is required"}
            
        return {"success": True, "selector": selector}
        
    async def _check(self, params: Dict) -> Dict:
        
        selector = params.get("selector")
        if not selector:
            return {"success": False, "error": "Selector is required"}
        await self.page.check(selector)
        return {"success": True, "selector": selector}
        
    async def _uncheck(self, params: Dict) -> Dict:
        
        selector = params.get("selector")
        if not selector:
            return {"success": False, "error": "Selector is required"}
        await self.page.uncheck(selector)
        return {"success": True, "selector": selector}
        
    async def _wait(self, params: Dict) -> Dict:
        
        wait_time = params.get("time")
        selector = params.get("selector")
        state = params.get("state", "visible")
        
        if wait_time is not None:
            await asyncio.sleep(wait_time)
            return {"success": True, "waited": wait_time}
        elif selector:
            await self.page.wait_for_selector(selector, state=state)
            return {"success": True, "selector": selector, "state": state}
        else:
            return {"success": False, "error": "Either time or selector is required"}
    
    
    
    async def get_screenshot(self, full_page: bool = False) -> str:
        
        screenshot_bytes = await self.page.screenshot(full_page=full_page)
        return base64.b64encode(screenshot_bytes).decode('utf-8')
        
    async def get_viewport_info(self) -> Dict[str, Any]:
        
        viewport_info = await self.page.evaluate("""
            () => ({
                width: window.innerWidth,
                height: window.innerHeight,
                devicePixelRatio: window.devicePixelRatio,
                scrollX: window.scrollX,
                scrollY: window.scrollY,
                documentWidth: document.documentElement.scrollWidth,
                documentHeight: document.documentElement.scrollHeight
            })
        """)
        return viewport_info
    
    async def get_page_info(self) -> Dict[str, Any]:
        
        viewport_info = await self.get_viewport_info()
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "viewport": {
                "width": viewport_info["width"],
                "height": viewport_info["height"],
                "scrollX": viewport_info["scrollX"],
                "scrollY": viewport_info["scrollY"],
                "dpr": viewport_info["devicePixelRatio"]
            }
        }
        
    async def get_dom_snapshot(self) -> str:
        
        return await self.page.content()

    async def _build_dom_observation(self, apply_filter: bool, apply_perturb: bool) -> Dict[str, str]:
        
        selectors_json = json.dumps(self.dom_filter_selectors or [])
        modes_json = json.dumps(self.dom_perturbation_modes or [])
        result = await self.page.evaluate(f"""
            () => {{ 
                const filterSelectors = {selectors_json};
                const perturbModes = {modes_json};
                const doFilter = {str(apply_filter).lower()};
                const doPerturb = {str(apply_perturb).lower()} && Array.isArray(perturbModes) && perturbModes.length > 0;

                const rawHtml = document.documentElement.outerHTML;
                const cloneDoc = document.cloneNode(true);

                // Auto-remove injected perturbation code to prevent info leakage
                function removePerturbationInjections(root) {{ 
                    // Auto-remove all types of perturbation injection code to prevent info leakage
                    
                    // 1. Remove comment nodes containing perturbation markers
                    const walker = document.createTreeWalker(
                        root,
                        NodeFilter.SHOW_COMMENT,
                        null,
                        false
                    );
                    const commentsToRemove = [];
                    while (walker.nextNode()) {{ 
                        const comment = walker.currentNode.textContent || '';
                        if (comment.includes('PERTURBATION') ||
                            comment.includes('Layout Chaos') ||
                            comment.includes('Semantic Conflict') ||
                            comment.includes('Failure Perturbation')) {{ 
                            commentsToRemove.push(walker.currentNode);
                        }} 
                    }} 
                    commentsToRemove.forEach(c => c.remove());

                    // 2. Remove style tags injected by perturbation (all types)
                    const styles = root.querySelectorAll('style');
                    styles.forEach(style => {{ 
                        const id = style.id || '';
                        const content = style.textContent || '';
                        // HTML popup perturbation styles
                        if (content.includes('perturbation-dialog') || 
                            content.includes('perturbation-warning') ||
                            content.includes('PERTURBATION') ||
                            content.includes('.perturbation-') ||
                            // Layout Chaos styles
                            id === 'layout-chaos-base-styles' ||
                            id.includes('chaos') ||
                            content.includes('Layout Chaos Perturbation')) {{ 
                            style.remove();
                        }} 
                    }} );

                    // 3. Remove script tags injected by perturbation (all types)
                    const scripts = root.querySelectorAll('script');
                    scripts.forEach(script => {{ 
                        const content = script.textContent || '';
                        // HTML popup perturbation
                        if (content.includes('POPUP_PROBABILITY') ||
                            content.includes('POPUP_CONFIGS') ||
                            content.includes('perturbationSlideIn') ||
                            content.includes('showConfirmationPopup') ||
                            content.includes('getPerturbationLogs') ||
                            content.includes('Perturbation Injected') ||
                            content.includes('clickInterceptor') ||
                            content.includes('PERTURBATION') ||
                            // Failure perturbation
                            content.includes('FAILURE_PROBABILITY') ||
                            content.includes('failurePerturbations') ||
                            content.includes('logFailureEvent') ||
                            // Semantic Conflict perturbation
                            content.includes('semanticConflictLogs') ||
                            content.includes('[Semantic Conflict]') ||
                            content.includes('Semantic Conflict') ||
                            // Layout Chaos perturbation
                            content.includes('Layout Chaos') ||
                            content.includes('layoutChaosPerturbation') ||
                            content.includes('applyLayoutChaos')) {{ 
                            script.remove();
                        }} 
                    }} );

                    // 4. Remove perturbation popup overlays (if any)
                    const overlays = root.querySelectorAll('[data-perturbation-dialog]');
                    overlays.forEach(el => el.remove());

                    // 5. Remove other perturbation-related elements
                    const perturbElements = root.querySelectorAll('.perturbation-dialog-overlay, .perturbation-dialog-box');
                    perturbElements.forEach(el => el.remove());
                }} 

                function removeFilteredElements(root) {{ 
                    for (const selector of filterSelectors) {{ 
                        try {{ 
                            const elements = root.querySelectorAll(selector);
                            elements.forEach(el => el.remove());
                        }}  catch (e) {{ 
                            // ignore invalid selectors
                        }} 
                    }} 
                }} 

                function wrapWithNestedDivs(el, layers = 2) {{ 
                    let node = el;
                    for (let i = 0; i < layers; i++) {{ 
                        const wrapper = el.ownerDocument.createElement('div');
                        wrapper.className = `dom-perturb-wrapper depth-${{ i}}  rand-${{ Math.random().toString(36).slice(2, 6)}} `;
                        wrapper.setAttribute('data-perturb', 'nested-wrapper');
                        wrapper.setAttribute('style', 'all: unset; display: contents;'); // No layout impact
                        node.replaceWith(wrapper);
                        wrapper.appendChild(node);
                        node = wrapper;
                    }} 
                }} 

                function addNestedWrappers(root) {{ 
                    const candidates = Array.from(root.querySelectorAll('p, div, span, h1, h2, h3, h4, h5, h6, button, label, a'))
                        .filter(el => (el.textContent || '').trim().length > 8);
                    // Increase to 15 elements for more interference
                    candidates.slice(0, 15).forEach(el => {{ 
                        const layers = 3 + Math.floor(Math.random() * 3); // 3-5 layers of nesting
                        wrapWithNestedDivs(el, layers);
                    }} );
                }} 

                function addTextDecoys(root) {{ 
                    const targets = Array.from(root.querySelectorAll('button, a, input[type="submit"], input[type="button"], label'))
                        .filter(el => (el.textContent || '').trim().length > 2);
                    // Increase to 20 decoy elements
                    targets.slice(0, 20).forEach((el, idx) => {{ 
                        const decoy = el.cloneNode(true);
                        decoy.setAttribute('data-perturb', 'decoy-text');
                        decoy.classList.add('dom-perturb-decoy');
                        decoy.setAttribute('aria-hidden', 'true');
                        // Deliberately create confusing IDs and classes
                        decoy.id = el.id ? `${{ el.id}} -fake-${{ idx}} ` : `decoy-${{ idx}} -${{ Math.random().toString(36).slice(2, 6)}} `;
                        if (decoy.hasAttribute('name')) decoy.setAttribute('name', `fake-${{ decoy.getAttribute('name')}} `);
                        decoy.setAttribute('disabled', 'true'); // Mark as disabled, but agent can't see the effect
                        decoy.setAttribute('onclick', 'return false');
                        // Random insertion: before or after
                        if (el.parentNode) {{ 
                            if (Math.random() > 0.5) {{ 
                                el.parentNode.insertBefore(decoy, el);
                            }}  else {{ 
                                el.parentNode.insertBefore(decoy, el.nextSibling);
                            }} 
                        }} 
                    }} );
                }} 

                function addExecutableBaits(root) {{ 
                    const container = root.body || root;
                    // Add multiple decoy buttons and links
                    for (let i = 0; i < 5; i++) {{ 
                        const bait = root.createElement('button');
                        bait.textContent = ['Run Diagnostics', 'Test Connection', 'Verify System', 'Check Status', 'Refresh Data'][i];
                        bait.className = 'dom-perturb-bait';
                        bait.setAttribute('data-perturb', 'executable');
                        bait.setAttribute('onclick', "console.log('noop bait')");
                        bait.setAttribute('type', 'button');
                        // Random insertion at different positions
                        const insertPos = Math.floor(Math.random() * container.children.length);
                        if (container.children[insertPos]) {{ 
                            container.insertBefore(bait, container.children[insertPos]);
                        }}  else {{ 
                            container.appendChild(bait);
                        }} 
                    }} 

                    // Add fake links
                    for (let i = 0; i < 3; i++) {{ 
                        const link = root.createElement('a');
                        link.href = '#';
                        link.textContent = ['Open Debug Panel', 'View Logs', 'System Settings'][i];
                        link.className = 'dom-perturb-bait-link';
                        link.setAttribute('data-perturb', 'executable');
                        link.setAttribute('onclick', "return false");
                        container.appendChild(link);
                    }} 
                }} 

                function breakStructure(root) {{ 
                    const body = root.body;
                    if (!body) return;
                    // Add multiple layers of meaningless structure wrapping
                    const wrapper1 = root.createElement('div');
                    wrapper1.className = 'dom-perturb-outer-wrapper';
                    wrapper1.setAttribute('data-perturb', 'structure');
                    const wrapper2 = root.createElement('section');
                    wrapper2.className = 'dom-perturb-section-break';
                    wrapper2.setAttribute('data-perturb', 'structure');
                    while (body.firstChild) {{ 
                        wrapper2.appendChild(body.firstChild);
                    }} 
                    wrapper1.appendChild(wrapper2);
                    body.appendChild(wrapper1);
                }} 

                function addAttributeNoise(root) {{ 
                    // Add noise attributes to random elements
                    const allElements = Array.from(root.querySelectorAll('*')).slice(0, 50);
                    allElements.forEach(el => {{ 
                        if (Math.random() > 0.5) {{ 
                            el.setAttribute('data-noise-id', Math.random().toString(36).slice(2, 10));
                            el.setAttribute('data-perturb-marker', 'noise');
                        }} 
                    }} );
                }} 

                function duplicateElements(root) {{ 
                    // Randomly duplicate some elements to create redundancy
                    const targets = Array.from(root.querySelectorAll('div, section, article')).slice(0, 10);
                    targets.forEach(el => {{ 
                        if (Math.random() > 0.6 && el.parentNode) {{ 
                            const dup = el.cloneNode(true);
                            dup.setAttribute('data-perturb', 'duplicate');
                            dup.classList.add('dom-perturb-duplicate');
                            el.parentNode.insertBefore(dup, el.nextSibling);
                        }} 
                    }} );
                }} 

                function addInvisibleText(root) {{ 
                    // Add invisible noise text nodes
                    const containers = Array.from(root.querySelectorAll('body, main, div')).slice(0, 10);
                    containers.forEach(container => {{ 
                        const noise = root.createTextNode(`<!-- perturb noise ${{ Math.random()}}  -->`);
                        if (container.firstChild) {{ 
                            container.insertBefore(noise, container.firstChild);
                        }} 
                    }} );
                }} 

                // ==================== Advanced perturbations: anti-crawling/privacy scenarios ====================

                function obfuscateText(root) {{ 
                    // Text obfuscation: insert Unicode zero-width or special chars
                    // Simulate anti-crawling text protection
                    const zeroWidthChars = ['\\u200B', '\\u200C', '\\u200D', '\\uFEFF'];
                    const textNodes = [];
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
                    while (walker.nextNode()) {{ 
                        const text = walker.currentNode.textContent.trim();
                        if (text.length > 5) textNodes.push(walker.currentNode);
                    }} 
                    // Randomly select 30% of text nodes for obfuscation
                    textNodes.slice(0, Math.ceil(textNodes.length * 0.3)).forEach(node => {{ 
                        const chars = node.textContent.split('');
                        const obfuscated = chars.map((c, i) => {{ 
                            if (i > 0 && i % 3 === 0 && Math.random() > 0.5) {{ 
                                return zeroWidthChars[Math.floor(Math.random() * zeroWidthChars.length)] + c;
                            }} 
                            return c;
                        }} ).join('');
                        node.textContent = obfuscated;
                    }} );
                }} 

                function redactSensitiveContent(root) {{ 
                    // Content masking: simulate sensitive info protection with placeholders
                    const sensitivePatterns = [
                        {{  regex: /\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{{ 2,}} \\b/g, replacement: '[EMAIL REDACTED]' }} ,
                        {{  regex: /\\b\\d{{ 3}} [-.]?\\d{{ 3}} [-.]?\\d{{ 4}} \\b/g, replacement: '[PHONE REDACTED]' }} ,
                        {{  regex: /\\$\\d+(\\.\\d{{ 2}} )?/g, replacement: '[PRICE HIDDEN]' }} ,
                        {{  regex: /\\b\\d{{ 4}} [-/]\\d{{ 2}} [-/]\\d{{ 2}} \\b/g, replacement: '[DATE REDACTED]' }} 
                    ];
                    const textNodes = [];
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
                    while (walker.nextNode()) {{ 
                        textNodes.push(walker.currentNode);
                    }} 
                    textNodes.forEach(node => {{ 
                        let text = node.textContent;
                        sensitivePatterns.forEach(p => {{ 
                            text = text.replace(p.regex, p.replacement);
                        }} );
                        node.textContent = text;
                    }} );
                }} 

                function encodeTextAsEntities(root) {{ 
                    // HTML entity encoding: convert text to entities to increase parsing difficulty
                    const textNodes = [];
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
                    while (walker.nextNode()) {{ 
                        const text = walker.currentNode.textContent.trim();
                        if (text.length > 3 && text.length < 50) textNodes.push(walker.currentNode);
                    }} 
                    // Randomly select 20% of short text for encoding
                    textNodes.slice(0, Math.ceil(textNodes.length * 0.2)).forEach(node => {{ 
                        const encoded = node.textContent.split('').map(c => {{ 
                            if (Math.random() > 0.7 && c.match(/[a-zA-Z]/)) {{ 
                                return `&#${{ c.charCodeAt(0)}} ;`;
                            }} 
                            return c;
                        }} ).join('');
                        node.textContent = encoded;
                    }} );
                }} 

                function splitTextWithSpans(root) {{ 
                    // Text splitting: split text into spans to simulate anti-crawling techniques
                    const targets = Array.from(root.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, a, button, label'))
                        .filter(el => {{ 
                            const text = el.textContent.trim();
                            return text.length > 5 && text.length < 100 && el.children.length === 0;
                        }} );
                    targets.slice(0, 15).forEach(el => {{ 
                        const text = el.textContent;
                        el.textContent = '';
                        // Split text into random-length chunks
                        let i = 0;
                        while (i < text.length) {{ 
                            const chunkSize = 1 + Math.floor(Math.random() * 3);
                            const chunk = text.slice(i, i + chunkSize);
                            const span = root.createElement('span');
                            span.textContent = chunk;
                            span.className = `txt-${{ Math.random().toString(36).slice(2, 6)}} `;
                            span.setAttribute('data-perturb', 'split-text');
                            el.appendChild(span);
                            i += chunkSize;
                        }} 
                    }} );
                }} 

                function addHoneypotElements(root) {{ 
                    // Honeypot elements: add hidden bait to detect crawlers/automation
                    const container = root.body || root;
                    const honeypots = [
                        {{  tag: 'input', attrs: {{  type: 'text', name: 'email_confirm', autocomplete: 'off', tabindex: '-1' }} , style: 'position:absolute;left:-9999px;' }} ,
                        {{  tag: 'a', attrs: {{  href: '/trap/bot-detector' }} , text: 'Click here for special offer', style: 'position:absolute;left:-9999px;opacity:0;' }} ,
                        {{  tag: 'div', attrs: {{  class: 'hidden-content-trap', 'aria-hidden': 'true' }} , text: 'This content is for verification purposes only', style: 'display:none;' }} ,
                        {{  tag: 'button', attrs: {{  type: 'button', class: 'verify-human' }} , text: 'Verify', style: 'visibility:hidden;position:absolute;' }} 
                    ];
                    honeypots.forEach((hp, idx) => {{ 
                        const el = root.createElement(hp.tag);
                        Object.entries(hp.attrs).forEach(([k, v]) => el.setAttribute(k, v));
                        if (hp.text) el.textContent = hp.text;
                        el.setAttribute('style', hp.style);
                        el.setAttribute('data-perturb', 'honeypot');
                        // Random insertion position
                        const insertPos = Math.floor(Math.random() * container.children.length);
                        if (container.children[insertPos]) {{ 
                            container.insertBefore(el, container.children[insertPos]);
                        }}  else {{ 
                            container.appendChild(el);
                        }} 
                    }} );
                }} 

                function scrambleAttributes(root) {{ 
                    // Attribute obfuscation: rename/obfuscate class and id
                    const elements = Array.from(root.querySelectorAll('[class], [id]')).slice(0, 30);
                    elements.forEach(el => {{ 
                        // Add random classes
                        if (el.className && typeof el.className === 'string') {{ 
                            const randomClasses = Array.from({{ length: 2}} , () => 
                                `_${{ Math.random().toString(36).slice(2, 8)}} `
                            ).join(' ');
                            el.className = el.className + ' ' + randomClasses;
                        }} 
                        // Add data-* noise attributes
                        el.setAttribute('data-v-' + Math.random().toString(36).slice(2, 6), '');
                        el.setAttribute('data-testid', 'el-' + Math.random().toString(36).slice(2, 8));
                    }} );
                }} 

                function injectCSSCounters(root) {{ 
                    // CSS content injection: use ::before/::after pseudo-elements
                    // This content is invisible in DOM but visually visible
                    const style = root.createElement('style');
                    style.setAttribute('data-perturb', 'css-counter');
                    style.textContent = `
                        [data-perturb-label]::before {{ 
                            content: attr(data-perturb-label) " ";
                        }} 
                        .perturb-price::after {{ 
                            content: " (estimated)";
                            font-size: 0.8em;
                            color: #999;
                        }} 
                    `;
                    if (root.head) root.head.appendChild(style);
                    
                    // Add data-perturb-label to some elements
                    const labels = ['Note:', 'Important:', 'Info:', 'Tip:'];
                    const targets = Array.from(root.querySelectorAll('p, li')).slice(0, 5);
                    targets.forEach((el, i) => {{ 
                        el.setAttribute('data-perturb-label', labels[i % labels.length]);
                    }} );
                }} 

                function addLoadingPlaceholders(root) {{ 
                    // Loading placeholders: simulate skeleton screens/loading states
                    const placeholders = [
                        '<div class="skeleton-loader" data-perturb="skeleton" style="background:linear-gradient(90deg,#f0f0f0 25%,#e0e0e0 50%,#f0f0f0 75%);height:20px;width:60%;margin:8px 0;border-radius:4px;"></div>',
                        '<div class="content-placeholder" data-perturb="skeleton" style="background:#f5f5f5;height:100px;width:100%;margin:10px 0;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#999;">Loading...</div>'
                    ];
                    const container = root.body || root;
                    placeholders.forEach(html => {{ 
                        const wrapper = root.createElement('div');
                        wrapper.innerHTML = html;
                        wrapper.setAttribute('data-perturb', 'loading-placeholder');
                        // Random insertion
                        const insertPos = Math.floor(Math.random() * Math.min(5, container.children.length));
                        if (container.children[insertPos]) {{ 
                            container.insertBefore(wrapper, container.children[insertPos]);
                        }} 
                    }} );
                }} 

                function obfuscateNumbers(root) {{ 
                    // Number obfuscation: replace numbers with images or special representations
                    const textNodes = [];
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
                    while (walker.nextNode()) {{ 
                        if (/\\d{{ 2,}} /.test(walker.currentNode.textContent)) {{ 
                            textNodes.push(walker.currentNode);
                        }} 
                    }} 
                    textNodes.slice(0, 10).forEach(node => {{ 
                        // Convert numbers to words or add noise
                        node.textContent = node.textContent.replace(/\\d+/g, match => {{ 
                            if (Math.random() > 0.5) {{ 
                                // Replace with Unicode digits
                                const unicodeDigits = ['𝟎', '𝟏', '𝟐', '𝟑', '𝟒', '𝟓', '𝟔', '𝟕', '𝟖', '𝟗'];
                                return match.split('').map(d => unicodeDigits[parseInt(d)] || d).join('');
                            }} 
                            return match;
                        }} );
                    }} );
                }} 

                function applyPerturbations(root) {{ 
                    if (!doPerturb) return;
                    const modeSet = new Set(perturbModes);
                    
                    // Basic perturbations
                    if (modeSet.has('nested_wrappers')) addNestedWrappers(root);
                    if (modeSet.has('false_text') || modeSet.has('decoy_text')) addTextDecoys(root);
                    if (modeSet.has('extra_actions') || modeSet.has('executable')) addExecutableBaits(root);
                    if (modeSet.has('structure_break')) breakStructure(root);
                    if (modeSet.has('attribute_noise')) addAttributeNoise(root);
                    if (modeSet.has('duplicate_elements')) duplicateElements(root);
                    if (modeSet.has('invisible_text')) addInvisibleText(root);
                    
                    // Advanced perturbations: anti-crawling/privacy simulation
                    if (modeSet.has('obfuscate_text')) obfuscateText(root);
                    if (modeSet.has('redact_content')) redactSensitiveContent(root);
                    if (modeSet.has('entity_encode')) encodeTextAsEntities(root);
                    if (modeSet.has('split_text')) splitTextWithSpans(root);
                    if (modeSet.has('honeypot')) addHoneypotElements(root);
                    if (modeSet.has('scramble_attrs')) scrambleAttributes(root);
                    if (modeSet.has('css_content')) injectCSSCounters(root);
                    if (modeSet.has('loading_skeleton')) addLoadingPlaceholders(root);
                    if (modeSet.has('obfuscate_numbers')) obfuscateNumbers(root);
                    
                    // 'all' mode: apply all perturbations (basic, anti-crawl, privacy), max disruption, exclude nested_wrappers
                    if (modeSet.has('all')) {{ 
                        // Basic perturbations (exclude nested_wrappers)
                        addTextDecoys(root);
                        addExecutableBaits(root);
                        breakStructure(root);
                        addAttributeNoise(root);
                        duplicateElements(root);
                        addInvisibleText(root);
                        
                        // Anti-crawling perturbations
                        obfuscateText(root);
                        splitTextWithSpans(root);
                        addHoneypotElements(root);
                        scrambleAttributes(root);
                        obfuscateNumbers(root);
                        
                        // Privacy protection perturbations
                        redactSensitiveContent(root);
                        encodeTextAsEntities(root);
                    }} 
                    
                    // 'anti_crawl' mode: apply all anti-crawling perturbations
                    if (modeSet.has('anti_crawl')) {{ 
                        obfuscateText(root);
                        splitTextWithSpans(root);
                        addHoneypotElements(root);
                        scrambleAttributes(root);
                        obfuscateNumbers(root);
                    }} 
                    
                    // 'privacy' mode: apply privacy protection perturbations
                    if (modeSet.has('privacy')) {{ 
                        redactSensitiveContent(root);
                        encodeTextAsEntities(root);
                    }} 
                }} 

                // First remove injected perturbation code to prevent info leakage
                removePerturbationInjections(cloneDoc);

                if (doFilter) {{ 
                    removeFilteredElements(cloneDoc);
                }} 

                applyPerturbations(cloneDoc);

                return {{  raw: rawHtml, dom: cloneDoc.documentElement.outerHTML }} ;
            }} 
        """)
        return result
    
    async def get_filtered_dom_snapshot(self) -> str:
        
        snapshots = await self._build_dom_observation(apply_filter=bool(self.dom_filter_selectors), apply_perturb=bool(self.dom_perturbation_modes))
        return snapshots["dom"]
    
    async def get_simplified_dom(self) -> Dict[str, Any]:
        
        simplified = await self.page.evaluate("""
            () => {
                const result = [];
                let elementId = 0;
                
                function traverse(node, depth = 0, parentPath = '') {
                    if (node.nodeType === Node.TEXT_NODE) {
                        const text = node.textContent.trim();
                        if (text && text.length > 0) {
                            result.push({
                                type: 'text',
                                content: text.slice(0, 100),  // Limit length
                                depth: depth
                            });
                        }
                    } else if (node.nodeType === Node.ELEMENT_NODE) {
                        const tag = node.tagName.toLowerCase();
                        const interactive = ['button', 'a', 'input', 'textarea', 'select', 'form'];
                        const semantic = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'nav', 'header', 'footer', 'main'];
                        
                        // Record interactive and semantic elements
                        if (interactive.includes(tag) || semantic.includes(tag)) {
                            const info = {
                                type: 'element',
                                tag: tag,
                                depth: depth,
                                id: elementId++
                            };
                            
                            if (node.id) info.domId = node.id;
                            if (node.className && typeof node.className === 'string') {
                                info.class = node.className.split(' ').slice(0, 3).join(' ');
                            }
                            if (node.name) info.name = node.name;
                            if (node.type) info.inputType = node.type;
                            if (node.placeholder) info.placeholder = node.placeholder.slice(0, 50);
                            if (node.value) info.value = node.value.slice(0, 50);
                            if (node.href) info.href = node.href.slice(0, 100);
                            
                            // Get text content (excluding child elements)
                            let text = '';
                            for (const child of node.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE) {
                                    text += child.textContent.trim() + ' ';
                                }
                            }
                            if (text.trim()) {
                                info.text = text.trim().slice(0, 100);
                            }
                            
                            result.push(info);
                        }
                        
                        // Recursively traverse child nodes
                        Array.from(node.childNodes).forEach(child => {
                            traverse(child, depth + 1, parentPath + '>' + tag);
                        });
                    }
                }
                
                traverse(document.body);
                return result;
            }
        """)
        
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "elements": simplified
        }
    
    async def get_compact_dom(self) -> str:
        
        simplified = await self.get_simplified_dom()
        
        lines = []
        lines.append(f"URL: {simplified['url']}")
        lines.append(f"Title: {simplified['title']}")
        lines.append("\nDOM Structure:")
        lines.append("=" * 60)
        
        for item in simplified['elements']:
            indent = "  " * item['depth']
            
            if item['type'] == 'text':
                
                text = item['content'][:80]
                lines.append(f"{indent}📝 {text}")
            else:
                
                tag = item['tag']
                parts = [f"<{tag}"]
                
                if 'domId' in item:
                    parts.append(f"id='{item['domId']}'")
                if 'class' in item:
                    parts.append(f"class='{item['class']}'")
                if 'text' in item:
                    parts.append(f"text='{item['text'][:50]}'")
                if 'placeholder' in item:
                    parts.append(f"placeholder='{item['placeholder']}'")
                if 'href' in item:
                    parts.append(f"href='{item['href'][:50]}'")
                
                line = f"{indent}🔘 {' '.join(parts)}>"  
                lines.append(line[:150])  
        
        return "\n".join(lines)
        
    async def get_observation(
        self,
        include_screenshot: bool = True,
        include_dom: bool = False,
        dom_format: str = "full"
    ) -> Dict[str, Any]:
        
        observation = {
            "page_info": await self.get_page_info(),
            "timestamp": time.time()
        }

        if include_screenshot:
            observation["screenshot"] = await self.get_screenshot()

        if include_dom:
            if dom_format == "full":
                
                snapshots = await self._build_dom_observation(
                    apply_filter=bool(self.dom_filter_selectors),
                    apply_perturb=bool(self.dom_perturbation_modes),
                )
                observation["dom"] = snapshots["dom"]
                observation["raw_dom"] = snapshots.get("raw")
            elif dom_format == "simplified":
                observation["dom"] = await self.get_simplified_dom()
            elif dom_format == "compact":
                observation["dom"] = await self.get_compact_dom()
            else:
                raise ValueError(f"Unknown dom_format: {dom_format}")

        return observation