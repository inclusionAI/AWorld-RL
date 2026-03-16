

import asyncio
import json
import logging
import sys
import os
import base64
import datetime
import platform
from typing import Dict, Any
from pathlib import Path


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from playwright_controller import PlaywrightController
from test_api import call_openai
from web_actions import get_action_description


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NoteWebsiteAgent:
    

    def __init__(
        self,
        controller: PlaywrightController,
        model: str = "gemini-2.5-flash",
        max_steps: int = 30,
        timeout: float = 1800.0,  
        result_dir: str = None,
        use_dom: bool = True,  
        dom_filter_selectors: list = None,  
        dom_perturbation_modes: list = None  
    ):
        
        self.controller = controller
        self.model = model
        self.max_steps = max_steps
        self.timeout = timeout
        self.use_dom = use_dom  
        self.dom_filter_selectors = dom_filter_selectors or []  
        self.dom_perturbation_modes = dom_perturbation_modes or []  

        
        if self.dom_filter_selectors:
            self.controller.dom_filter_selectors = self.dom_filter_selectors
            logger.info(f"🔍 DOM Filter enabled: {self.dom_filter_selectors}")

        if self.dom_perturbation_modes:
            self.controller.dom_perturbation_modes = self.dom_perturbation_modes
            logger.info(f"🌀 DOM Perturbations enabled: {self.dom_perturbation_modes}")

        self.current_step = 0
        self.action_history = []
        self.observation_history = []

        
        if result_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(current_dir, "test_results", f"run_{timestamp}")
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.result_dir}")

    async def _save_observation(self, observation: Dict[str, Any], step: int, prefix: str = "step"):
        
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")

        
        if 'screenshot' in observation:
            screenshot_file = f"{prefix}_{step}_{timestamp}.png"
            screenshot_path = os.path.join(self.result_dir, screenshot_file)

            
            screenshot_data = base64.b64decode(observation['screenshot'])
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot_data)
            logger.info(f"  📸 Saved screenshot: {screenshot_file}")

        
        if 'raw_dom' in observation and observation['raw_dom']:
            raw_dom_file = f"{prefix}_{step}_{timestamp}_raw.html"
            raw_dom_path = os.path.join(self.result_dir, raw_dom_file)
            
            with open(raw_dom_path, 'w', encoding='utf-8') as f:
                f.write(observation['raw_dom'])
            logger.info(f"  📄 Saved raw DOM: {raw_dom_file}")

        
        if 'dom' in observation and observation['dom']:
            if self.dom_filter_selectors:
                
                dom_file = f"{prefix}_{step}_{timestamp}_observation.html"
            else:
                
                dom_file = f"{prefix}_{step}_{timestamp}.html"
            
            dom_path = os.path.join(self.result_dir, dom_file)
            
            with open(dom_path, 'w', encoding='utf-8') as f:
                f.write(observation['dom'])
            logger.info(f"  📄 Saved observation DOM: {dom_file}")

        
        if 'page_info' in observation:
            info_file = f"{prefix}_{step}_{timestamp}_info.json"
            info_path = os.path.join(self.result_dir, info_file)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(observation['page_info'], f, indent=2, ensure_ascii=False)
            logger.info(f"  📋 Saved page info: {info_file}")

    def _get_system_prompt(self) -> str:
        
        mode_name = 'DOM Mode (selectors available)' if self.use_dom else 'Vision-Only Mode (coordinates only)'
        selector_instruction = "Use CSS selectors or coordinates" if self.use_dom else "**You can only use coordinates** (no DOM access, no selectors)"
        
        
        is_macos = platform.system() == 'Darwin'
        modifier_key = 'Meta' if is_macos else 'Control'
        modifier_name = 'Command' if is_macos else 'Control'
        
        prompt = f"""You are a web GUI automation agent that can interact with web pages using Playwright.

Available Actions:
{get_action_description()}

**CRITICAL INSTRUCTIONS:**

**Mode: {mode_name}**

**For CLICK actions:**
- {selector_instruction}
"""

        if self.use_dom:
            prompt += """- Use CSS selectors based on DOM analysis:
  - **Text-based (most reliable)**: `button:has-text("Text")`, `a:has-text("Link")`, `text=Exact Text`
  - **Element + class**: `button.class-name`, `div.card`, `input.search-input`
  - **IDs**: `#element-id` (when available)
  - **Attributes**: `[placeholder="Search"]`, `[type="submit"]`
  - **Combination**: `button.primary:has-text("Submit")`
  
- **Selector strategy** (try in order):
  1. Text-based: `element:has-text("visible text")` - works if text is visible in DOM
  2. ID: `#unique-id` - most reliable if element has ID
  3. Class: `.class-name` or `element.class-name` - common pattern
  4. **Fallback: Use coordinates for dynamic elements** - if selector fails 2+ times, use visual coordinates
  
- **Important rules**:
  - Keep selectors SIMPLE - avoid deep nesting
  - Use `:has-text()` for flexibility (partial match)
  - Check element TYPE first (button, input, a, div, etc.)
  - If selector fails 2+ times, try different approach (alternative element, coordinates)
  - Don't use overly complex paths like `div > div > div > button`
  - Don't assume specific class names without seeing DOM
"""
        
        prompt += "\n**Output format:**\n"
        
        if self.use_dom:
            prompt += """```json
{
    "action_type": "CLICK",
    "parameters": {
        "selector": "button:has-text('Blank Note')"
    },
    "reasoning": "Clicking the 'Blank Note' button using text selector"
}
```

Or with coordinates as backup:
```json
{
    "action_type": "CLICK",
    "parameters": {
        "selector": ".btn-new-doc",
        "x": 200,
        "y": 250
    },
    "reasoning": "Dialog button has dynamic class name, using visual coordinates instead"
}
```

**Advanced CLICK options:**
- Right-click: `"button": "right"` (for context menus)
- Double-click: `"click_count": 2` (for edit mode, select word)
- Triple-click: `"click_count": 3` (for select line)

Examples:
```json
{"action_type": "CLICK", "parameters": {"selector": ".item", "button": "right"}, "reasoning": "Right-click to open context menu"}
{"action_type": "CLICK", "parameters": {"selector": ".title", "click_count": 2}, "reasoning": "Double-click to enter edit mode"}
```
"""
        else:
            prompt += f"""```json
{ 
    "action_type": "CLICK",
    "parameters": { 
        "x": 200,
        "y": 250
    } ,
    "reasoning": "Clicking at coordinates (200, 250) based on screenshot analysis"
} 
```

**Advanced CLICK options (also available in coordinate mode):**
- Right-click: `"button": "right"` (for context menus)
- Double-click: `"click_count": 2` (for edit mode, select word)
- Triple-click: `"click_count": 3` (for select line)

Examples:
```json
{ "action_type": "CLICK", "parameters": { "x": 400, "y": 200, "button": "right"} , "reasoning": "Right-click at position"} 
{ "action_type": "CLICK", "parameters": { "x": 300, "y": 150, "click_count": 2} , "reasoning": "Double-click to select word"} 
```

CRITICAL RULES:
- **ALWAYS respond with valid JSON** - Never return plain text explanations
- **Viewport size: {self.controller.viewport_width} x {self.controller.viewport_height}**
- **Coordinates are relative to viewport** (top-left is 0,0, bottom-right is {self.controller.viewport_width},{self.controller.viewport_height})
- **Analyze screenshot dimensions carefully** - estimate element positions as percentages first, then convert to pixels
- **Click at the CENTER of elements** for better accuracy (not edges)
- **For large buttons/cards**, aim for the visual center
- If previous action didn't work, try adjusting coordinates by 50-100 pixels or use a different approach
"""
        
        prompt += "\n**For FILL actions:**\n"
        
        if self.use_dom:
            prompt += """```json
{
    "action_type": "FILL",
    "parameters": {
        "selector": "input[placeholder='Title']",
        "text": "test"
    }
}
```
"""
        else:
            prompt += """```json
{
    "action_type": "CLICK",
    "parameters": {
        "x": 300,
        "y": 150
    },
    "reasoning": "Click on input field first"
}
```
Then use TYPE to input text.
"""
        
        prompt += f"""
**For TYPE actions:**
```json
{ 
    "action_type": "TYPE",
    "parameters": { 
        "text": "hello"
    } 
} 
```

**For HOTKEY actions (keyboard shortcuts):**
```json
{ 
    "action_type": "HOTKEY",
    "parameters": { 
        "keys": "{modifier_key}+A"
    } ,
    "reasoning": "Select all text using {modifier_name}+A"
} 
```

**IMPORTANT - Keyboard Shortcuts:**
- This system is running on **{'macOS' if is_macos else 'Windows/Linux'}**
- For copy/paste/select all operations, use **{modifier_key}** (not Control)
- Examples: `{modifier_key}+A` (select all), `{modifier_key}+C` (copy), `{modifier_key}+V` (paste)
- Common shortcuts: `{modifier_key}+Z` (undo), `{modifier_key}+Shift+Z` (redo)

**Important guidelines:**
"""
        
        if self.use_dom:
            prompt += """- **CLICK**: Always prefer `selector`, only use `x,y` as fallback
- **FILL**: Use `selector` to target input field, then provide `text`
- **TYPE**: Types text at current cursor position
- **WAIT**: Provide time in seconds
- **DONE**: When task is completed successfully
- **FAIL**: When task truly cannot be completed

**Selector Tips:**
- Use `text=` for buttons with text: `text=Save`, `text=Create`
- Use `:has-text()` for partial match: `button:has-text("Save")`
- Combine selectors: `div.card:has(h3:text("Title"))`
"""
        else:
            prompt += """- **CLICK**: Use `x,y` coordinates based on screenshot
- **FILL**: Not available (use CLICK + TYPE instead)
- **TYPE**: Types text at current cursor position (after clicking on input)
- **WAIT**: Provide time in seconds
- **DONE**: When task is completed successfully
- **FAIL**: When task truly cannot be completed

**Coordinate Tips:**
- Study the screenshot carefully before estimating coordinates
- Always click the CENTER of clickable elements
- For buttons/links: estimate the middle point
- For input fields: click first, then TYPE

**ABSOLUTE REQUIREMENT:**
You MUST respond with a JSON object in this exact format:
```json
{
    "action_type": "CLICK",
    "parameters": {...},
    "reasoning": "..."
}
```
NEVER respond with plain text or explanations outside of JSON. If you cannot determine the exact action, make your best estimate and still return valid JSON format.
"""
        
        return prompt

    def _build_prompt(self, observation: Dict[str, Any], task: str) -> str:
        
        page_info = observation.get("page_info", {})
        viewport = page_info.get("viewport", {})
        
        
        viewport_info = f"{viewport.get('width', 'Unknown')}x{viewport.get('height', 'Unknown')}"
        scroll_info = ""
        if viewport.get('scrollX', 0) != 0 or viewport.get('scrollY', 0) != 0:
            scroll_info = f" (scrolled: {viewport.get('scrollX', 0)}, {viewport.get('scrollY', 0)})"

        prompt = f"""Task: {task}

Current page information:
- URL: {page_info.get('url', 'Unknown')}
- Title: {page_info.get('title', 'Unknown')}
- Viewport: {viewport_info}{scroll_info}

Step: {self.current_step + 1}/{self.max_steps}

"""

        
        if self.action_history:
            prompt += "\nAll previous actions:\n"
            for action_entry in self.action_history:  
                action = action_entry["action"]
                result = action_entry.get("result", {})
                
                
                success_mark = '✓' if result.get('success') else '✗'
                prompt += f"  Step {action_entry['step'] + 1}: [{success_mark}] {action.get('action_type')} "
                
                if action.get('parameters'):
                    
                    params = action.get('parameters')
                    if 'selector' in params:
                        prompt += f"selector={params['selector'][:50]}"
                    if 'x' in params and 'y' in params:
                        prompt += f" coords=({params['x']}, {params['y']})"
                    if 'text' in params:
                        prompt += f" text='{params['text'][:30]}'"
                
                
                if not result.get('success') and result.get('error'):
                    error_msg = result['error'][:100]
                    prompt += f" ERROR: {error_msg}"
                
                prompt += "\n"
            
            
            if len(self.action_history) > 0:
                last_action = self.action_history[-1]
                last_result = last_action.get('result', {})
                if not last_result.get('success'):
                    prompt += "\n⚠️ Last action FAILED. Consider:\n"
                    prompt += "  - Try a different selector or approach\n"
                    prompt += "  - Use alternative UI elements\n"
                    prompt += "  - Use coordinates if selector keeps timing out\n"

        prompt += "\nPlease analyze the screenshots (current and recent history) and decide the next action. Output ONLY the JSON action without any additional text."

        return prompt

    async def get_next_action(self, observation: Dict[str, Any], task: str) -> Dict[str, Any]:
        
        
        prompt = self._build_prompt(observation, task)

        
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        
        
        recent_screenshots = []

        
        if len(self.observation_history) > 0:
            
            start_idx = max(0, len(self.observation_history) - 2)
            for i in range(start_idx, len(self.observation_history)):
                if 'screenshot' in self.observation_history[i]:
                    recent_screenshots.append({
                        "step": i,
                        "screenshot": self.observation_history[i]['screenshot']
                    })

        
        for item in recent_screenshots:
            content.append({
                "type": "text",
                "text": f"\n--- Screenshot from Step {item['step'] + 1} ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{item['screenshot']}"
                }
            })

        
        content.append({
            "type": "text",
            "text": f"\n--- Current Screenshot (Step {self.current_step + 1}) ---"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{observation['screenshot']}"
            }
        })

        
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": content
            }
        ]

        
        logger.info(f"Calling API with model: {self.model}")
        logger.info(f"Sending {len(recent_screenshots) + 1} screenshots (recent 2 + current)")
        max_retries = 5
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                response = call_openai(messages, model=self.model, max_tokens=16384)
                logger.info(f"API response: {response[:200]}...")  
                
                action = self._parse_response(response)
                return action
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying API call in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    return {
                        "action_type": "FAIL",
                        "parameters": {"reason": f"API call failed: {str(e)}"}
                    }

    def _save_parse_error_response(self, response: str, error_msg: str, parse_attempts: list, json_str: str = None):
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            error_file = os.path.join(self.result_dir, f"parse_error_{self.current_step}_{timestamp}.txt")
            
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"Parse Error at Step {self.current_step}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Parse Attempts: {parse_attempts}\n")
                f.write("="*80 + "\n\n")
                
                if json_str:
                    f.write("Extracted JSON String:\n")
                    f.write("-"*80 + "\n")
                    f.write(json_str + "\n")
                    f.write("-"*80 + "\n\n")
                
                f.write("Full Raw Response:\n")
                f.write("-"*80 + "\n")
                f.write(response + "\n")
                f.write("-"*80 + "\n")
            
            logger.info(f"💾 Saved parse error response to: {os.path.basename(error_file)}")
        except Exception as e:
            logger.error(f"Failed to save parse error response: {e}")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        
        json_str = None
        parse_attempts = []
        
        try:
            
            if "```json" in response:
                try:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    parse_attempts.append("markdown json block")
                except Exception as e:
                    logger.debug(f"Failed to extract from ```json block: {e}")
            
            
            if not json_str and "```" in response:
                try:
                    json_start = response.find("```") + 3
                    
                    while json_start < len(response) and response[json_start] in ' \t\n\r':
                        json_start += 1
                    
                    if json_start < len(response) and response[json_start:json_start+4].lower() == 'json':
                        json_start += 4
                    while json_start < len(response) and response[json_start] in ' \t\n\r':
                        json_start += 1
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    parse_attempts.append("markdown code block")
                except Exception as e:
                    logger.debug(f"Failed to extract from ``` block: {e}")
            
            
            if not json_str:
                try:
                    start_idx = response.find('{')
                    if start_idx >= 0:
                        
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(response)):
                            if response[i] == '{':
                                brace_count += 1
                            elif response[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        if brace_count == 0:
                            json_str = response[start_idx:end_idx].strip()
                            parse_attempts.append("braces matching")
                except Exception as e:
                    logger.debug(f"Failed to extract JSON by braces: {e}")
            
            
            if not json_str:
                json_str = response.strip()
                parse_attempts.append("direct parse")

            
            if not json_str:
                raise ValueError("Could not extract JSON from response")
            
            action_data = json.loads(json_str)

            
            if "action_type" not in action_data:
                raise ValueError("Missing action_type in response")

            logger.debug(f"✓ Successfully parsed JSON using: {parse_attempts[-1]}")
            return {
                "action_type": action_data["action_type"],
                "parameters": action_data.get("parameters", {}),
                "reasoning": action_data.get("reasoning", "")
            }

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON decode error: {str(e)}")
            logger.debug(f"Parse attempts: {parse_attempts}")
            logger.debug(f"Extracted JSON string: {json_str[:200] if json_str else 'None'}...")
            logger.debug(f"Full response: {response[:500]}...")
            
            
            self._save_parse_error_response(response, str(e), parse_attempts, json_str)
            
            
            return {
                "action_type": "PARSE_ERROR",
                "parameters": {
                    "reason": f"JSON decode error: {str(e)}",
                    "raw_response": response[:500]
                },
                "_parse_attempts": parse_attempts
            }
        except Exception as e:
            logger.warning(f"⚠️ Error parsing API response: {str(e)}")
            logger.debug(f"Parse attempts: {parse_attempts}")
            logger.debug(f"Response: {response[:500]}...")
            
            
            self._save_parse_error_response(response, str(e), parse_attempts, json_str)
            
            
            return {
                "action_type": "PARSE_ERROR",
                "parameters": {
                    "reason": f"Parse error: {str(e)}",
                    "raw_response": response[:500]
                },
                "_parse_attempts": parse_attempts
            }

    async def run_task(self, task: str, start_url: str) -> Dict[str, Any]:
        
        logger.info("="*60)
        logger.info(f"Starting task: {task}")
        logger.info(f"Start URL: {start_url}")
        logger.info("="*60)

        
        await self.controller.execute_action({
            "action_type": "NAVIGATE",
            "parameters": {"url": start_url}
        })

        
        await asyncio.sleep(2)
        
        
        await self.controller.page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.2)
        
        
        viewport_info = await self.controller.get_viewport_info()
        logger.info(f"📐 Viewport: {viewport_info['width']}x{viewport_info['height']} (DPR: {viewport_info['devicePixelRatio']})")
        logger.info(f"📜 Scroll position: ({viewport_info['scrollX']}, {viewport_info['scrollY']})")
        logger.info(f"📄 Document size: {viewport_info['documentWidth']}x{viewport_info['documentHeight']}")

        
        logger.info("\n📸 Saving initial state...")
        initial_observation = await self.controller.get_observation(
            include_screenshot=True,
            include_dom=self.use_dom,  
            dom_format="full"  
        )
        await self._save_observation(initial_observation, step=0, prefix="initial")

        
        traj_file = os.path.join(self.result_dir, "traj.jsonl")

        
        with open(traj_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "step_num": 0,
                "state": "initial",
                "task": task,
                "start_url": start_url,
                "page_info": initial_observation.get('page_info', {}),
                "timestamp": datetime.datetime.now().isoformat()
            }, ensure_ascii=False))
            f.write("\n")

        self.current_step = 0
        start_time = asyncio.get_event_loop().time()
        done = False
        consecutive_parse_errors = 0  
        max_parse_retries = 3  

        while not done and self.current_step < self.max_steps:
            
            
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > self.timeout:
                logger.warning(f"⚠️ Task running longer than expected timeout ({self.timeout}s), but continuing until max_steps")

            
            observation = await self.controller.get_observation(
                include_screenshot=True,
                include_dom=self.use_dom,  
                dom_format="full"  
            )
            self.observation_history.append(observation)

            logger.info(f"\n{'='*60}")
            logger.info(f"Step {self.current_step + 1}/{self.max_steps}")
            logger.info(f"Current URL: {observation.get('page_info', {}).get('url', 'Unknown')}")
            logger.info(f"{'='*60}")

            
            try:
                action = await self.get_next_action(observation, task)
            except Exception as e:
                logger.error(f"Error getting next action: {str(e)}")
                return {
                    "success": False,
                    "reason": "agent_error",
                    "error": str(e),
                    "steps": self.current_step,
                    "action_history": self.action_history
                }

            
            self.action_history.append({
                "step": self.current_step,
                "action": action,
                "result": None  
            })

            logger.info(f"Action: {action.get('action_type')}")
            logger.info(f"Parameters: {action.get('parameters')}")
            logger.info(f"Reasoning: {action.get('reasoning')}")

            
            if action.get("action_type") == "DONE":
                
                logger.info("\n📸 Saving final state...")
                final_observation = await self.controller.get_observation(
                    include_screenshot=True,
                    include_dom=self.use_dom,
                    dom_format="full"  
                )
                await self._save_observation(final_observation, step=self.current_step + 1, prefix="final")

                
                with open(traj_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "step_num": self.current_step + 1,
                        "action": action,
                        "state": "done",
                        "timestamp": datetime.datetime.now().isoformat()
                    }, ensure_ascii=False))
                    f.write("\n")

                logger.info("\n" + "="*60)
                logger.info("✅ Task completed successfully!")
                logger.info("="*60)
                return {
                    "success": True,
                    "reason": "done",
                    "steps": self.current_step + 1,
                    "action_history": self.action_history
                }
            elif action.get("action_type") == "PARSE_ERROR":
                
                consecutive_parse_errors += 1
                logger.warning(f"⚠️ Parse error ({consecutive_parse_errors}/{max_parse_retries})")
                logger.warning(f"Reason: {action.get('parameters', {}).get('reason', 'Unknown')}")
                
                if consecutive_parse_errors >= max_parse_retries:
                    logger.error("\n" + "="*60)
                    logger.error(f"❌ Task failed: Too many consecutive parse errors ({consecutive_parse_errors})")
                    logger.error("="*60)
                    return {
                        "success": False,
                        "reason": "parse_error",
                        "message": f"Failed to parse agent response after {consecutive_parse_errors} attempts",
                        "steps": self.current_step,
                        "action_history": self.action_history
                    }
                
                
                logger.info("🔄 Retrying with fresh observation...")
                
                self.action_history.pop()
                continue
            elif action.get("action_type") == "FAIL":
                logger.info("\n" + "="*60)
                logger.error("❌ Task failed!")
                logger.error(f"Reason: {action.get('parameters', {}).get('reason', 'Unknown')}")
                logger.info("="*60)
                return {
                    "success": False,
                    "reason": "fail",
                    "message": action.get("parameters", {}).get("reason", "Unknown"),
                    "steps": self.current_step + 1,
                    "action_history": self.action_history
                }
            
            
            consecutive_parse_errors = 0

            
            result = await self.controller.execute_action(action)
            
            
            self.action_history[-1]["result"] = result

            if not result.get("success"):
                logger.warning(f"⚠️ Action execution failed: {result.get('error')}")
                logger.info("Agent will see the error and decide how to proceed...")
                
            else:
                logger.info(f"✅ Action executed successfully")

            
            await asyncio.sleep(1)

            
            logger.info(f"\n📸 Saving state after step {self.current_step + 1}...")
            post_action_observation = await self.controller.get_observation(
                include_screenshot=True,
                include_dom=self.use_dom,
                dom_format="full"  
            )
            await self._save_observation(post_action_observation, step=self.current_step + 1, prefix="step")

            
            with open(traj_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "step_num": self.current_step + 1,
                    "action": action,
                    "result": result,
                    "page_info": post_action_observation.get('page_info', {}),
                    "timestamp": datetime.datetime.now().isoformat()
                }, ensure_ascii=False))
                f.write("\n")

            self.current_step += 1

        logger.warning("\n" + "="*60)
        logger.warning("⚠️  Max steps reached")
        logger.warning("="*60)
        return {
            "success": False,
            "reason": "max_steps",
            "steps": self.current_step,
            "action_history": self.action_history
        }

    def reset(self):
        
        self.current_step = 0
        self.action_history = []
        self.observation_history = []


async def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Note Website Agent')
    parser.add_argument('--task', type=str, 
                        default="click the button blank note, and then revise Untitled Document to test, finally save it.",
                        help='Task description')
    parser.add_argument('--url', type=str, default='http://localhost:3001',
                        help='Website start URL')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-5-20250929',
                        help='Model to use')
    parser.add_argument('--max-steps', type=int, default=30,
                        help='Max steps')
    parser.add_argument('--headless', action='store_true',
                        help='Use headless mode')
    parser.add_argument('--no-dom', action='store_true',
                        help='Disable DOM mode (vision only)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset environment before starting task')
    parser.add_argument('--dom-filter', action='store_true',
                        help='Enable DOM filtering (keep only interactive elements)')
    
    args = parser.parse_args()
    
    
    task = args.task
    
    start_url = args.url

    
    model = args.model

    
    controller = PlaywrightController(
        headless=args.headless,
        slow_mo=500 if not args.headless else 300
    )

    try:
        
        await controller.initialize()
        
        
        if args.reset:
            logger.info("🔄 Resetting environment before starting task...")
            await controller.page.goto(start_url, wait_until='load', timeout=60000)
            
            
            storage_cleared = await controller.clear_storage()
            if storage_cleared:
                logger.info("✅ Browser storage cleared")
                
                await controller.page.reload(wait_until='load', timeout=30000)
                await asyncio.sleep(1)
            else:
                logger.warning("⚠️ Failed to clear browser storage")

        
        dom_filter_selectors = []
        if args.dom_filter:
            dom_filter_selectors = [
                
                "button", "a", "input", "textarea", "select", "[onclick]",
                
                "nav", "header", "menu", "[role='navigation']", "[role='menuitem']",
                
                "form", "label", "[role='button']", "[role='link']",
                
                "h1", "h2", "h3", "p[data-*]", "[data-testid]", "[data-id]",
                
                "[role='alert']", "[role='status']", ".notification", ".toast",
                
                "[role='dialog']", "[role='alertdialog']", ".modal", ".popup",
                
                "table", "tr", "td", "li", "[role='listitem']",
            ]
            logger.info(f"✂️  DOM Filter enabled with {len(dom_filter_selectors)} selector groups")

        
        
        
        agent = NoteWebsiteAgent(
            controller=controller,
            model=model,
            max_steps=args.max_steps,
            use_dom=not args.no_dom,
            dom_filter_selectors=dom_filter_selectors if args.dom_filter else None
        )

        
        result = await agent.run_task(task, start_url)

        
        logger.info("\n" + "="*60)
        logger.info("Task Result:")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Reason: {result['reason']}")
        logger.info(f"  Steps: {result['steps']}")
        logger.info("="*60)

        
        result_file = os.path.join(agent.result_dir, "result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\n📊 Result saved to: {result_file}")
        logger.info(f"📂 All results saved in: {agent.result_dir}")
        logger.info(f"\n   Contents:")
        logger.info(f"   - initial_0_*.png/html        : Initial state screenshot and DOM")
        logger.info(f"   - step_N_*.png/html           : Screenshot and DOM after each step")
        logger.info(f"   - final_N_*.png/html          : Final state screenshot and DOM")
        logger.info(f"   - traj.jsonl                  : Complete execution trajectory")
        logger.info(f"   - result.json                 : Final result")

        
        if not args.headless:
            await asyncio.sleep(3)

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
    finally:
        await controller.close()


if __name__ == "__main__":
    asyncio.run(main())
