#!/usr/bin/env python3
# Evaluator for food Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.in_cart = False
        self.in_cart_item = False
        self.current_item_name = ""
        self.current_item_size = ""
        self.current_customization = ""
        self.cart_items = []
        self.in_item_name = False
        self.in_size_section = False
        self.in_customization_section = False
        self.current_tag = None
        self.current_attrs = []

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Detect cart section
        if tag == 'div':
            class_attr = self.current_attrs.get('class', '')
            if 'cart' in class_attr.lower():
                self.in_cart = True

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        # Look for pizza items in cart
        if self.in_cart:
            if 'margherita' in text.lower() and 'pizza' in text.lower():
                self.current_item_name = text
                self.in_cart_item = True
            elif self.in_cart_item:
                if 'large' in text.lower():
                    self.current_item_size = text
                elif 'extra cheese' in text.lower():
                    self.current_customization = text
                    
    def handle_endtag(self, tag):
        if tag == 'div' and self.in_cart_item and self.current_item_name:
            self.cart_items.append({
                'name': self.current_item_name,
                'size': self.current_item_size,
                'customization': self.current_customization
            })
            self.in_cart_item = False
            self.current_item_name = ""
            self.current_item_size = ""
            self.current_customization = ""


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_restaurant_found": False,
            "cp2_menu_accessed": False,
            "cp3_pizza_found": False,
            "cp4_added_to_cart": False,
        }

        self.issues = []
        self.details = {}

    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        if step_files:
            return step_files[-1]
        return None

    def find_step_html(self, step_pattern: str) -> List[Path]:
        """Find intermediate step HTML files."""
        return sorted(self.result_dir.glob(f"{step_pattern}_raw.html"))

    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data from traj.jsonl."""
        if not self.traj_file.exists():
            return []

        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Restaurant 'Bella Italia' was found in search results"""
        try:
            result_data = self.load_result_json()
            
            # Check if any step successfully navigated to Bella Italia
            # Look for successful search or navigation actions
            for action in result_data.get('action_history', []):
                if action['step'] <= 5:  # Check early steps
                    # Check if search was performed
                    if action['action'].get('action_type') == 'FILL':
                        params = action['action'].get('parameters', {})
                        if 'Bella Italia' in params.get('text', ''):
                            # Search was performed - but we know it returned 0 results
                            pass
                    
                    # Check if View Menu was clicked
                    if action['action'].get('action_type') == 'CLICK':
                        result = action.get('result', {})
                        if result.get('success') and 'View Menu' in str(action['action'].get('parameters', {})):
                            # At least got to a restaurant page initially
                            self.details['search_attempted'] = True
                            
            # Check intermediate HTML files to see if restaurant was ever found
            search_steps = self.find_step_html("step_[23]*")
            for step_file in search_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Look for "Bella Italia" restaurant card AND successful results
                    if 'Bella Italia' in html_content and '2 restaurants found' in html_content:
                        return True
                        
            # Final check - the result shows restaurant was never found
            if '0 restaurants found' in result_data.get('message', ''):
                self.issues.append("Restaurant 'Bella Italia' does not exist in the database")
                return False
                
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking restaurant found: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Successfully accessed Bella Italia menu page"""
        try:
            result_data = self.load_result_json()
            
            # Check if any step successfully accessed the menu
            for action in result_data.get('action_history', []):
                if action['action'].get('action_type') == 'CLICK':
                    params = action['action'].get('parameters', {})
                    if 'View Menu' in str(params):
                        # Check next step to see if menu was loaded
                        next_step = action['step'] + 1
                        # Find next action
                        for next_action in result_data.get('action_history', []):
                            if next_action['step'] == next_step:
                                # Check if we're on menu page (not going back immediately)
                                if next_action['action'].get('action_type') != 'GO_BACK':
                                    # Check HTML to see if menu was there
                                    step_files = self.find_step_html(f"step_{next_step}_*")
                                    if step_files:
                                        with open(step_files[0], 'r', encoding='utf-8') as f:
                                            html = f.read()
                                            if 'Search menu items' in html or 'menu items' in html.lower():
                                                return True
            
            self.issues.append("Never successfully accessed Bella Italia menu page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking menu access: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Found 'Margherita Pizza' in the menu"""
        try:
            # Check all step files for Margherita Pizza in menu
            all_steps = self.find_step_html("step_*")
            
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Look for Margherita Pizza menu item
                    if 'margherita' in html_content.lower() and 'pizza' in html_content.lower():
                        # Make sure it's in a menu context, not just search
                        if 'menu' in html_content.lower() and 'add to cart' in html_content.lower():
                            return True
            
            self.issues.append("Margherita Pizza not found in menu (restaurant may not have menu items)")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking pizza found: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Added Margherita Pizza (large, extra cheese) to cart"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML to find cart items
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            # Check if Margherita Pizza with correct specs is in cart
            for item in parser.cart_items:
                if 'margherita' in item['name'].lower() and 'pizza' in item['name'].lower():
                    if 'large' in item['size'].lower() and 'extra cheese' in item['customization'].lower():
                        return True
            
            # Alternative: search for cart indicators in HTML
            if 'margherita' in html_content.lower() and 'pizza' in html_content.lower():
                if 'large' in html_content.lower() and 'extra cheese' in html_content.lower():
                    if 'cart' in html_content.lower() or 'checkout' in html_content.lower():
                        return True
            
            self.issues.append("Margherita Pizza (large, extra cheese) not found in cart")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking cart: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_restaurant_found'] = self.check_checkpoint_1()
            self.checkpoints['cp2_menu_accessed'] = self.check_checkpoint_2()
            self.checkpoints['cp3_pizza_found'] = self.check_checkpoint_3()
            self.checkpoints['cp4_added_to_cart'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 6,
                'query': result_data.get('query', ''),
                'overall_success': overall_success,
                'success_rate': success_rate,
                'checkpoints': self.checkpoints,
                'checkpoints_passed': f"{passed_count}/{total_count}",
                'issues': self.issues,
                'details': self.details,
                'agent_claimed_success': result_data.get('success', False),
                'execution_time': result_data.get('execution_time', 0),
                'total_steps': result_data.get('steps', 0)
            }

        except Exception as e:
            return {
                'query_id': 6,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python food_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
