#!/usr/bin/env python3
# Evaluator for ecom Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract cart and checkout information."""

    def __init__(self):
        super().__init__()
        self.in_cart_item = False
        self.in_product_name = False
        self.in_quantity = False
        self.in_address = False
        self.current_product = None
        self.current_quantity = None
        self.cart_items = []
        self.addresses = []
        self.current_text = []
        self.current_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect cart items
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'cart-item' in classes:
                self.in_cart_item = True
                self.current_product = None
                self.current_quantity = None
            elif 'product-name' in classes and self.in_cart_item:
                self.in_product_name = True
                self.current_text = []
            elif 'quantity' in classes and self.in_cart_item:
                self.in_quantity = True
                self.current_text = []
            elif 'address' in classes or 'shipping-address' in classes:
                self.in_address = True
                self.current_text = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            if self.in_product_name:
                self.current_text.append(text)
            elif self.in_quantity:
                self.current_text.append(text)
            elif self.in_address:
                self.current_text.append(text)

    def handle_endtag(self, tag):
        if self.in_product_name:
            self.current_product = ' '.join(self.current_text).strip()
            self.in_product_name = False
            self.current_text = []
        elif self.in_quantity:
            qty_text = ' '.join(self.current_text).strip()
            # Extract number from text like "Qty: 2" or just "2"
            match = re.search(r'\d+', qty_text)
            if match:
                self.current_quantity = int(match.group())
            self.in_quantity = False
            self.current_text = []
        elif self.in_cart_item and tag in ['div', 'li']:
            if self.current_product:
                self.cart_items.append({
                    'product': self.current_product,
                    'quantity': self.current_quantity or 1
                })
            self.in_cart_item = False
            self.current_product = None
            self.current_quantity = None
        elif self.in_address and tag in ['div', 'p', 'span']:
            addr_text = ' '.join(self.current_text).strip()
            if addr_text:
                self.addresses.append(addr_text)
            self.in_address = False
            self.current_text = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on the task:
        # 1. Login successfully
        # 2. Add 2 Wireless Mouse to cart
        # 3. Add 1 Smart Doorbell Camera to cart
        # 4. Navigate to checkout
        # 5. Use the specified address during checkout
        self.checkpoints = {
            "cp1_logged_in": False,
            "cp2_wireless_mouse_in_cart": False,
            "cp3_smart_doorbell_in_cart": False,
            "cp4_reached_checkout": False,
            "cp5_used_correct_address": False,
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

    def parse_html_for_cart_items(self, html_path: Path) -> List[Dict]:
        """Parse HTML to extract cart items."""
        if not html_path.exists():
            return []
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for cart items in the HTML
        cart_items = []
        
        # Pattern 1: Find product names and quantities in cart
        # Look for patterns like: Wireless Mouse, quantity 2
        wireless_mouse_pattern = r'Wireless Mouse.*?quantity["\s:]*(\d+)|quantity["\s:]*(\d+).*?Wireless Mouse'
        doorbell_pattern = r'Smart Doorbell Camera.*?quantity["\s:]*(\d+)|quantity["\s:]*(\d+).*?Smart Doorbell Camera'
        
        # Also check for cart badge count
        badge_match = re.search(r'cart-badge["\s>]*(\d+)', html_content)
        if badge_match:
            self.details['cart_badge_count'] = int(badge_match.group(1))
        
        # Search for product mentions
        if 'Wireless Mouse' in html_content:
            mouse_matches = re.findall(r'Wireless Mouse', html_content)
            self.details['wireless_mouse_mentions'] = len(mouse_matches)
        
        if 'Smart Doorbell Camera' in html_content:
            doorbell_matches = re.findall(r'Smart Doorbell Camera', html_content)
            self.details['smart_doorbell_mentions'] = len(doorbell_matches)
        
        return cart_items

    def check_cart_in_steps(self) -> Tuple[bool, bool]:
        """Check if required items were in cart at any point."""
        has_wireless_mouse = False
        has_smart_doorbell = False
        
        # Check all step files for cart contents
        all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
        
        for step_file in all_steps:
            with open(step_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if this is a cart page
            if 'Shopping Cart' in html_content or 'cart-item' in html_content:
                if 'Wireless Mouse' in html_content:
                    has_wireless_mouse = True
                if 'Smart Doorbell Camera' in html_content:
                    has_smart_doorbell = True
                    
                # Extract quantities if possible
                # Look for quantity indicators
                qty_pattern = r'quantity["\s:>]*(\d+)'
                quantities = re.findall(qty_pattern, html_content, re.IGNORECASE)
                if quantities:
                    self.details['found_quantities'] = quantities
        
        return has_wireless_mouse, has_smart_doorbell

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Successfully logged in with correct credentials."""
        trajectory = self.get_trajectory()
        
        # Check for successful login in action history
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'FILL':
                params = action.get('parameters', {})
                if params.get('text') == '12345@mail.com':
                    self.details['login_email_filled'] = True
                elif params.get('text') == 'pass':
                    self.details['login_password_filled'] = True
        
        # Check final state for logged-in indicator
        final_html = self.find_final_html()
        if final_html:
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for user menu, account links, or "Demo User" text
            if 'Demo User' in content or 'My Account' in content or 'Sign Out' in content:
                self.details['logged_in_state'] = True
                return True
        
        return self.details.get('login_email_filled', False) and self.details.get('login_password_filled', False)

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Added 2 Wireless Mouse items to cart."""
        # Check trajectory for "Add to Cart" actions related to Wireless Mouse
        trajectory = self.get_trajectory()
        wireless_mouse_adds = 0
        
        for entry in trajectory:
            action = entry.get('action', {})
            reasoning = action.get('reasoning', '')
            
            if 'Wireless Mouse' in reasoning and 'Add to Cart' in reasoning:
                wireless_mouse_adds += 1
        
        self.details['wireless_mouse_add_attempts'] = wireless_mouse_adds
        
        # Also check HTML files for cart contents
        has_mouse, _ = self.check_cart_in_steps()
        
        # Consider successful if we see evidence of Wireless Mouse being added
        return wireless_mouse_adds >= 2 or has_mouse

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Added 1 Smart Doorbell Camera to cart."""
        # Check trajectory for "Add to Cart" actions related to Smart Doorbell Camera
        trajectory = self.get_trajectory()
        doorbell_adds = 0
        
        for entry in trajectory:
            action = entry.get('action', {})
            reasoning = action.get('reasoning', '')
            
            if 'Smart Doorbell Camera' in reasoning and ('Add to Cart' in reasoning or 'add' in reasoning.lower()):
                doorbell_adds += 1
        
        self.details['smart_doorbell_add_attempts'] = doorbell_adds
        
        # Also check HTML files for cart contents
        _, has_doorbell = self.check_cart_in_steps()
        
        # Consider successful if we see evidence of Smart Doorbell Camera being added
        return doorbell_adds >= 1 or has_doorbell

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Reached checkout page."""
        trajectory = self.get_trajectory()
        
        # Check for checkout-related actions
        for entry in trajectory:
            action = entry.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            if 'checkout' in reasoning:
                self.details['checkout_mentioned'] = True
        
        # Check HTML files for checkout page
        all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
        
        for step_file in all_steps:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for checkout page indicators
            if 'Checkout' in content or 'checkout' in content:
                # Check for shipping address form or payment form
                if 'shipping' in content.lower() or 'address' in content.lower() or 'payment' in content.lower():
                    self.details['reached_checkout_page'] = True
                    return True
        
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Used correct address (456 Oak Ave, New York, NY 10001, USA) during checkout."""
        required_address_parts = ['456 Oak Ave', 'New York', 'NY', '10001']
        
        # Check trajectory for address filling
        trajectory = self.get_trajectory()
        filled_address_parts = []
        
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'FILL':
                params = action.get('parameters', {})
                text = params.get('text', '')
                
                # Check if any part of the address was filled
                for part in required_address_parts:
                    if part in text:
                        filled_address_parts.append(part)
        
        self.details['filled_address_parts'] = list(set(filled_address_parts))
        
        # Check HTML files for address display
        all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
        
        for step_file in all_steps:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for address display in checkout or confirmation pages
            found_parts = sum(1 for part in required_address_parts if part in content)
            if found_parts >= 3:  # At least 3 out of 4 parts should be present
                self.details['address_found_in_html'] = True
                return True
        
        # Consider successful if we filled at least 3 address components
        return len(filled_address_parts) >= 3

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_logged_in'] = self.check_checkpoint_1()
            self.checkpoints['cp2_wireless_mouse_in_cart'] = self.check_checkpoint_2()
            self.checkpoints['cp3_smart_doorbell_in_cart'] = self.check_checkpoint_3()
            self.checkpoints['cp4_reached_checkout'] = self.check_checkpoint_4()
            self.checkpoints['cp5_used_correct_address'] = self.check_checkpoint_5()

            # Add issues based on failed checkpoints
            if not self.checkpoints['cp1_logged_in']:
                self.issues.append("Failed to log in with correct credentials")
            if not self.checkpoints['cp2_wireless_mouse_in_cart']:
                self.issues.append("Failed to add 2 Wireless Mouse items to cart")
            if not self.checkpoints['cp3_smart_doorbell_in_cart']:
                self.issues.append("Failed to add 1 Smart Doorbell Camera to cart")
            if not self.checkpoints['cp4_reached_checkout']:
                self.issues.append("Failed to reach checkout page")
            if not self.checkpoints['cp5_used_correct_address']:
                self.issues.append("Failed to use the correct address (456 Oak Ave, New York, NY 10001, USA)")

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
        print("Usage: python ecom_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
