#!/usr/bin/env python3
# Evaluator for ecom Query 11

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
        self.in_title = False
        self.page_title = ""
        self.in_product_title = False
        self.product_titles = []
        self.current_product_title = ""
        self.in_order_details = False
        self.order_info = {}
        self.current_tag = None
        self.current_attrs = {}
        self.in_success_message = False
        self.success_messages = []
        self.current_text = ""

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        if tag == "title":
            self.in_title = True
        
        # Check for product titles
        if tag == "h1" or tag == "h2":
            self.in_product_title = True
            self.current_product_title = ""

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
            
        if self.in_title:
            self.page_title += text
        
        if self.in_product_title:
            self.current_product_title += text
        
        self.current_text = text

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        
        if tag in ["h1", "h2"] and self.in_product_title:
            self.in_product_title = False
            if self.current_product_title:
                self.product_titles.append(self.current_product_title.strip())


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_account_created": False,
            "cp2_product_added_to_cart": False,
            "cp3_checkout_with_address": False,
            "cp4_payment_completed": False,
            "cp5_order_placed": False,
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

    def check_html_content(self, html_path: Path, search_patterns: List[str]) -> bool:
        """Check if HTML contains any of the search patterns."""
        if not html_path.exists():
            return False
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in search_patterns:
                    if pattern.lower() in content.lower():
                        return True
        except Exception as e:
            self.issues.append(f"Error reading {html_path}: {str(e)}")
        
        return False

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Account created successfully with email newcustomer@test.com"""
        try:
            # Check trajectory for account creation steps
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for successful account creation indicators
            # Check if they reached "Create Account" page and filled the form
            create_account_reached = False
            email_filled = False
            password_filled = False
            
            for action in action_history:
                reasoning = action.get('action', {}).get('reasoning', '').lower()
                params = action.get('action', {}).get('parameters', {})
                
                # Check if they navigated to sign up page
                if 'sign up' in reasoning or 'create account' in reasoning:
                    create_account_reached = True
                
                # Check if they filled email with newcustomer@test.com
                if 'newcustomer@test.com' in str(params.get('text', '')):
                    email_filled = True
                
                # Check if they filled password
                if params.get('text') == 'pass' and 'password' in reasoning:
                    password_filled = True
            
            # Check intermediate steps for successful registration
            step_files = self.find_step_html("step_*")
            for step_file in step_files:
                if self.check_html_content(step_file, [
                    'welcome',
                    'account created',
                    'registration successful',
                    'newcustomer@test.com'
                ]):
                    self.details['account_creation_detected'] = True
                    return True
            
            # Check final state for logged in status with new account
            final_html = self.find_final_html()
            if final_html and self.check_html_content(final_html, [
                'newcustomer@test.com',
                'account',
                'logout',
                'sign out'
            ]):
                self.details['logged_in_as_new_customer'] = True
                return True
            
            if create_account_reached and email_filled and password_filled:
                self.details['account_form_filled'] = True
                self.issues.append("Account form was filled but registration may not have completed")
                return False
            
            self.issues.append("Account creation not completed")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking account creation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Wireless Mouse product added to cart"""
        try:
            # Check trajectory for product search and add to cart
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            wireless_mouse_found = False
            add_to_cart_action = False
            
            for action in action_history:
                reasoning = action.get('action', {}).get('reasoning', '').lower()
                
                if 'wireless mouse' in reasoning:
                    wireless_mouse_found = True
                
                if 'add to cart' in reasoning or 'cart' in reasoning:
                    add_to_cart_action = True
            
            # Check intermediate steps for cart with Wireless Mouse
            step_files = self.find_step_html("step_*")
            for step_file in step_files:
                if self.check_html_content(step_file, [
                    'wireless mouse',
                    'cart',
                    'shopping cart'
                ]):
                    self.details['wireless_mouse_in_cart'] = True
                    return True
            
            # Check final HTML for cart contents
            final_html = self.find_final_html()
            if final_html and self.check_html_content(final_html, [
                'wireless mouse'
            ]):
                content = final_html.read_text(encoding='utf-8')
                if 'cart' in content.lower():
                    self.details['wireless_mouse_in_final_cart'] = True
                    return True
            
            if wireless_mouse_found:
                self.issues.append("Wireless Mouse was searched but may not have been added to cart")
            else:
                self.issues.append("No evidence of Wireless Mouse product interaction")
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking product in cart: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Checkout process with correct address"""
        try:
            # Check for address: 456 Oak Ave, New York, NY 10001, USA
            required_address_parts = ['456 oak ave', 'new york', '10001']
            
            # Check intermediate steps for checkout with address
            step_files = self.find_step_html("step_*")
            for step_file in step_files:
                content = step_file.read_text(encoding='utf-8').lower()
                if 'checkout' in content:
                    address_match_count = sum(1 for part in required_address_parts if part in content)
                    if address_match_count >= 2:
                        self.details['checkout_with_correct_address'] = True
                        return True
            
            # Check trajectory for address entry
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            address_filled = False
            for action in action_history:
                params = action.get('action', {}).get('parameters', {})
                text = str(params.get('text', '')).lower()
                
                if '456 oak ave' in text or '10001' in text:
                    address_filled = True
                    break
            
            if address_filled:
                self.issues.append("Address was entered but checkout may not have been completed")
            else:
                self.issues.append("Required address not entered during checkout")
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkout address: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Payment with credit card ending in 1234"""
        try:
            # Check for credit card ending in 1234
            step_files = self.find_step_html("step_*")
            for step_file in step_files:
                content = step_file.read_text(encoding='utf-8')
                if re.search(r'1234', content) and ('payment' in content.lower() or 'card' in content.lower()):
                    self.details['payment_card_1234_used'] = True
                    return True
            
            # Check trajectory for card number entry
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            for action in action_history:
                params = action.get('action', {}).get('parameters', {})
                text = str(params.get('text', ''))
                
                if '1234' in text and len(text) >= 15:  # Credit card number length
                    self.details['card_ending_1234_entered'] = True
                    return True
            
            self.issues.append("Credit card ending in 1234 not detected in payment")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking payment: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Order successfully placed"""
        try:
            # Check for order confirmation
            step_files = self.find_step_html("step_*")
            for step_file in step_files:
                if self.check_html_content(step_file, [
                    'order confirmation',
                    'thank you for your order',
                    'order placed',
                    'order #',
                    'order number'
                ]):
                    self.details['order_confirmation_found'] = True
                    return True
            
            # Check final HTML for order confirmation
            final_html = self.find_final_html()
            if final_html and self.check_html_content(final_html, [
                'order confirmation',
                'thank you',
                'order placed',
                'order number'
            ]):
                self.details['order_in_final_state'] = True
                return True
            
            self.issues.append("No order confirmation detected")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking order placement: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_account_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_product_added_to_cart'] = self.check_checkpoint_2()
            self.checkpoints['cp3_checkout_with_address'] = self.check_checkpoint_3()
            self.checkpoints['cp4_payment_completed'] = self.check_checkpoint_4()
            self.checkpoints['cp5_order_placed'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 11,
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
                'query_id': 11,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
