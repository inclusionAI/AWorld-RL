#!/usr/bin/env python3
# Evaluator for ecom Query 7

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
        self.in_order_history = False
        self.orders = []
        self.current_order = {}
        self.in_order_item = False
        self.in_order_id = False
        self.in_order_date = False
        self.in_order_total = False
        self.in_product_name = False
        self.in_review_section = False
        self.reviews = []
        self.current_review = {}
        self.in_review_rating = False
        self.in_review_text = False
        self.current_text = ""
        self.in_cart = False
        self.cart_items = []
        self.in_cart_item_name = False
        self.current_cart_item = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect cart page
        if 'class' in attrs_dict and 'cart' in attrs_dict['class'].lower():
            self.in_cart = True
        
        # Detect cart items
        if self.in_cart and tag == 'div' and 'class' in attrs_dict:
            if 'cart-item' in attrs_dict['class'] or 'item-name' in attrs_dict['class']:
                self.in_cart_item_name = True
        
        # Detect order history page
        if 'class' in attrs_dict and 'order' in attrs_dict['class'].lower():
            self.in_order_history = True
        
        # Detect order items
        if self.in_order_history and tag == 'div' and 'class' in attrs_dict:
            if 'order-item' in attrs_dict['class'] or 'order-card' in attrs_dict['class']:
                self.in_order_item = True
                self.current_order = {}
        
        # Detect product name in order
        if self.in_order_item and tag in ['h3', 'h4', 'span', 'p'] and 'class' in attrs_dict:
            if 'product' in attrs_dict['class'].lower() or 'name' in attrs_dict['class'].lower():
                self.in_product_name = True
        
        # Detect review section
        if 'class' in attrs_dict and 'review' in attrs_dict['class'].lower():
            self.in_review_section = True
        
        # Detect rating (stars)
        if self.in_review_section and tag == 'span' and 'class' in attrs_dict:
            if 'star' in attrs_dict['class'].lower() or 'rating' in attrs_dict['class'].lower():
                self.in_review_rating = True
        
        # Detect review text
        if self.in_review_section and tag in ['p', 'div', 'span'] and 'class' in attrs_dict:
            if 'text' in attrs_dict['class'].lower() or 'comment' in attrs_dict['class'].lower():
                self.in_review_text = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        # Collect cart item names
        if self.in_cart_item_name:
            if data and not data.isdigit() and '$' not in data:
                self.cart_items.append(data)
        
        # Collect product names from orders
        if self.in_product_name:
            self.current_order['product'] = data
        
        # Collect review text
        if self.in_review_text:
            self.current_text += data + " "
    
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_order_item:
                if self.current_order:
                    self.orders.append(self.current_order)
                self.in_order_item = False
            
            if self.in_review_section:
                if self.current_text.strip():
                    self.reviews.append(self.current_text.strip())
                    self.current_text = ""
                self.in_review_section = False
            
            self.in_cart_item_name = False
        
        if tag in ['h3', 'h4', 'span', 'p']:
            self.in_product_name = False
            self.in_review_rating = False
            self.in_review_text = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for Query 7
        self.checkpoints = {
            "cp1_logged_in": False,
            "cp2_wireless_mouse_purchased": False,
            "cp3_order_found_in_history": False,
            "cp4_order_details_viewed": False,
            "cp5_review_submitted": False,
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

    def check_login(self, html_content: str) -> bool:
        """Check if user is logged in by looking for sign out link or user menu."""
        return 'sign out' in html_content.lower() or 'logout' in html_content.lower() or 'my account' in html_content.lower()

    def check_wireless_mouse_in_cart(self, html_content: str) -> bool:
        """Check if Wireless Mouse is in the cart."""
        return 'wireless mouse' in html_content.lower() and 'cart' in html_content.lower()

    def check_order_history_page(self, html_content: str) -> bool:
        """Check if on order history page."""
        return 'order history' in html_content.lower() or 'my orders' in html_content.lower()

    def check_wireless_mouse_order(self, html_content: str) -> bool:
        """Check if Wireless Mouse order exists in order history."""
        return self.check_order_history_page(html_content) and 'wireless mouse' in html_content.lower()

    def check_review_with_text(self, html_content: str) -> bool:
        """Check if review with expected text exists."""
        expected_text = "excellent quality and fast delivery"
        return expected_text in html_content.lower()

    def check_five_star_rating(self, html_content: str) -> bool:
        """Check for 5-star rating indicators."""
        # Look for rating patterns like "5" or "★★★★★"
        five_star_patterns = [
            r'rating["\s:]*5',
            r'5\s*star',
            r'★★★★★',
            r'⭐⭐⭐⭐⭐',
            r'stars["\s:]*5'
        ]
        for pattern in five_star_patterns:
            if re.search(pattern, html_content.lower()):
                return True
        return False

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: User logged in successfully."""
        try:
            # Check trajectory for successful login
            trajectory = self.get_trajectory()
            for entry in trajectory:
                if entry.get('action', {}).get('action_type') == 'FILL':
                    params = entry.get('action', {}).get('parameters', {})
                    if params.get('text') == '12345@mail.com':
                        # Check if login succeeded
                        final_html = self.find_final_html()
                        if final_html:
                            with open(final_html, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if self.check_login(content):
                                    self.details['login_status'] = 'Logged in successfully'
                                    return True
            
            self.issues.append("User did not log in successfully")
            return False
        except Exception as e:
            self.issues.append(f"Error checking login: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Wireless Mouse purchased."""
        try:
            # Check if purchase was completed by looking for order confirmation or checkout success
            all_steps = self.find_step_html("step_*")
            
            # Look for checkout flow and Wireless Mouse
            found_wireless_mouse_in_cart = False
            found_checkout = False
            
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if self.check_wireless_mouse_in_cart(content):
                        found_wireless_mouse_in_cart = True
                    if 'checkout' in content.lower() and 'wireless mouse' in content.lower():
                        found_checkout = True
                    if 'order confirmation' in content.lower() or 'thank you for your order' in content.lower():
                        if 'wireless mouse' in content.lower():
                            self.details['purchase_status'] = 'Wireless Mouse purchased successfully'
                            return True
            
            if not found_wireless_mouse_in_cart:
                self.issues.append("Wireless Mouse was never added to cart")
            elif not found_checkout:
                self.issues.append("Checkout was not completed")
            else:
                self.issues.append("Purchase was not completed")
            
            return False
        except Exception as e:
            self.issues.append(f"Error checking purchase: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Order found in order history."""
        try:
            all_steps = self.find_step_html("step_*")
            
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if self.check_wireless_mouse_order(content):
                        self.details['order_history'] = 'Found Wireless Mouse order in history'
                        return True
            
            self.issues.append("Wireless Mouse order not found in order history")
            return False
        except Exception as e:
            self.issues.append(f"Error checking order history: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Order details viewed."""
        try:
            all_steps = self.find_step_html("step_*")
            
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for order details page with Wireless Mouse
                    if ('order details' in content.lower() or 'order #' in content.lower()) and 'wireless mouse' in content.lower():
                        self.details['order_details'] = 'Order details viewed'
                        return True
            
            self.issues.append("Order details were not viewed")
            return False
        except Exception as e:
            self.issues.append(f"Error checking order details: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: 5-star review submitted with correct text."""
        try:
            all_steps = self.find_step_html("step_*")
            final_html = self.find_final_html()
            
            files_to_check = all_steps + ([final_html] if final_html else [])
            
            for step_file in files_to_check:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    has_review_text = self.check_review_with_text(content)
                    has_five_stars = self.check_five_star_rating(content)
                    
                    if has_review_text and has_five_stars:
                        self.details['review_status'] = '5-star review submitted with correct text'
                        return True
                    elif has_review_text:
                        self.issues.append("Review text found but 5-star rating not confirmed")
                    elif has_five_stars:
                        self.issues.append("5-star rating found but review text incorrect")
            
            self.issues.append("Review was not submitted with correct text and rating")
            return False
        except Exception as e:
            self.issues.append(f"Error checking review: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_logged_in'] = self.check_checkpoint_1()
            self.checkpoints['cp2_wireless_mouse_purchased'] = self.check_checkpoint_2()
            self.checkpoints['cp3_order_found_in_history'] = self.check_checkpoint_3()
            self.checkpoints['cp4_order_details_viewed'] = self.check_checkpoint_4()
            self.checkpoints['cp5_review_submitted'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 7,
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
                'query_id': 7,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
