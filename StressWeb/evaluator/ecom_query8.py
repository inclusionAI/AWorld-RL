#!/usr/bin/env python3
# Evaluator for ecom Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class CartHTMLParser(HTMLParser):
    """Parse HTML to extract cart items."""

    def __init__(self):
        super().__init__()
        self.cart_items = []
        self.in_cart_item = False
        self.in_item_name = False
        self.in_item_price = False
        self.in_item_quantity = False
        self.current_item = {}
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect cart item containers
        if tag == 'div' and 'class' in attrs_dict:
            if 'cart-item' in attrs_dict['class'] or 'item' in attrs_dict['class']:
                self.in_cart_item = True
                self.current_item = {}
        
        # Detect item name
        if self.in_cart_item and tag in ['h3', 'h4', 'div', 'span']:
            if 'class' in attrs_dict and any(x in attrs_dict['class'].lower() for x in ['name', 'title', 'product']):
                self.in_item_name = True
                self.current_data = []

    def handle_data(self, data):
        data = data.strip()
        if data:
            if self.in_item_name:
                self.current_data.append(data)

    def handle_endtag(self, tag):
        if self.in_item_name and tag in ['h3', 'h4', 'div', 'span']:
            self.in_item_name = False
            if self.current_data:
                self.current_item['name'] = ' '.join(self.current_data)
                self.cart_items.append(self.current_item.copy())

        if self.in_cart_item and tag == 'div':
            self.in_cart_item = False


class OrderHistoryParser(HTMLParser):
    """Parse HTML to extract order information."""

    def __init__(self):
        super().__init__()
        self.orders = []
        self.in_order = False
        self.in_order_id = False
        self.in_order_status = False
        self.in_order_items = False
        self.current_order = {}
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict:
            if 'order' in attrs_dict['class'].lower():
                self.in_order = True
                self.current_order = {'items': []}

    def handle_data(self, data):
        data = data.strip()
        if data and self.in_order:
            # Check for order ID pattern
            if re.match(r'Order #?\d+', data, re.IGNORECASE):
                self.current_order['order_id'] = data
            # Check for status
            if data.lower() in ['pending', 'processing', 'completed', 'delivered', 'cancelled']:
                self.current_order['status'] = data

    def handle_endtag(self, tag):
        if self.in_order and tag == 'div':
            if self.current_order:
                self.orders.append(self.current_order.copy())
            self.in_order = False


class ReviewParser(HTMLParser):
    """Parse HTML to check for review forms or submitted reviews."""

    def __init__(self):
        super().__init__()
        self.has_review_form = False
        self.has_rating_buttons = False
        self.submitted_reviews = []
        self.in_review = False
        self.current_rating = None
        self.review_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for review form
        if tag == 'form' and 'class' in attrs_dict:
            if 'review' in attrs_dict['class'].lower():
                self.has_review_form = True
        
        # Check for rating buttons/stars
        if tag == 'button' and 'data-rating' in attrs_dict:
            self.has_rating_buttons = True
        
        # Check for star icons
        if tag == 'i' and 'class' in attrs_dict:
            if 'star' in attrs_dict['class'].lower():
                self.has_rating_buttons = True

    def handle_data(self, data):
        data = data.strip()
        if data:
            # Look for rating patterns
            rating_match = re.search(r'(\d)\s*star', data.lower())
            if rating_match:
                self.current_rating = int(rating_match.group(1))


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_logged_in": False,
            "cp2_items_added_to_cart": False,
            "cp3_checkout_completed": False,
            "cp4_order_found_in_history": False,
            "cp5_reviews_submitted": False,
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
        """Checkpoint 1: User logged in with correct credentials"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for successful login with correct email
            email_filled = False
            password_filled = False
            login_submitted = False
            
            for action in action_history:
                act = action.get('action', {})
                params = act.get('parameters', {})
                result = action.get('result', {})
                
                if act.get('action_type') == 'FILL' and result.get('success'):
                    if params.get('text') == '12345@mail.com':
                        email_filled = True
                    if params.get('text') == 'pass':
                        password_filled = True
                
                # Check for sign-in button click after credentials
                if email_filled and password_filled and act.get('action_type') == 'CLICK':
                    if result.get('success'):
                        login_submitted = True
            
            if email_filled and password_filled and login_submitted:
                self.details['login'] = 'Successfully logged in with 12345@mail.com'
                return True
            else:
                self.issues.append(f"Login incomplete - email:{email_filled}, password:{password_filled}, submitted:{login_submitted}")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking login: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Added Wireless Mouse, USB-C Cable, and HDMI Cable to cart"""
        try:
            # Check intermediate steps for cart additions
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for Electronics navigation and Add to Cart actions
            navigated_to_electronics = False
            add_to_cart_count = 0
            
            for action in action_history:
                act = action.get('action', {})
                params = act.get('parameters', {})
                result = action.get('result', {})
                reasoning = act.get('reasoning', '')
                
                # Check Electronics navigation
                if 'Electronics' in str(params) or 'electronics' in reasoning.lower():
                    if result.get('success'):
                        navigated_to_electronics = True
                
                # Count Add to Cart clicks after Electronics navigation
                if navigated_to_electronics and act.get('action_type') == 'CLICK':
                    if 'Add to Cart' in str(params) and result.get('success'):
                        add_to_cart_count += 1
            
            # Check cart page for the three items
            cart_files = self.find_step_html("step_*")
            items_found = {'wireless mouse': False, 'usb-c cable': False, 'hdmi cable': False}
            
            for cart_file in cart_files:
                with open(cart_file, 'r', encoding='utf-8') as f:
                    html_content = f.read().lower()
                    
                    if 'wireless mouse' in html_content:
                        items_found['wireless mouse'] = True
                    if 'usb-c cable' in html_content or 'usb c cable' in html_content:
                        items_found['usb-c cable'] = True
                    if 'hdmi cable' in html_content:
                        items_found['hdmi cable'] = True
            
            all_items_found = all(items_found.values())
            
            if all_items_found:
                self.details['cart_items'] = 'All three required items added to cart'
                return True
            else:
                missing = [k for k, v in items_found.items() if not v]
                self.issues.append(f"Missing items in cart: {missing}")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking cart items: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Checkout completed successfully"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for checkout-related navigation and completion
            checkout_initiated = False
            order_placed = False
            
            for action in action_history:
                act = action.get('action', {})
                params = act.get('parameters', {})
                result = action.get('result', {})
                reasoning = act.get('reasoning', '').lower()
                
                # Check for checkout navigation
                if 'checkout' in reasoning or 'proceed to checkout' in str(params).lower():
                    if result.get('success'):
                        checkout_initiated = True
                
                # Check for order placement
                if 'place order' in reasoning or 'complete order' in reasoning:
                    if result.get('success'):
                        order_placed = True
            
            # Check HTML for order confirmation
            step_files = sorted(self.find_step_html("step_*"))
            if step_files:
                # Check later steps for order confirmation
                for step_file in step_files[-20:]:  # Check last 20 steps
                    with open(step_file, 'r', encoding='utf-8') as f:
                        html_content = f.read().lower()
                        
                        # Look for order confirmation indicators
                        if any(phrase in html_content for phrase in [
                            'order confirmation',
                            'order complete',
                            'thank you for your order',
                            'order placed',
                            'order #',
                            'order number'
                        ]):
                            self.details['checkout'] = 'Order placed successfully'
                            return True
            
            # If checkout was initiated but not confirmed
            if checkout_initiated:
                self.issues.append("Checkout initiated but not completed")
            else:
                self.issues.append("Checkout not initiated")
            
            return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkout: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Found order in order history"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for order history navigation
            order_history_accessed = False
            
            for action in action_history:
                act = action.get('action', {})
                params = act.get('parameters', {})
                result = action.get('result', {})
                reasoning = act.get('reasoning', '').lower()
                
                # Check for order history navigation
                if any(phrase in reasoning for phrase in ['order history', 'order', 'account']):
                    if result.get('success'):
                        order_history_accessed = True
            
            # Check HTML for order in history
            step_files = sorted(self.find_step_html("step_*"))
            if step_files:
                for step_file in step_files[-30:]:  # Check last 30 steps
                    with open(step_file, 'r', encoding='utf-8') as f:
                        html_content = f.read().lower()
                        
                        # Look for order history page with orders
                        if 'order' in html_content and any(phrase in html_content for phrase in [
                            'wireless mouse',
                            'usb-c cable',
                            'hdmi cable'
                        ]):
                            self.details['order_history'] = 'Order found in history'
                            return True
            
            if order_history_accessed:
                self.issues.append("Order history accessed but order not found")
            else:
                self.issues.append("Order history not accessed")
            
            return False
                
        except Exception as e:
            self.issues.append(f"Error checking order history: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Reviews submitted with correct ratings (3, 5, 4 stars)"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for review submission actions
            review_actions = []
            
            for action in action_history:
                act = action.get('action', {})
                params = act.get('parameters', {})
                result = action.get('result', {})
                reasoning = act.get('reasoning', '').lower()
                
                # Check for review-related actions
                if 'review' in reasoning and result.get('success'):
                    review_actions.append(action)
            
            # Check HTML for review forms or submitted reviews
            step_files = sorted(self.find_step_html("step_*"))
            reviews_found = 0
            ratings_found = []
            
            if step_files:
                for step_file in step_files[-40:]:  # Check last 40 steps
                    with open(step_file, 'r', encoding='utf-8') as f:
                        html_content = f.read().lower()
                        
                        # Look for review submission or review page
                        if 'review' in html_content:
                            # Count stars or ratings
                            rating_matches = re.findall(r'(\d)\s*star', html_content)
                            if rating_matches:
                                ratings_found.extend([int(r) for r in rating_matches])
                            
                            # Check for review form
                            if 'submit review' in html_content or 'write a review' in html_content:
                                reviews_found += 1
            
            # Check if reviews were submitted
            if len(review_actions) >= 3:
                self.details['reviews'] = f'Found {len(review_actions)} review-related actions'
                return True
            
            # Alternative: check if ratings 3, 5, 4 are present
            if 3 in ratings_found and 5 in ratings_found and 4 in ratings_found:
                self.details['reviews'] = f'Found ratings: {ratings_found}'
                return True
            
            self.issues.append(f"Reviews not submitted (found {len(review_actions)} review actions)")
            return False
                
        except Exception as e:
            self.issues.append(f"Error checking reviews: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_logged_in'] = self.check_checkpoint_1()
            self.checkpoints['cp2_items_added_to_cart'] = self.check_checkpoint_2()
            self.checkpoints['cp3_checkout_completed'] = self.check_checkpoint_3()
            self.checkpoints['cp4_order_found_in_history'] = self.check_checkpoint_4()
            self.checkpoints['cp5_reviews_submitted'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 8,
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
                'query_id': 8,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
