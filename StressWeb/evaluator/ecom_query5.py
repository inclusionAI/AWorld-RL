#!/usr/bin/env python3
# Evaluator for ecom Query 5

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class CheckoutHTMLParser(HTMLParser):
    """Parse HTML to extract checkout and cart information."""

    def __init__(self):
        super().__init__()
        self.in_cart_item = False
        self.in_item_name = False
        self.in_item_quantity = False
        self.in_checkout = False
        self.in_order_summary = False
        self.cart_items = []
        self.current_item = {}
        self.current_tag = None
        self.current_attrs = []
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Check for cart item
        if tag == 'div':
            classes = self.current_attrs.get('class', '')
            if 'cart-item' in classes:
                self.in_cart_item = True
                self.current_item = {}
            elif 'summary-item' in classes:
                self.in_cart_item = True
                self.current_item = {}
                
        # Check for item name
        if self.in_cart_item and tag == 'a':
            classes = self.current_attrs.get('class', '')
            if 'item-name' in classes:
                self.in_item_name = True
                
        # Check for item name in checkout
        if self.in_cart_item and tag == 'p':
            classes = self.current_attrs.get('class', '')
            if 'summary-item-name' in classes:
                self.in_item_name = True
                
        # Check for quantity
        if self.in_cart_item and tag == 'input':
            input_type = self.current_attrs.get('type', '')
            if input_type == 'number':
                value = self.current_attrs.get('value', '0')
                self.current_item['quantity'] = int(value)
                
        # Check for quantity in checkout summary
        if self.in_cart_item and tag == 'p':
            classes = self.current_attrs.get('class', '')
            if 'summary-item-qty' in classes:
                self.in_item_quantity = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture item name
        if self.in_item_name:
            self.current_item['name'] = data
            self.in_item_name = False
            
        # Capture quantity from text like "Qty: 2"
        if self.in_item_quantity:
            match = re.search(r'Qty:\s*(\d+)', data)
            if match:
                self.current_item['quantity'] = int(match.group(1))
            self.in_item_quantity = False

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_cart_item:
            if 'name' in self.current_item:
                self.cart_items.append(self.current_item.copy())
            self.in_cart_item = False
            self.current_item = {}


class AccountPageParser(HTMLParser):
    """Parse account page to detect successful navigation."""
    
    def __init__(self):
        super().__init__()
        self.on_account_page = False
        self.cart_badge_count = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for account page indicators
        if tag == 'div':
            classes = attrs_dict.get('class', '')
            if 'account-page' in classes or 'account-container' in classes:
                self.on_account_page = True
                
        # Check cart badge (should be 0 after successful order)
        if tag == 'span':
            classes = attrs_dict.get('class', '')
            if 'cart-badge' in classes:
                self.cart_badge_count = None
                self.in_cart_badge = True
                
    def handle_data(self, data):
        if hasattr(self, 'in_cart_badge') and self.in_cart_badge:
            try:
                self.cart_badge_count = int(data.strip())
            except ValueError:
                pass
            self.in_cart_badge = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_logged_in": False,
            "cp2_searched_laptop": False,
            "cp3_added_correct_product": False,
            "cp4_correct_quantity": False,
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

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Successfully logged in with correct credentials"""
        trajectory = self.get_trajectory()
        
        # Check if login was successful by looking for navigation to account page
        for entry in trajectory:
            page_info = entry.get('page_info', {})
            url = page_info.get('url', '')
            
            # After login, should navigate to account page
            if '/account' in url:
                self.details['login_successful'] = True
                return True
                
        # Check if email was filled correctly
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'FILL':
                params = action.get('parameters', {})
                if params.get('text') == '12345@mail.com':
                    # Found correct email, but check if login succeeded
                    pass
                    
        self.issues.append("Login with 12345@mail.com not verified or failed")
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Searched for 'Laptop'"""
        trajectory = self.get_trajectory()
        
        # Check for search action
        searched = False
        for entry in trajectory:
            action = entry.get('action', {})
            
            # Check TYPE action for "Laptop"
            if action.get('action_type') == 'TYPE':
                text = action.get('parameters', {}).get('text', '')
                if 'laptop' in text.lower():
                    searched = True
                    
            # Check if navigated to search results page
            page_info = entry.get('page_info', {})
            url = page_info.get('url', '')
            if '/search' in url and 'laptop' in url.lower():
                self.details['search_executed'] = True
                return True
                
        if not searched:
            self.issues.append("Did not search for 'Laptop'")
            return False
            
        return searched

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Added 'Laptop Sleeve 15-inch' to cart"""
        trajectory = self.get_trajectory()
        
        # Check if the correct product was accessed
        for entry in trajectory:
            action = entry.get('action', {})
            
            # Check if clicked on Laptop Sleeve 15-inch
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if 'Laptop Sleeve 15-inch' in selector:
                    self.details['correct_product_clicked'] = True
                    
            # Check page URL for product
            page_info = entry.get('page_info', {})
            url = page_info.get('url', '')
            if '/product/' in url:
                # Need to verify in cart
                pass
                
        # Check cart contents in step files
        cart_files = self.find_step_html("step_14_*")
        if cart_files:
            with open(cart_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            parser = CheckoutHTMLParser()
            parser.feed(html_content)
            
            for item in parser.cart_items:
                if 'Laptop Sleeve 15-inch' in item.get('name', ''):
                    self.details['product_in_cart'] = item['name']
                    return True
                    
        self.issues.append("Laptop Sleeve 15-inch not found in cart")
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Added 2 units of the product"""
        # Check cart contents for quantity
        cart_files = self.find_step_html("step_14_*")
        if cart_files:
            with open(cart_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            parser = CheckoutHTMLParser()
            parser.feed(html_content)
            
            for item in parser.cart_items:
                if 'Laptop Sleeve 15-inch' in item.get('name', ''):
                    quantity = item.get('quantity', 0)
                    self.details['cart_quantity'] = quantity
                    if quantity == 2:
                        return True
                    else:
                        self.issues.append(f"Incorrect quantity: expected 2, found {quantity}")
                        return False
                        
        # Also check checkout page
        checkout_files = self.find_step_html("step_15_*")
        if not checkout_files:
            checkout_files = self.find_step_html("step_16_*")
            
        if checkout_files:
            with open(checkout_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            parser = CheckoutHTMLParser()
            parser.feed(html_content)
            
            for item in parser.cart_items:
                if 'Laptop Sleeve 15-inch' in item.get('name', ''):
                    quantity = item.get('quantity', 0)
                    self.details['checkout_quantity'] = quantity
                    if quantity == 2:
                        return True
                    else:
                        self.issues.append(f"Incorrect quantity at checkout: expected 2, found {quantity}")
                        return False
                        
        self.issues.append("Could not verify quantity of 2 units")
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Successfully placed order using default address/payment"""
        trajectory = self.get_trajectory()
        
        # Check if "Place Order" was clicked
        place_order_clicked = False
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if 'Place Order' in selector:
                    place_order_clicked = True
                    # Check if it was successful
                    result = action.get('result', {})
                    if result and entry.get('result', {}).get('success'):
                        # Check next page
                        next_url = entry.get('page_info', {}).get('url', '')
                        if '/account' in next_url:
                            self.details['order_placed_successfully'] = True
                            
        if not place_order_clicked:
            self.issues.append("Place Order button was not clicked")
            return False
            
        # Check final HTML - should be on account page with empty cart
        final_html = self.find_final_html()
        if final_html:
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            parser = AccountPageParser()
            parser.feed(html_content)
            
            if parser.on_account_page:
                self.details['final_page'] = 'account'
                # Cart should be empty (0) after successful order
                if parser.cart_badge_count == 0:
                    self.details['cart_emptied'] = True
                    return True
                elif parser.cart_badge_count is not None:
                    self.details['cart_badge'] = parser.cart_badge_count
                    
        # Check trajectory for final state
        if trajectory:
            last_entry = trajectory[-1]
            if last_entry.get('action', {}).get('action_type') == 'DONE':
                final_url = last_entry.get('page_info', {}).get('url', '')
                if '/account' in final_url:
                    self.details['ended_on_account_page'] = True
                    return True
                    
        self.issues.append("Order placement not confirmed - did not return to account page with empty cart")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_logged_in'] = self.check_checkpoint_1()
            self.checkpoints['cp2_searched_laptop'] = self.check_checkpoint_2()
            self.checkpoints['cp3_added_correct_product'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_quantity'] = self.check_checkpoint_4()
            self.checkpoints['cp5_order_placed'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 5,
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
                'query_id': 5,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
