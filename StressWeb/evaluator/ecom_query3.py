#!/usr/bin/env python3
# Evaluator for ecom Query 3

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
        self.in_product_title = False
        self.in_price = False
        self.in_quantity_input = False
        self.in_cart_badge = False
        self.in_breadcrumb = False
        
        self.product_title = None
        self.product_price = None
        self.quantity_value = None
        self.cart_count = None
        self.breadcrumb_text = []
        self.current_url = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for product title
        if tag == 'h1' and attrs_dict.get('class') == 'product-title':
            self.in_product_title = True
        
        # Check for price
        if tag == 'span' and attrs_dict.get('class') == 'price':
            self.in_price = True
        
        # Check for quantity input
        if tag == 'input' and attrs_dict.get('type') == 'number':
            value = attrs_dict.get('value')
            if value:
                self.quantity_value = value
        
        # Check for cart badge
        if tag == 'span' and attrs_dict.get('class') == 'cart-badge':
            self.in_cart_badge = True
        
        # Check for breadcrumb
        if tag == 'div' and attrs_dict.get('class') == 'breadcrumb':
            self.in_breadcrumb = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_product_title:
            self.product_title = data
            self.in_product_title = False
        
        if self.in_price:
            self.product_price = data
            self.in_price = False
        
        if self.in_cart_badge:
            self.cart_count = data
            self.in_cart_badge = False
        
        if self.in_breadcrumb:
            self.breadcrumb_text.append(data)

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_breadcrumb:
            self.in_breadcrumb = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_electronics": False,
            "cp2_found_action_camera_4k": False,
            "cp3_correct_price_199_99": False,
            "cp4_quantity_set_to_2": False,
            "cp5_added_to_cart": False,
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
        """Checkpoint 1: Navigated to Electronics category"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any step navigated to electronics category
            for entry in trajectory:
                page_info = entry.get('page_info', {})
                url = page_info.get('url', '')
                if '/category/electronics' in url:
                    self.details['electronics_url'] = url
                    return True
            
            self.issues.append("Did not navigate to Electronics category")
            return False
        except Exception as e:
            self.issues.append(f"Error checking electronics navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Found the product Action Camera 4K"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)

            if parser.product_title and 'Action Camera 4K' in parser.product_title:
                self.details['product_title'] = parser.product_title
                return True
            
            # Also check trajectory for product page navigation
            trajectory = self.get_trajectory()
            for entry in trajectory:
                page_info = entry.get('page_info', {})
                url = page_info.get('url', '')
                if '/product/' in url:
                    # Check if breadcrumb or title contains Action Camera 4K
                    if 'Action Camera 4K' in html_content:
                        self.details['product_found_in_html'] = True
                        return True
            
            self.issues.append("Product 'Action Camera 4K' not found on final page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking product: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Correct price $199.99"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)

            if parser.product_price and '$199.99' in parser.product_price:
                self.details['product_price'] = parser.product_price
                return True
            
            # Fallback: search in HTML content
            if '$199.99' in html_content and 'Action Camera 4K' in html_content:
                self.details['price_found_in_html'] = True
                return True
            
            self.issues.append(f"Price not $199.99 (found: {parser.product_price})")
            return False
        except Exception as e:
            self.issues.append(f"Error checking price: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Quantity set to 2"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)

            if parser.quantity_value == '2':
                self.details['quantity'] = parser.quantity_value
                return True
            
            # Fallback: regex search for quantity input
            quantity_match = re.search(r'<input[^>]*type="number"[^>]*value="(\d+)"', html_content)
            if quantity_match:
                quantity = quantity_match.group(1)
                self.details['quantity'] = quantity
                if quantity == '2':
                    return True
                else:
                    self.issues.append(f"Quantity is {quantity}, expected 2")
                    return False
            
            self.issues.append("Could not find quantity input value")
            return False
        except Exception as e:
            self.issues.append(f"Error checking quantity: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Added to cart (cart badge shows 2 items)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)

            # Check cart badge count
            if parser.cart_count:
                cart_count_int = int(parser.cart_count)
                self.details['cart_count'] = cart_count_int
                
                # Cart should have at least 2 items (2 units of Action Camera 4K)
                if cart_count_int >= 2:
                    return True
                else:
                    self.issues.append(f"Cart count is {cart_count_int}, expected at least 2")
                    return False
            
            # Fallback: regex search for cart badge
            cart_badge_match = re.search(r'cart-badge">(\d+)<', html_content)
            if cart_badge_match:
                cart_count = int(cart_badge_match.group(1))
                self.details['cart_count'] = cart_count
                if cart_count >= 2:
                    return True
                else:
                    self.issues.append(f"Cart count is {cart_count}, expected at least 2")
                    return False
            
            self.issues.append("Could not find cart badge count")
            return False
        except Exception as e:
            self.issues.append(f"Error checking cart: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_electronics'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_action_camera_4k'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_price_199_99'] = self.check_checkpoint_3()
            self.checkpoints['cp4_quantity_set_to_2'] = self.check_checkpoint_4()
            self.checkpoints['cp5_added_to_cart'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 3,
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
                'query_id': 3,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
