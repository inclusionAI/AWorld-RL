#!/usr/bin/env python3
# Evaluator for ecom Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ProductHTMLParser(HTMLParser):
    """Parse HTML to extract product information from Electronics category."""
    def __init__(self):
        super().__init__()
        self.products = []
        self.current_product = {}
        self.in_product_name = False
        self.in_product_price = False
        self.in_sort_select = False
        self.current_option_value = None
        self.selected_sort = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'h3' and attrs_dict.get('class') == 'product-name':
            self.in_product_name = True
            
        elif tag == 'p' and attrs_dict.get('class') == 'product-price':
            self.in_product_price = True
            
        elif tag == 'select' and 'sort-select' in attrs_dict.get('class', ''):
            self.in_sort_select = True
            
        elif tag == 'option' and self.in_sort_select:
            self.current_option_value = attrs_dict.get('value')
            if 'selected' in attrs_dict:
                self.selected_sort = self.current_option_value
        
    def handle_endtag(self, tag):
        if tag == 'h3' and self.in_product_name:
            self.in_product_name = False
            
        elif tag == 'p' and self.in_product_price:
            self.in_product_price = False
            if 'name' in self.current_product and 'price' in self.current_product:
                self.products.append(self.current_product)
                self.current_product = {}
                
        elif tag == 'select':
            self.in_sort_select = False
            
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_product_name:
            self.current_product['name'] = data
            
        elif self.in_product_price:
            # Extract price value (e.g., "$399.99" -> 399.99)
            price_match = re.search(r'\$?([\d.]+)', data)
            if price_match:
                self.current_product['price'] = float(price_match.group(1))


class CartHTMLParser(HTMLParser):
    """Parse HTML to extract cart information."""
    def __init__(self):
        super().__init__()
        self.cart_count = 0
        self.cart_items = []
        self.in_cart_badge = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'span' and attrs_dict.get('class') == 'cart-badge':
            self.in_cart_badge = True
            
    def handle_endtag(self, tag):
        if tag == 'span' and self.in_cart_badge:
            self.in_cart_badge = False
            
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_cart_badge:
            try:
                self.cart_count = int(data)
            except ValueError:
                pass


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_navigated_to_electronics": False,
            "cp2_sorted_by_price_high_to_low": False,
            "cp3_second_product_added": False,
            "cp4_last_product_added": False,
            "cp5_cart_has_two_items": False,
        }
        
        self.issues = []
        self.details = {}
    
    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        return step_files[-1] if step_files else None
    
    def find_step_html(self, step_pattern: str) -> List[Path]:
        """Find intermediate step HTML files."""
        return sorted(self.result_dir.glob(f"{step_pattern}_raw.html"))
    
    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data."""
        if not self.traj_file.exists():
            return []
        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def parse_products_from_html(self, html_path: Path) -> Tuple[List[Dict], Optional[str]]:
        """Parse products and sort order from HTML."""
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = ProductHTMLParser()
        parser.feed(content)
        
        return parser.products, parser.selected_sort
    
    def parse_cart_from_html(self, html_path: Path) -> int:
        """Parse cart count from HTML."""
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = CartHTMLParser()
        parser.feed(content)
        
        return parser.cart_count
    
    def check_product_in_html(self, html_path: Path, product_name: str) -> bool:
        """Check if a specific product name appears in the HTML."""
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return product_name in content
    
    def check_checkpoint_1(self) -> bool:
        """Check if navigated to Electronics category."""
        # Look for step 1 or later to see if we're in Electronics category
        step_files = self.find_step_html("step_*")
        if not step_files:
            self.issues.append("No step files found")
            return False
        
        # Check if any step shows Electronics category
        for step_file in step_files[:3]:  # Check first few steps
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<span>Electronics</span>' in content or 'category/electronics' in content:
                    self.details['electronics_category_found'] = True
                    return True
        
        self.issues.append("Did not navigate to Electronics category")
        return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if products were sorted by price high to low."""
        # Find a step after sorting (step 2 or later)
        step_files = self.find_step_html("step_*")
        if len(step_files) < 2:
            self.issues.append("Not enough steps to verify sorting")
            return False
        
        # Check step 2 onwards for the sort selection
        for step_file in step_files[1:4]:  # Check steps 2-4
            products, selected_sort = self.parse_products_from_html(step_file)
            
            if selected_sort == 'price-high':
                self.details['sort_applied'] = 'price-high'
                
                # Verify products are actually sorted by price (high to low)
                if len(products) >= 2:
                    prices = [p['price'] for p in products if 'price' in p]
                    if len(prices) >= 2:
                        # Check if prices are in descending order
                        is_sorted = all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
                        if is_sorted:
                            self.details['products_sorted_correctly'] = True
                            self.details['first_product'] = products[0]['name'] if products else None
                            self.details['first_price'] = products[0]['price'] if products else None
                            return True
                        else:
                            self.issues.append("Products not sorted correctly by price")
                            return False
        
        self.issues.append("Price sorting (high to low) not applied")
        return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if the second product was added to cart."""
        # The second product in the sorted list should be "Monitor 27-inch 4K" at $399.99
        # Check trajectory and final HTML
        
        step_files = self.find_step_html("step_*")
        if len(step_files) < 2:
            self.issues.append("Not enough steps to verify second product")
            return False
        
        # Get products after sorting (step 2)
        products, _ = self.parse_products_from_html(step_files[1])
        
        if len(products) < 2:
            self.issues.append("Could not identify second product in sorted list")
            return False
        
        second_product = products[1]
        self.details['second_product'] = second_product['name']
        self.details['second_product_price'] = second_product['price']
        
        # Check if this product appears in final HTML (indicating it was added to cart)
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        # Look for the product name in the final HTML
        if self.check_product_in_html(final_html, second_product['name']):
            self.details['second_product_in_cart'] = True
            return True
        
        self.issues.append(f"Second product '{second_product['name']}' not found in cart")
        return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if the last product was added to cart."""
        # The last product should be "USB-C Cable 6ft" at $12.99
        
        step_files = self.find_step_html("step_*")
        if len(step_files) < 2:
            self.issues.append("Not enough steps to verify last product")
            return False
        
        # Get products after sorting (step 2)
        products, _ = self.parse_products_from_html(step_files[1])
        
        if len(products) < 1:
            self.issues.append("Could not identify last product in sorted list")
            return False
        
        last_product = products[-1]
        self.details['last_product'] = last_product['name']
        self.details['last_product_price'] = last_product['price']
        
        # Check if this product appears in final HTML
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        if self.check_product_in_html(final_html, last_product['name']):
            self.details['last_product_in_cart'] = True
            return True
        
        self.issues.append(f"Last product '{last_product['name']}' not found in cart")
        return False
    
    def check_checkpoint_5(self) -> bool:
        """Check if cart has exactly 2 items."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        cart_count = self.parse_cart_from_html(final_html)
        self.details['final_cart_count'] = cart_count
        
        if cart_count == 2:
            return True
        
        self.issues.append(f"Cart has {cart_count} items, expected 2")
        return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_navigated_to_electronics'] = self.check_checkpoint_1()
            self.checkpoints['cp2_sorted_by_price_high_to_low'] = self.check_checkpoint_2()
            self.checkpoints['cp3_second_product_added'] = self.check_checkpoint_3()
            self.checkpoints['cp4_last_product_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_cart_has_two_items'] = self.check_checkpoint_5()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 9,
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
                'query_id': 9,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query9.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
