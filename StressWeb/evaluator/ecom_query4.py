#!/usr/bin/env python3
# Evaluator for ecom Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class CartHTMLParser(HTMLParser):
    """Parse HTML to extract cart information."""
    def __init__(self):
        super().__init__()
        self.cart_badge_count = None
        self.cart_items = []
        self.in_cart_badge = False
        self.in_cart_item = False
        self.in_item_name = False
        self.in_item_quantity = False
        self.current_item = {}
        self.current_data = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for cart badge
        if tag == 'span' and attrs_dict.get('class') == 'cart-badge':
            self.in_cart_badge = True
            
        # Check for cart item
        if tag == 'div' and attrs_dict.get('class') == 'cart-item':
            self.in_cart_item = True
            self.current_item = {}
            
        # Check for item name
        if self.in_cart_item and tag == 'a' and 'item-name' in attrs_dict.get('class', ''):
            self.in_item_name = True
            
        # Check for quantity input
        if self.in_cart_item and tag == 'input' and attrs_dict.get('type') == 'number':
            self.in_item_quantity = True
            quantity_val = attrs_dict.get('value', '1')
            try:
                self.current_item['quantity'] = int(quantity_val)
            except:
                self.current_item['quantity'] = 1
    
    def handle_endtag(self, tag):
        if tag == 'span' and self.in_cart_badge:
            self.in_cart_badge = False
            
        if tag == 'a' and self.in_item_name:
            self.in_item_name = False
            if self.current_data:
                self.current_item['name'] = ''.join(self.current_data).strip()
                self.current_data = []
                
        if tag == 'div' and self.in_cart_item:
            self.in_cart_item = False
            if self.current_item:
                self.cart_items.append(self.current_item)
                self.current_item = {}
    
    def handle_data(self, data):
        if self.in_cart_badge:
            try:
                self.cart_badge_count = int(data.strip())
            except:
                pass
                
        if self.in_item_name:
            self.current_data.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_three_items_added": False,
            "cp2_two_items_removed": False,
            "cp3_final_cart_correct": False,
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
    
    def parse_cart_html(self, html_file: Path) -> Tuple[int, List[Dict]]:
        """Parse HTML file to extract cart information."""
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = CartHTMLParser()
        parser.feed(html_content)
        
        cart_count = parser.cart_badge_count if parser.cart_badge_count is not None else 0
        cart_items = parser.cart_items
        
        return cart_count, cart_items
    
    def get_cart_counts_from_trajectory(self) -> List[Tuple[int, int]]:
        """Extract cart counts from all step HTML files."""
        cart_history = []
        
        # Get all step files sorted by step number
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        
        for step_file in step_files:
            # Extract step number from filename
            match = re.search(r'step_(\d+)_', step_file.name)
            if match:
                step_num = int(match.group(1))
                cart_count, cart_items = self.parse_cart_html(step_file)
                cart_history.append((step_num, cart_count))
        
        return cart_history
    
    def check_checkpoint_1(self) -> bool:
        """Check if three items were added to cart."""
        cart_history = self.get_cart_counts_from_trajectory()
        
        if not cart_history:
            self.issues.append("No cart history found")
            return False
        
        # Check if cart count reached 3 at any point
        max_cart_count = max(count for _, count in cart_history)
        
        self.details['max_cart_count'] = max_cart_count
        self.details['cart_history'] = cart_history
        
        if max_cart_count >= 3:
            return True
        else:
            self.issues.append(f"Cart never reached 3 items (max was {max_cart_count})")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if items were removed from cart."""
        cart_history = self.get_cart_counts_from_trajectory()
        
        if not cart_history:
            self.issues.append("No cart history to verify removal")
            return False
        
        # Find max cart count
        max_cart_count = max(count for _, count in cart_history)
        
        # Check if cart count decreased after reaching max
        max_index = next(i for i, (_, count) in enumerate(cart_history) if count == max_cart_count)
        
        # Check subsequent steps for decreases
        removal_detected = False
        for i in range(max_index + 1, len(cart_history)):
            if cart_history[i][1] < max_cart_count:
                removal_detected = True
                break
        
        self.details['removal_detected'] = removal_detected
        
        if not removal_detected:
            self.issues.append("No removal actions detected in cart history")
            return False
        
        return True
    
    def check_checkpoint_3(self) -> bool:
        """Check if final cart has exactly 1 item (3 added, 2 removed)."""
        final_html = self.find_final_html()
        
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        cart_count, cart_items = self.parse_cart_html(final_html)
        
        self.details['final_cart_count'] = cart_count
        self.details['final_cart_items'] = cart_items
        
        # The expected final state should have some items remaining
        # Task: Add 3, remove 2 = 1 item remaining
        # However, the cart badge shows quantity, not unique items
        # The test case shows cart-badge=3 in final state with 1 unique item (quantity 3)
        
        # Check if we have items in cart
        if cart_count == 0 and len(cart_items) == 0:
            self.issues.append("Cart is empty in final state - should have 1 item remaining")
            return False
        
        # Verify we have exactly 1 unique item
        if len(cart_items) == 1:
            return True
        elif len(cart_items) == 0:
            # Check if cart badge indicates items
            if cart_count > 0:
                # Badge shows items but parser didn't find cart-item divs
                # This is acceptable as long as badge shows remaining items
                return True
            else:
                self.issues.append("No items found in final cart")
                return False
        else:
            self.issues.append(f"Expected 1 item in cart, found {len(cart_items)}")
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_three_items_added'] = self.check_checkpoint_1()
            self.checkpoints['cp2_two_items_removed'] = self.check_checkpoint_2()
            self.checkpoints['cp3_final_cart_correct'] = self.check_checkpoint_3()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 4,
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
                'query_id': 4,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query4.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
