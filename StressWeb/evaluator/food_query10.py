#!/usr/bin/env python3
# Evaluator for food Query 10

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
        self.in_order_card = False
        self.in_order_title = False
        self.in_status_badge = False
        self.in_order_items = False
        self.in_item_name = False
        self.in_price_row = False
        self.in_scheduled_info = False
        self.current_order_id = None
        self.current_status = None
        self.current_item = None
        self.current_price_label = None
        self.current_price_value = None
        self.scheduled_delivery = None
        
        self.orders = {}
        self.current_order_data = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get('class', '').split()
        
        if 'order-card' in classes:
            self.in_order_card = True
            self.current_order_data = {
                'items': [],
                'price_breakdown': {},
                'scheduled_delivery': None
            }
        
        if self.in_order_card:
            if tag == 'h3' and not self.in_order_title:
                self.in_order_title = True
            elif 'status-badge' in classes:
                self.in_status_badge = True
            elif 'order-items-section' in classes:
                self.in_order_items = True
            elif 'item-name' in classes:
                self.in_item_name = True
            elif 'price-row' in classes:
                self.in_price_row = True
                self.current_price_label = None
                self.current_price_value = None
            elif 'scheduled-delivery-info' in classes:
                self.in_scheduled_info = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_order_title and data.startswith('Order #'):
            self.current_order_id = data
            self.in_order_title = False
        
        if self.in_status_badge:
            self.current_status = data
            self.in_status_badge = False
        
        if self.in_item_name:
            self.current_item = data
        
        if self.in_price_row:
            if self.current_price_label is None:
                self.current_price_label = data
            else:
                self.current_price_value = data
        
        if self.in_scheduled_info and 'Scheduled for' in data:
            self.scheduled_delivery = data.replace('Scheduled for ', '')

    def handle_endtag(self, tag):
        if self.in_item_name and tag in ['span', 'div']:
            if self.current_item:
                self.current_order_data['items'].append(self.current_item)
                self.current_item = None
            self.in_item_name = False
        
        if self.in_price_row and tag == 'div':
            if self.current_price_label and self.current_price_value:
                self.current_order_data['price_breakdown'][self.current_price_label] = self.current_price_value
            self.in_price_row = False
        
        if self.in_scheduled_info and tag == 'div':
            if self.scheduled_delivery:
                self.current_order_data['scheduled_delivery'] = self.scheduled_delivery
                self.scheduled_delivery = None
            self.in_scheduled_info = False
        
        if self.in_order_card and tag == 'div' and 'order-card' in str(tag):
            if self.current_order_id:
                self.current_order_data['status'] = self.current_status
                self.orders[self.current_order_id] = self.current_order_data.copy()
                self.current_order_id = None
                self.current_status = None
            self.in_order_card = False


class CheckoutPageParser(HTMLParser):
    """Parse checkout page to extract delivery and tip information."""
    
    def __init__(self):
        super().__init__()
        self.in_address_card = False
        self.address_card_selected = False
        self.in_address_label = False
        self.in_scheduled_preview = False
        self.in_custom_tip_preview = False
        self.current_address_label = None
        
        self.selected_address = None
        self.scheduled_delivery_info = None
        self.tip_amount = None
        self.delivery_date = None
        self.delivery_time = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get('class', '').split()
        
        if 'address-card' in classes:
            self.in_address_card = True
            self.address_card_selected = 'selected' in classes
        
        if self.in_address_card and tag == 'h3' and 'address-label' in classes:
            self.in_address_label = True
        
        if 'scheduled-preview' in classes:
            self.in_scheduled_preview = True
        
        if 'custom-tip-preview' in classes:
            self.in_custom_tip_preview = True
        
        if tag == 'input':
            input_type = attrs_dict.get('type', '')
            value = attrs_dict.get('value', '')
            if input_type == 'date' and value:
                self.delivery_date = value
            elif input_type == 'time' and value:
                self.delivery_time = value

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_address_label and self.address_card_selected:
            self.current_address_label = data
            self.in_address_label = False
        
        if self.in_scheduled_preview and 'delivered on' in data.lower():
            self.scheduled_delivery_info = data
        
        if self.in_custom_tip_preview and 'Custom tip:' in data:
            # Extract tip amount from "Custom tip: $6.70"
            match = re.search(r'\$(\d+\.?\d*)', data)
            if match:
                self.tip_amount = match.group(1)

    def handle_endtag(self, tag):
        if self.in_address_card and tag == 'div':
            if self.current_address_label and self.address_card_selected:
                self.selected_address = self.current_address_label
                self.current_address_label = None
            self.in_address_card = False
            self.address_card_selected = False
        
        if self.in_scheduled_preview and tag in ['div', 'span']:
            self.in_scheduled_preview = False
        
        if self.in_custom_tip_preview and tag == 'p':
            self.in_custom_tip_preview = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_two_cheapest_burgers_added": False,
            "cp2_delivery_to_parents_house": False,
            "cp3_scheduled_jan3_7pm": False,
            "cp4_tip_6_70_added": False,
            "cp5_order_placed_successfully": False,
            "cp6_order_cancelled": False,
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
        """Checkpoint 1: Two cheapest burgers (Classic Cheeseburger @ $10.99 each) added to cart"""
        try:
            # Check step 27 (when cart was finalized before checkout)
            cart_steps = self.find_step_html("step_2[67]*")
            if not cart_steps:
                self.issues.append("Cannot find cart step HTML files")
                return False
            
            for step_file in cart_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Check if cart contains 2 Classic Cheeseburgers
                classic_cheeseburger_count = html_content.count('Classic Cheeseburger')
                
                # Look for cart with exactly 2 items
                if '2 items from cart' in html_content or classic_cheeseburger_count >= 2:
                    self.details['burgers_added'] = "2x Classic Cheeseburger ($10.99 each)"
                    return True
            
            self.issues.append("Did not find 2 cheapest burgers in cart")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking burgers added: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Delivery address set to Parent's House"""
        try:
            # Check checkout step (step 62) for selected address
            checkout_steps = self.find_step_html("step_6[0-2]*")
            if not checkout_steps:
                self.issues.append("Cannot find checkout step HTML files")
                return False
            
            for step_file in checkout_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                parser = CheckoutPageParser()
                parser.feed(html_content)
                
                if parser.selected_address and "Parent" in parser.selected_address:
                    self.details['delivery_address'] = parser.selected_address
                    return True
            
            self.issues.append("Parent's House not selected as delivery address")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking delivery address: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Scheduled delivery on January 3, 2026 at 7:00 PM"""
        try:
            # Check checkout step for scheduled delivery
            checkout_steps = self.find_step_html("step_6[0-2]*")
            if not checkout_steps:
                return False
            
            for step_file in checkout_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                parser = CheckoutPageParser()
                parser.feed(html_content)
                
                # Check if date and time match requirements
                if parser.delivery_date == "2026-01-03" and parser.delivery_time == "19:00":
                    self.details['scheduled_delivery'] = "2026/01/03 19:00 (7:00 PM)"
                    return True
                
                # Also check in order confirmation and final state
                if '2026/01/03 7:00 PM' in html_content or '2026/01/03 19:00' in html_content:
                    self.details['scheduled_delivery'] = "2026/01/03 7:00 PM"
                    return True
            
            self.issues.append("Delivery not scheduled for January 3, 2026 at 7:00 PM")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking scheduled delivery: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Tip amount of $6.70 added"""
        try:
            # Check checkout step for tip
            checkout_steps = self.find_step_html("step_6[0-2]*")
            if not checkout_steps:
                return False
            
            for step_file in checkout_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                parser = CheckoutPageParser()
                parser.feed(html_content)
                
                if parser.tip_amount == "6.70":
                    self.details['tip_amount'] = "$6.70"
                    return True
                
                # Also check in HTML for tip display
                if 'Custom tip: $6.70' in html_content or 'Tip (6.7%)' in html_content:
                    self.details['tip_amount'] = "$6.70"
                    return True
            
            self.issues.append("Tip amount of $6.70 not found")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking tip: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Order placed successfully (Order #1002 exists)"""
        try:
            # Check order confirmation page (step 63)
            confirmation_steps = self.find_step_html("step_63*")
            if not confirmation_steps:
                self.issues.append("Cannot find order confirmation step")
                return False
            
            for step_file in confirmation_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                if 'Order #1002' in html_content and 'Order Confirmed!' in html_content:
                    self.details['order_placed'] = "Order #1002 confirmed"
                    return True
            
            self.issues.append("Order was not placed successfully")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking order placement: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Order cancelled in order history"""
        try:
            # Check final state for order status
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Cannot find final HTML state")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse orders in final state
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            # Check if Order #1002 exists and is cancelled
            if 'Order #1002' in parser.orders:
                order_1002 = parser.orders['Order #1002']
                status = order_1002.get('status', '')
                
                # Check if status indicates cancellation
                if 'Cancelled' in status or 'Canceled' in status:
                    self.details['order_status'] = "Order #1002 cancelled"
                    return True
                elif 'Scheduled' in status:
                    self.issues.append("Order #1002 is still scheduled (not cancelled)")
                    self.details['order_status'] = f"Order #1002 status: {status}"
                    return False
                else:
                    self.issues.append(f"Order #1002 has unexpected status: {status}")
                    return False
            
            # Alternative: Check if cancel button is no longer present (order was cancelled)
            # If order is in history but button says "Cancel Order", it's not cancelled yet
            if 'Order #1002' in html_content:
                if 'Cancel Order' in html_content and 'Scheduled' in html_content:
                    self.issues.append("Order #1002 still has 'Cancel Order' button (not cancelled)")
                    return False
            
            self.issues.append("Order #1002 not found in order history")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking order cancellation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_two_cheapest_burgers_added'] = self.check_checkpoint_1()
            self.checkpoints['cp2_delivery_to_parents_house'] = self.check_checkpoint_2()
            self.checkpoints['cp3_scheduled_jan3_7pm'] = self.check_checkpoint_3()
            self.checkpoints['cp4_tip_6_70_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_order_placed_successfully'] = self.check_checkpoint_5()
            self.checkpoints['cp6_order_cancelled'] = self.check_checkpoint_6()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python food_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
