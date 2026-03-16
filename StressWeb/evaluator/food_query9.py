#!/usr/bin/env python3
# Evaluator for food Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class OrderHistoryParser(HTMLParser):
    """Parse HTML to extract order history information."""

    def __init__(self):
        super().__init__()
        self.orders = []
        self.in_order_card = False
        self.current_order = {}
        self.current_tag = None
        self.current_class = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_class = attrs_dict.get('class', '').split()
        
        if 'order-card' in self.current_class:
            self.in_order_card = True
            self.current_order = {}

    def handle_data(self, data):
        if self.in_order_card:
            text = data.strip()
            if text:
                # Capture order details
                if 'order-id' in self.current_class or '#' in text:
                    self.current_order['order_id'] = text
                elif '$' in text and 'total' not in self.current_order:
                    self.current_order['total'] = text

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_order_card:
            if self.current_order:
                self.orders.append(self.current_order.copy())
            self.in_order_card = False
            self.current_order = {}


class CheckoutParser(HTMLParser):
    """Parse checkout page to extract promo code and payment details."""

    def __init__(self):
        super().__init__()
        self.promo_code_applied = False
        self.promo_code_value = None
        self.discount_amount = None
        self.in_promo_section = False
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get('class', '')
        
        # Check for promo code input
        if tag == 'input' and 'promo' in class_name.lower():
            self.promo_code_value = attrs_dict.get('value', '')
            
    def handle_data(self, data):
        text = data.strip().lower()
        self.current_text.append(data.strip())
        
        # Check for promo/discount indicators
        if 'promo' in text or 'discount' in text or 'code applied' in text:
            self.in_promo_section = True
            
        # Check for specific promo codes
        if 'welcome' in text or 'lucky' in text:
            self.promo_code_applied = True
            
        # Capture discount amount
        if self.in_promo_section and '-$' in data:
            self.discount_amount = data.strip()


class ReceiptParser(HTMLParser):
    """Parse receipt/order details page."""

    def __init__(self):
        super().__init__()
        self.has_receipt = False
        self.order_number = None
        self.items = []
        self.total = None
        self.promo_discount = None
        self.current_text = []

    def handle_data(self, data):
        text = data.strip().lower()
        self.current_text.append(data.strip())
        
        if any(keyword in text for keyword in ['receipt', 'order details', 'order summary']):
            self.has_receipt = True
            
        if 'order #' in text or 'order id' in text:
            self.order_number = data.strip()
            
        if 'promo' in text or 'discount' in text:
            # Look for discount in nearby text
            for t in self.current_text[-5:]:
                if '-$' in t:
                    self.promo_discount = t


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_order_history_accessed": False,
            "cp2_order_reordered": False,
            "cp3_promo_code_applied": False,
            "cp4_payment_completed": False,
            "cp5_receipt_checked": False,
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
        """Checkpoint 1: User accessed order history page."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any step navigated to order history
            for entry in trajectory:
                if 'action' in entry:
                    # Check URL patterns
                    if 'url' in str(entry).lower() and 'order' in str(entry).lower():
                        self.details['order_history_accessed'] = True
                        return True
                        
            # Check HTML files for order history page
            all_steps = self.find_step_html("step_*")
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'order-history-page' in content or 'Order History' in content:
                        self.details['order_history_accessed'] = True
                        return True
                        
            self.issues.append("Agent did not access order history page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking order history access: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: User found and reordered an item from history."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if there were orders available to reorder
            all_steps = self.find_step_html("step_*")
            has_orders = False
            
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check if orders were found
                    if 'No orders found' in content or '0 orders shown' in content:
                        continue
                    
                    # Check for order cards or reorder buttons
                    if 'order-card' in content or 'reorder' in content.lower():
                        has_orders = True
                        
            if not has_orders:
                self.issues.append("No orders found in order history to reorder")
                self.details['no_orders_available'] = True
                return False
                
            # Check if reorder action was taken
            for entry in trajectory:
                if 'action' in entry:
                    action_text = json.dumps(entry).lower()
                    if 'reorder' in action_text or 'order again' in action_text:
                        self.details['reorder_attempted'] = True
                        return True
                        
            self.issues.append("Agent did not attempt to reorder")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking reorder: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Promo code was applied during checkout."""
        try:
            # Check all steps for promo code application
            all_steps = self.find_step_html("step_*")
            
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for promo code input and valid codes
                    if ('welcome' in content.lower() or 'lucky' in content.lower()):
                        # Check if discount/promo was applied
                        if 'discount' in content.lower() or 'promo' in content.lower():
                            # Look for discount amount indicator
                            if '-$' in content or 'applied' in content.lower():
                                self.details['promo_code_applied'] = True
                                return True
                                
            # Check trajectory for promo code actions
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action_text = json.dumps(entry).lower()
                if ('welcome' in action_text or 'lucky' in action_text) and 'promo' in action_text:
                    self.details['promo_code_entered'] = True
                    return True
                    
            self.issues.append("Promo code (welcome/lucky) was not applied")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking promo code: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Payment was completed successfully."""
        try:
            # Check for payment completion indicators
            all_steps = self.find_step_html("step_*")
            
            for step_file in all_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Look for order confirmation or success messages
                    if any(indicator in content for indicator in [
                        'order placed',
                        'order confirmed',
                        'payment successful',
                        'thank you for your order',
                        'order-confirmation'
                    ]):
                        self.details['payment_completed'] = True
                        return True
                        
            # Check trajectory for payment actions
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action_text = json.dumps(entry).lower()
                if any(keyword in action_text for keyword in ['place order', 'confirm payment', 'pay now']):
                    # Need to verify success
                    self.details['payment_attempted'] = True
                    
            if 'payment_attempted' not in self.details:
                self.issues.append("Payment was not attempted")
            else:
                self.issues.append("Payment attempted but not confirmed successful")
                
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking payment: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Receipt was checked in order history after purchase."""
        try:
            # Check final state and later steps for receipt viewing
            all_steps = self.find_step_html("step_*")
            
            # Look at the last several steps for receipt viewing
            recent_steps = all_steps[-10:] if len(all_steps) > 10 else all_steps
            
            for step_file in recent_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for receipt/order details page
                    if any(indicator in content for indicator in [
                        'receipt',
                        'order details',
                        'view receipt',
                        'order summary',
                        'receipt-page'
                    ]):
                        self.details['receipt_viewed'] = True
                        return True
                        
            # Check trajectory for receipt viewing actions
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action_text = json.dumps(entry).lower()
                if 'receipt' in action_text or 'view order' in action_text or 'order details' in action_text:
                    self.details['receipt_action_taken'] = True
                    return True
                    
            self.issues.append("Receipt was not checked after purchase")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking receipt: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Check if agent reported finding no orders (expected failure scenario)
            if not result_data.get('success', False):
                message = result_data.get('message', '').lower()
                if 'no orders' in message or 'no order history' in message:
                    self.issues.append("Task precondition not met: No orders exist in order history")
                    self.details['precondition_failed'] = True

            # Run checkpoint validations
            self.checkpoints['cp1_order_history_accessed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_order_reordered'] = self.check_checkpoint_2()
            self.checkpoints['cp3_promo_code_applied'] = self.check_checkpoint_3()
            self.checkpoints['cp4_payment_completed'] = self.check_checkpoint_4()
            self.checkpoints['cp5_receipt_checked'] = self.check_checkpoint_5()

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
        print("Usage: python food_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
