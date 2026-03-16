#!/usr/bin/env python3
# Evaluator for food Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class CartHTMLParser(HTMLParser):
    """Parse HTML to extract cart and order information."""

    def __init__(self):
        super().__init__()
        self.in_cart_item = False
        self.in_item_name = False
        self.in_item_price = False
        self.in_customization = False
        self.in_remarks = False
        self.in_delivery_time = False
        self.in_tip_section = False
        self.in_order_total = False
        
        self.cart_items = []
        self.current_item = {}
        self.current_text = []
        self.delivery_date = None
        self.delivery_time = None
        self.tip_percentage = None
        self.tip_amount = None
        self.order_completed = False
        self.order_id = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get('class', '')
        
        # Check for cart items
        if 'cart-item' in class_name or 'item-card' in class_name:
            self.in_cart_item = True
            self.current_item = {'customizations': [], 'remarks': ''}
        
        # Check for item name
        if self.in_cart_item and ('item-name' in class_name or 'item-title' in class_name):
            self.in_item_name = True
            self.current_text = []
        
        # Check for customizations
        if self.in_cart_item and ('customization' in class_name or 'option' in class_name):
            self.in_customization = True
            self.current_text = []
        
        # Check for remarks
        if self.in_cart_item and ('remark' in class_name or 'special-instruction' in class_name or 'note' in class_name):
            self.in_remarks = True
            self.current_text = []
        
        # Check for delivery time
        if 'delivery-time' in class_name or 'scheduled-time' in class_name or 'schedule' in class_name:
            self.in_delivery_time = True
            self.current_text = []
        
        # Check for tip
        if 'tip' in class_name.lower():
            self.in_tip_section = True
            self.current_text = []
        
        # Check for order confirmation
        if 'order-id' in class_name or 'order-number' in class_name:
            self.current_text = []
        
        if tag == 'input':
            if attrs_dict.get('type') == 'radio' and 'tip' in attrs_dict.get('name', ''):
                if attrs_dict.get('checked') or attrs_dict.get('checked') == '':
                    value = attrs_dict.get('value', '')
                    if '%' in value or 'percent' in value.lower():
                        self.tip_percentage = value

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        if self.in_item_name:
            self.current_text.append(text)
        
        if self.in_customization:
            self.current_text.append(text)
        
        if self.in_remarks:
            self.current_text.append(text)
        
        if self.in_delivery_time:
            self.current_text.append(text)
        
        if self.in_tip_section:
            self.current_text.append(text)
        
        # Check for order confirmation
        if 'order' in text.lower() and ('#' in text or 'id' in text.lower()):
            self.order_completed = True
            match = re.search(r'#?(\d+)', text)
            if match:
                self.order_id = match.group(1)
        
        # Check for success messages
        if 'success' in text.lower() or 'confirmed' in text.lower() or 'placed' in text.lower():
            if 'order' in text.lower():
                self.order_completed = True

    def handle_endtag(self, tag):
        if self.in_item_name:
            self.in_item_name = False
            name = ' '.join(self.current_text)
            if name:
                self.current_item['name'] = name
        
        if self.in_customization:
            self.in_customization = False
            customization = ' '.join(self.current_text)
            if customization:
                self.current_item['customizations'].append(customization)
        
        if self.in_remarks:
            self.in_remarks = False
            remarks = ' '.join(self.current_text)
            if remarks:
                self.current_item['remarks'] = remarks
        
        if self.in_delivery_time:
            self.in_delivery_time = False
            time_text = ' '.join(self.current_text)
            if time_text:
                # Parse date and time
                date_match = re.search(r'(January|Jan)\s+(\d+),?\s+(\d{4})', time_text)
                if date_match:
                    self.delivery_date = f"{date_match.group(1)} {date_match.group(2)}, {date_match.group(3)}"
                time_match = re.search(r'(\d+):(\d+)\s*(AM|PM)', time_text, re.IGNORECASE)
                if time_match:
                    self.delivery_time = f"{time_match.group(1)}:{time_match.group(2)} {time_match.group(3).upper()}"
        
        if self.in_tip_section:
            self.in_tip_section = False
            tip_text = ' '.join(self.current_text)
            percent_match = re.search(r'(\d+)%', tip_text)
            if percent_match:
                self.tip_percentage = percent_match.group(1)
            amount_match = re.search(r'\$(\d+\.?\d*)', tip_text)
            if amount_match:
                self.tip_amount = amount_match.group(1)
        
        if self.in_cart_item and tag in ['div', 'li', 'article']:
            if self.current_item.get('name'):
                self.cart_items.append(self.current_item)
            self.in_cart_item = False
            self.current_item = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_restaurant_found": False,
            "cp2_correct_pizzas_added": False,
            "cp3_remarks_added": False,
            "cp4_scheduled_delivery": False,
            "cp5_tip_applied": False,
            "cp6_order_completed": False
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
        """Checkpoint 1: Restaurant 'Bella Italia' was found and accessed"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any action involved Bella Italia search
            bella_italia_searched = False
            bella_italia_found = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                params = action.get('parameters', {})
                
                # Check if searched for Bella Italia
                if 'bella italia' in reasoning or 'bella italia' in str(params).lower():
                    bella_italia_searched = True
                
                # Check if successfully navigated to restaurant
                if 'view menu' in reasoning and 'bella italia' in reasoning:
                    result = entry.get('result', {})
                    if result.get('success'):
                        bella_italia_found = True
            
            # Also check HTML files for Bella Italia content
            step_files = self.find_step_html("step_*")
            for step_file in step_files[:20]:  # Check first 20 steps
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if 'bella italia' in content and 'margherita' in content:
                            bella_italia_found = True
                            break
                except:
                    pass
            
            if not bella_italia_searched:
                self.issues.append("Agent did not search for Bella Italia restaurant")
                return False
            
            if not bella_italia_found:
                self.issues.append("Bella Italia restaurant was not found or accessed")
                return False
            
            self.details['restaurant'] = 'Bella Italia found'
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking restaurant: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: 2 Large Margherita Pizzas with Extra Cheese and Stuffed Crust added to cart"""
        try:
            # Check intermediate steps for cart additions
            step_files = self.find_step_html("step_*")
            
            cart_items = []
            for step_file in step_files:
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Look for Margherita Pizza in cart
                        if 'margherita' in content.lower() and ('cart' in content.lower() or 'item' in content.lower()):
                            parser = CartHTMLParser()
                            parser.feed(content)
                            if parser.cart_items:
                                cart_items = parser.cart_items
                except:
                    pass
            
            # Check final state
            final_html = self.find_final_html()
            if final_html:
                try:
                    with open(final_html, 'r', encoding='utf-8') as f:
                        content = f.read()
                        parser = CartHTMLParser()
                        parser.feed(content)
                        if parser.cart_items:
                            cart_items = parser.cart_items
                except:
                    pass
            
            # Validate cart items
            margherita_count = 0
            has_large = False
            has_extra_cheese = False
            has_stuffed_crust = False
            
            for item in cart_items:
                name = item.get('name', '').lower()
                if 'margherita' in name:
                    margherita_count += 1
                    customizations = ' '.join(item.get('customizations', [])).lower()
                    full_text = (name + ' ' + customizations).lower()
                    
                    if 'large' in full_text:
                        has_large = True
                    if 'extra cheese' in full_text or 'extra-cheese' in full_text:
                        has_extra_cheese = True
                    if 'stuffed crust' in full_text or 'stuffed-crust' in full_text:
                        has_stuffed_crust = True
            
            if margherita_count < 2:
                self.issues.append(f"Expected 2 Margherita pizzas, found {margherita_count}")
                return False
            
            if not has_large:
                self.issues.append("Large size not selected for pizzas")
                return False
            
            if not has_extra_cheese:
                self.issues.append("Extra Cheese customization not added")
                return False
            
            if not has_stuffed_crust:
                self.issues.append("Stuffed Crust customization not added")
                return False
            
            self.details['pizzas'] = f"{margherita_count} Margherita pizzas with correct customizations"
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking pizzas: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Remarks 'No Mushrooms' added to order"""
        try:
            step_files = self.find_step_html("step_*")
            
            remarks_found = False
            for step_file in step_files:
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if 'no mushroom' in content and ('remark' in content or 'note' in content or 'instruction' in content):
                            remarks_found = True
                            break
                except:
                    pass
            
            # Check final state
            final_html = self.find_final_html()
            if final_html and not remarks_found:
                try:
                    with open(final_html, 'r', encoding='utf-8') as f:
                        content = f.read()
                        parser = CartHTMLParser()
                        parser.feed(content)
                        
                        for item in parser.cart_items:
                            remarks = item.get('remarks', '').lower()
                            if 'no mushroom' in remarks:
                                remarks_found = True
                                break
                except:
                    pass
            
            if not remarks_found:
                self.issues.append("Remarks 'No Mushrooms' not found in order")
                return False
            
            self.details['remarks'] = 'No Mushrooms added'
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking remarks: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Scheduled delivery set to January 3, 2026 at 7:00 PM"""
        try:
            step_files = self.find_step_html("step_*")
            
            date_found = False
            time_found = False
            
            for step_file in step_files:
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for January 3, 2026
                        if re.search(r'january\s+3,?\s+2026', content, re.IGNORECASE):
                            date_found = True
                        
                        # Check for 7:00 PM
                        if re.search(r'7:?00\s*pm', content, re.IGNORECASE):
                            time_found = True
                        
                        if date_found and time_found:
                            break
                except:
                    pass
            
            # Check final state
            final_html = self.find_final_html()
            if final_html and not (date_found and time_found):
                try:
                    with open(final_html, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        if re.search(r'january\s+3,?\s+2026', content, re.IGNORECASE):
                            date_found = True
                        if re.search(r'7:?00\s*pm', content, re.IGNORECASE):
                            time_found = True
                except:
                    pass
            
            if not date_found:
                self.issues.append("Scheduled delivery date January 3, 2026 not found")
                return False
            
            if not time_found:
                self.issues.append("Scheduled delivery time 7:00 PM not found")
                return False
            
            self.details['scheduled_delivery'] = 'January 3, 2026 at 7:00 PM'
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking scheduled delivery: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: 20% tip applied"""
        try:
            step_files = self.find_step_html("step_*")
            
            tip_found = False
            for step_file in step_files:
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for 20% tip
                        if re.search(r'20\s*%|20\s*percent', content, re.IGNORECASE):
                            if 'tip' in content.lower():
                                tip_found = True
                                break
                except:
                    pass
            
            # Check final state
            final_html = self.find_final_html()
            if final_html and not tip_found:
                try:
                    with open(final_html, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        if re.search(r'20\s*%|20\s*percent', content, re.IGNORECASE) and 'tip' in content.lower():
                            tip_found = True
                except:
                    pass
            
            if not tip_found:
                self.issues.append("20% tip not found in order")
                return False
            
            self.details['tip'] = '20% applied'
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking tip: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Order/payment completed successfully"""
        try:
            # Check trajectory for payment completion
            trajectory = self.get_trajectory()
            
            payment_attempted = False
            payment_completed = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                
                if 'payment' in reasoning or 'pay' in reasoning or 'complete' in reasoning or 'confirm' in reasoning:
                    payment_attempted = True
                    result = entry.get('result', {})
                    if result.get('success'):
                        payment_completed = True
            
            # Check HTML for order confirmation
            step_files = self.find_step_html("step_*")
            order_confirmed = False
            
            for step_file in step_files[-10:]:  # Check last 10 steps
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        # Look for order confirmation indicators
                        if any(phrase in content for phrase in [
                            'order confirmed', 'order placed', 'order successful',
                            'thank you for your order', 'order #', 'order id'
                        ]):
                            order_confirmed = True
                            break
                except:
                    pass
            
            # Check final state
            final_html = self.find_final_html()
            if final_html and not order_confirmed:
                try:
                    with open(final_html, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        if any(phrase in content for phrase in [
                            'order confirmed', 'order placed', 'order successful',
                            'thank you for your order', 'order #', 'order id'
                        ]):
                            order_confirmed = True
                except:
                    pass
            
            # Check result.json
            result_data = self.load_result_json()
            agent_success = result_data.get('success', False)
            
            if not payment_attempted:
                self.issues.append("Payment was not attempted")
                return False
            
            if not order_confirmed and not agent_success:
                self.issues.append("Order confirmation not found")
                return False
            
            self.details['order_status'] = 'Order completed'
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking order completion: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_restaurant_found'] = self.check_checkpoint_1()
            self.checkpoints['cp2_correct_pizzas_added'] = self.check_checkpoint_2()
            self.checkpoints['cp3_remarks_added'] = self.check_checkpoint_3()
            self.checkpoints['cp4_scheduled_delivery'] = self.check_checkpoint_4()
            self.checkpoints['cp5_tip_applied'] = self.check_checkpoint_5()
            self.checkpoints['cp6_order_completed'] = self.check_checkpoint_6()

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
        print("Usage: python food_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
