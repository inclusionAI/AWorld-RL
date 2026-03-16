#!/usr/bin/env python3
# Evaluator for food Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class OrderConfirmationParser(HTMLParser):
    """Parse HTML to extract order confirmation information."""

    def __init__(self):
        super().__init__()
        self.in_order_item_name = False
        self.in_address_text = False
        self.in_summary_row = False
        self.in_success_title = False
        self.current_summary_label = None
        
        self.order_items = []
        self.delivery_address = None
        self.order_summary = {}
        self.is_order_confirmed = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div':
            class_val = attrs_dict.get('class', '')
            if 'order-item-name' in class_val:
                self.in_order_item_name = True
            elif 'address-text' in class_val:
                self.in_address_text = True
            elif 'summary-row' in class_val:
                self.in_summary_row = True
                self.current_summary_label = None
        
        if tag == 'h1':
            class_val = attrs_dict.get('class', '')
            if 'success-title' in class_val:
                self.in_success_title = True

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
            
        if self.in_success_title:
            if 'Order Confirmed' in text:
                self.is_order_confirmed = True
        
        if self.in_order_item_name:
            self.order_items.append(text)
            self.in_order_item_name = False
        
        if self.in_address_text:
            self.delivery_address = text
            self.in_address_text = False
        
        if self.in_summary_row:
            if self.current_summary_label is None:
                self.current_summary_label = text
            else:
                self.order_summary[self.current_summary_label] = text
                self.in_summary_row = False


class AddressFormParser(HTMLParser):
    """Parse checkout page to extract address form information."""

    def __init__(self):
        super().__init__()
        self.in_input = False
        self.in_textarea = False
        self.current_field = None
        
        self.address_fields = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'input':
            placeholder = attrs_dict.get('placeholder', '')
            value = attrs_dict.get('value', '')
            
            if value:
                if 'Home, Work, Office' in placeholder or attrs_dict.get('name') == 'label':
                    self.address_fields['label'] = value
                elif 'Oak Street' in placeholder or '123 Street' in value:
                    self.address_fields['street'] = value
                elif 'Apt' in placeholder or 'Apt' in value:
                    self.address_fields['apartment'] = value
                elif 'Springfield' in placeholder or attrs_dict.get('name') == 'city':
                    self.address_fields['city'] = value
                elif '94102' in placeholder or attrs_dict.get('name') == 'zipCode':
                    self.address_fields['zipCode'] = value
        
        if tag == 'select':
            if attrs_dict.get('name') == 'state':
                self.current_field = 'state'
        
        if tag == 'option' and self.current_field == 'state':
            if attrs_dict.get('selected') is not None:
                self.address_fields['state'] = attrs_dict.get('value', '')
        
        if tag == 'textarea':
            placeholder = attrs_dict.get('placeholder', '')
            if 'doorbell' in placeholder or 'door' in placeholder:
                self.current_field = 'instructions'
    
    def handle_data(self, data):
        text = data.strip()
        if self.current_field == 'instructions' and text:
            self.address_fields['instructions'] = text


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_greek_salad_ordered": False,
            "cp2_address_created": False,
            "cp3_correct_address_used": False,
            "cp4_no_tip_applied": False,
            "cp5_order_confirmed": False
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
        """Checkpoint 1: Greek Salad was ordered"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False

        try:
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = OrderConfirmationParser()
            parser.feed(html_content)
            
            # Check if Greek Salad is in order items
            greek_salad_found = any('Greek Salad' in item for item in parser.order_items)
            
            if greek_salad_found:
                self.details['ordered_items'] = parser.order_items
                return True
            else:
                self.issues.append("Greek Salad not found in order items")
                self.details['ordered_items'] = parser.order_items
                return False
                
        except Exception as e:
            self.issues.append(f"Error parsing order items: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Address was created with correct details"""
        # Check intermediate steps where address form was filled (step 27)
        step_files = self.find_step_html("step_27_*")
        
        if not step_files:
            self.issues.append("Address form submission step not found")
            return False
        
        try:
            with open(step_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = AddressFormParser()
            parser.feed(html_content)
            
            required_fields = {
                'label': 'Temp',
                'street': '123 Street',
                'apartment': 'Apt A1',
                'city': 'Tempe',
                'state': 'AZ',
                'zipCode': '123456',
                'instructions': 'leave at door'
            }
            
            all_correct = True
            mismatches = []
            
            for field, expected in required_fields.items():
                actual = parser.address_fields.get(field, '')
                if expected.lower() not in actual.lower():
                    all_correct = False
                    mismatches.append(f"{field}: expected '{expected}', got '{actual}'")
            
            self.details['address_fields'] = parser.address_fields
            
            if mismatches:
                self.issues.append(f"Address field mismatches: {', '.join(mismatches)}")
                return False
            
            return all_correct
            
        except Exception as e:
            self.issues.append(f"Error checking address form: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Correct address was used for delivery"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        try:
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = OrderConfirmationParser()
            parser.feed(html_content)
            
            if parser.delivery_address:
                address = parser.delivery_address
                self.details['delivery_address'] = address
                
                # Check if address contains key components
                required_parts = ['123 Street', 'Tempe', 'AZ']
                all_present = all(part in address for part in required_parts)
                
                if not all_present:
                    self.issues.append(f"Delivery address missing required parts: {address}")
                    return False
                
                return True
            else:
                self.issues.append("Delivery address not found in order confirmation")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking delivery address: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: No tip was applied (tip = $0)"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        try:
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = OrderConfirmationParser()
            parser.feed(html_content)
            
            self.details['order_summary'] = parser.order_summary
            
            # Check if there's a tip line in the order summary
            tip_amount = parser.order_summary.get('Tip', None)
            
            if tip_amount is None:
                # No tip line means no tip was added (good)
                return True
            else:
                # If tip line exists, check if it's $0.00
                if '$0.00' in tip_amount or tip_amount == '0':
                    return True
                else:
                    self.issues.append(f"Tip was applied: {tip_amount}")
                    return False
                    
        except Exception as e:
            self.issues.append(f"Error checking tip amount: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Order was successfully confirmed"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        try:
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = OrderConfirmationParser()
            parser.feed(html_content)
            
            if parser.is_order_confirmed:
                return True
            else:
                self.issues.append("Order confirmation page not displayed")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking order confirmation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_greek_salad_ordered'] = self.check_checkpoint_1()
            self.checkpoints['cp2_address_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_address_used'] = self.check_checkpoint_3()
            self.checkpoints['cp4_no_tip_applied'] = self.check_checkpoint_4()
            self.checkpoints['cp5_order_confirmed'] = self.check_checkpoint_5()

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
        print("Usage: python food_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
