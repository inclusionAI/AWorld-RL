#!/usr/bin/env python3
# Evaluator for food Query 5

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
        self.in_price_row = False
        self.in_total_row = False
        self.current_price_label = None
        
        self.order_numbers = []
        self.order_statuses = []
        self.price_breakdown = {}
        self.current_url = None
        self.in_order_details_page = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for order details page
        if tag == 'div' and 'class' in attrs_dict:
            if 'order-details-page' in attrs_dict['class']:
                self.in_order_details_page = True
            elif 'order-card' in attrs_dict['class']:
                self.in_order_card = True
            elif 'price-row' in attrs_dict['class']:
                self.in_price_row = True
                if 'total' in attrs_dict['class'] or 'price-total' in attrs_dict['class']:
                    self.in_total_row = True
        
        if tag == 'h3' and self.in_order_card:
            self.in_order_title = True
            
        if tag == 'span' and 'class' in attrs_dict:
            if 'status-badge' in attrs_dict['class']:
                self.in_status_badge = True

    def handle_data(self, data):
        text = data.strip()
        
        if self.in_order_title and text.startswith('Order #'):
            self.order_numbers.append(text)
            
        if self.in_status_badge and text:
            # Clean up status text (remove icons)
            clean_status = text.replace('✓', '').replace('✗', '').strip()
            if clean_status:
                self.order_statuses.append(clean_status)
        
        if self.in_price_row and text:
            if ':' in text:
                # This is a label
                self.current_price_label = text.replace(':', '').strip()
            elif text.startswith('$') and self.current_price_label:
                # This is a value
                self.price_breakdown[self.current_price_label] = text
                
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_order_card:
                self.in_order_card = False
            if self.in_price_row:
                self.in_price_row = False
                self.in_total_row = False
                self.current_price_label = None
        if tag == 'h3':
            self.in_order_title = False
        if tag == 'span':
            self.in_status_badge = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define 3 checkpoints for this task
        self.checkpoints = {
            "cp1_navigated_to_order_history": False,
            "cp2_found_most_recent_order": False,
            "cp3_checked_receipt": False,
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
        """Checkpoint 1: User navigated to order history page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any step has URL containing "order-history"
            for step in trajectory:
                if 'page_info' in step and 'url' in step['page_info']:
                    url = step['page_info']['url']
                    if 'order-history' in url:
                        self.details['order_history_url'] = url
                        return True
            
            self.issues.append("Agent did not navigate to order history page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Found the most recent order in order history"""
        try:
            # Check step 1 HTML (order history page)
            step_1_files = self.find_step_html("step_1_*")
            
            if not step_1_files:
                self.issues.append("No step 1 HTML found to verify order history")
                return False
            
            with open(step_1_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            # Check if we found order #1001 in the order history
            if parser.order_numbers:
                self.details['orders_found'] = parser.order_numbers
                # The most recent order should be #1001
                if any('1001' in order for order in parser.order_numbers):
                    return True
            
            self.issues.append("Most recent order #1001 not found in order history")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking order history: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Checked the receipt (navigated to order details page)"""
        try:
            # Check trajectory for order details navigation
            trajectory = self.get_trajectory()
            
            # Look for navigation to order details page
            order_details_found = False
            for step in trajectory:
                if 'page_info' in step and 'url' in step['page_info']:
                    url = step['page_info']['url']
                    if 'order-details' in url or '/order-details/' in url:
                        order_details_found = True
                        self.details['order_details_url'] = url
                        break
            
            if not order_details_found:
                self.issues.append("Did not navigate to order details page to view receipt")
                return False
            
            # Verify the final HTML shows order details with receipt information
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            # Check if we're on the order details page
            if not parser.in_order_details_page:
                # Alternative check: look for order details indicators in HTML
                if 'order-details-page' not in html_content and 'Order #' not in html_content:
                    self.issues.append("Final page is not showing order details/receipt")
                    return False
            
            # Check if price breakdown is shown (receipt information)
            if 'Subtotal' in html_content and 'Total' in html_content:
                self.details['receipt_shown'] = True
                
                # Try to extract price breakdown details
                if parser.price_breakdown:
                    self.details['price_breakdown'] = parser.price_breakdown
                
                return True
            
            self.issues.append("Receipt information not displayed on order details page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking receipt: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_order_history'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_most_recent_order'] = self.check_checkpoint_2()
            self.checkpoints['cp3_checked_receipt'] = self.check_checkpoint_3()

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
        print("Usage: python food_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
