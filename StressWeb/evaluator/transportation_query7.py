#!/usr/bin/env python3
# Evaluator for transportation Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ServiceComparisonParser(HTMLParser):
    """Parse HTML to extract service selection information."""

    def __init__(self):
        super().__init__()
        self.in_service_card = False
        self.in_service_name = False
        self.in_price_value = False
        self.in_button = False
        self.in_ride_summary = False
        self.in_summary_value = False
        self.current_service = None
        self.service_selected = {}
        self.service_prices = {}
        self.current_button_text = ""
        self.buttons = []
        self.service_type_value = None
        self.capture_next_text = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for service card
        if tag == "div" and "class" in attrs_dict:
            classes = attrs_dict["class"]
            if "service-card" in classes:
                self.in_service_card = True
                self.current_service = None
                # Check if selected
                if "selected" in classes:
                    self.capture_next_text = True
            elif "service-name" in classes:
                self.in_service_name = True
            elif "price-value" in classes:
                self.in_price_value = True
            elif "ride-summary" in classes:
                self.in_ride_summary = True
            elif "summary-value" in classes and self.in_ride_summary:
                self.in_summary_value = True
                
        # Check for buttons
        if tag == "button":
            self.in_button = True
            self.current_button_text = ""
            
        # Check for service type in summary
        if tag == "span" and self.in_ride_summary:
            self.in_summary_value = True

    def handle_data(self, data):
        text = data.strip()
        
        if self.in_service_name and text:
            self.current_service = text
            if self.capture_next_text:
                self.service_selected[text] = True
                self.capture_next_text = False
                
        if self.in_price_value and text and self.current_service:
            self.service_prices[self.current_service] = text
            
        if self.in_button and text:
            self.current_button_text += text
            
        if self.in_summary_value and text and "Service Type" in getattr(self, '_last_label', ''):
            self.service_type_value = text
            
    def handle_endtag(self, tag):
        if tag == "div":
            if self.in_service_card:
                self.in_service_card = False
            if self.in_service_name:
                self.in_service_name = False
            if self.in_price_value:
                self.in_price_value = False
            if self.in_ride_summary:
                self.in_ride_summary = False
            if self.in_summary_value:
                self.in_summary_value = False
                
        if tag == "button":
            if self.current_button_text:
                self.buttons.append(self.current_button_text.strip())
            self.in_button = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_ride_requested": False,
            "cp2_fare_estimate_viewed": False,
            "cp3_service_comparison_opened": False,
            "cp4_premium_selected": False,
            "cp5_payment_completed": False,
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
        """Checkpoint 1: Initial ride requested with Standard service from current location to Home"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if ride was requested with appropriate pickup and dropoff
            ride_requested = False
            for entry in trajectory[:10]:  # Check first 10 steps
                action = entry.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                
                if 'request' in reasoning and 'ride' in reasoning:
                    ride_requested = True
                    break
                    
                # Also check for clicking "REQUEST A RIDE" button
                if action.get('action_type') == 'CLICK':
                    params = action.get('parameters', {})
                    selector = params.get('selector', '')
                    if 'REQUEST' in selector.upper():
                        ride_requested = True
                        break
            
            if not ride_requested:
                self.issues.append("No ride request action found in trajectory")
                return False
                
            self.details['ride_requested'] = True
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking ride request: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Fare estimate viewed for Standard ride (step 7 approximately)"""
        try:
            # Check step 7 where fare estimate should be visible
            step_files = self.find_step_html("step_7_*")
            
            if not step_files:
                self.issues.append("Step 7 HTML file not found")
                return False
                
            with open(step_files[0], 'r', encoding='utf-8') as f:
                html = f.read()
                
            # Look for fare estimate indicators
            if 'Estimated Fare' in html or 'fare-value' in html:
                # Extract the fare value
                fare_match = re.search(r'\$(\d+\.?\d*)', html)
                if fare_match:
                    self.details['standard_fare_viewed'] = fare_match.group(0)
                    return True
                    
            self.issues.append("Fare estimate not found at expected step")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking fare estimate: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Service comparison page opened (Change Service Type clicked)"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for action to change service type
            comparison_opened = False
            for entry in trajectory:
                action = entry.get('action', {})
                
                if action.get('action_type') == 'CLICK':
                    params = action.get('parameters', {})
                    selector = params.get('selector', '')
                    reasoning = action.get('reasoning', '').lower()
                    
                    if 'CHANGE SERVICE' in selector.upper() or 'change service' in reasoning:
                        comparison_opened = True
                        break
            
            # Also check final HTML for comparison page
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html = f.read()
                    
                if 'Choose Service Type' in html or 'service-comparison' in html:
                    comparison_opened = True
                    self.details['comparison_page_reached'] = True
                    
            if not comparison_opened:
                self.issues.append("Service comparison page not reached")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking service comparison: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Premium service selected and confirmed"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
                
            with open(final_html, 'r', encoding='utf-8') as f:
                html = f.read()
            
            # Check if Premium was selected (should show in ride summary or confirmation)
            parser = ServiceComparisonParser()
            parser.feed(html)
            
            # If still on comparison page, check if Premium is selected
            if 'Choose Service Type' in html or 'service-comparison' in html:
                self.issues.append("Still on service comparison page - Premium not confirmed")
                self.details['stuck_at'] = 'service_comparison'
                return False
            
            # Check if Premium appears in confirmed ride details
            if 'Premium' in html:
                # Look for ride summary or confirmation page with Premium
                if 'Service Type' in html:
                    # Extract service type from summary
                    service_match = re.search(r'Service Type.*?Premium', html, re.DOTALL)
                    if service_match:
                        self.details['premium_confirmed'] = True
                        return True
            
            self.issues.append("Premium service not confirmed in final state")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Premium selection: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Payment completed"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False
                
            with open(final_html, 'r', encoding='utf-8') as f:
                html = f.read()
            
            # Check for payment confirmation or trip confirmation
            payment_indicators = [
                'payment-confirmation',
                'Payment Successful',
                'Trip Confirmed',
                'booking-confirmed',
                'Thank you for your payment'
            ]
            
            for indicator in payment_indicators:
                if indicator in html:
                    self.details['payment_completed'] = True
                    return True
            
            # Check trajectory for payment actions
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action = entry.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                
                if 'payment' in reasoning and ('complete' in reasoning or 'confirm' in reasoning):
                    self.details['payment_action_found'] = True
                    return True
            
            self.issues.append("Payment not completed")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking payment: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_ride_requested'] = self.check_checkpoint_1()
            self.checkpoints['cp2_fare_estimate_viewed'] = self.check_checkpoint_2()
            self.checkpoints['cp3_service_comparison_opened'] = self.check_checkpoint_3()
            self.checkpoints['cp4_premium_selected'] = self.check_checkpoint_4()
            self.checkpoints['cp5_payment_completed'] = self.check_checkpoint_5()

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
        print("Usage: python transportation_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
