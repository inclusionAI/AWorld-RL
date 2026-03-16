#!/usr/bin/env python3
# Evaluator for transportation Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RideStatusParser(HTMLParser):
    """Parse HTML to extract ride-related information."""
    def __init__(self):
        super().__init__()
        self.in_page_div = False
        self.page_class = None
        self.current_tag = None
        self.current_attrs = {}
        self.text_content = []
        self.button_texts = []
        self.location_info = {'pickup': None, 'dropoff': None}
        self.service_type = None
        self.fare_amount = None
        self.in_location_label = False
        self.in_location_address = False
        self.current_location_type = None
        self.in_service_name = False
        self.in_fare_value = False
        self.in_button = False
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        if tag == 'div':
            class_val = self.current_attrs.get('class', '')
            if 'booking-confirm-page' in class_val:
                self.page_class = 'booking-confirm-page'
            elif 'driver-enroute-page' in class_val:
                self.page_class = 'driver-enroute-page'
            elif 'confirmed-page' in class_val:
                self.page_class = 'confirmed-page'
                
        if tag == 'span':
            class_val = self.current_attrs.get('class', '')
            if 'location-label' in class_val:
                self.in_location_label = True
            elif 'location-address' in class_val:
                self.in_location_address = True
                
        if tag == 'h2':
            class_val = self.current_attrs.get('class', '')
            if 'service-name' in class_val:
                self.in_service_name = True
                
        if tag == 'span':
            class_val = self.current_attrs.get('class', '')
            if 'fare-value' in class_val:
                self.in_fare_value = True
                
        if tag == 'button':
            self.in_button = True
        
    def handle_data(self, data):
        data = data.strip()
        if data:
            self.text_content.append(data)
            
            if self.in_location_label:
                if 'Pickup' in data:
                    self.current_location_type = 'pickup'
                elif 'Dropoff' in data or 'Drop-off' in data:
                    self.current_location_type = 'dropoff'
                    
            if self.in_location_address and self.current_location_type:
                self.location_info[self.current_location_type] = data
                self.current_location_type = None
                
            if self.in_service_name:
                self.service_type = data
                
            if self.in_fare_value:
                # Extract numeric value from fare string like "$18.50" or "$$18.50"
                match = re.search(r'\$+\s*([\d.]+)', data)
                if match:
                    self.fare_amount = float(match.group(1))
                    
            if self.in_button:
                self.button_texts.append(data)
    
    def handle_endtag(self, tag):
        if tag == 'span':
            self.in_location_label = False
            self.in_location_address = False
            self.in_fare_value = False
        if tag == 'h2':
            self.in_service_name = False
        if tag == 'button':
            self.in_button = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints for Query 4:
        # "Request a Standard ride using quick fill from current location to Home, 
        #  confirm the trip, and complete the payment."
        self.checkpoints = {
            "cp1_locations_filled": False,
            "cp2_service_type_standard": False,
            "cp3_trip_confirmed": False,
            "cp4_payment_completed": False,
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
    
    def parse_html_file(self, html_path: Path) -> RideStatusParser:
        """Parse HTML file and return parser with extracted data."""
        parser = RideStatusParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser
    
    def check_checkpoint_1(self) -> bool:
        """Check if pickup and dropoff locations were filled correctly."""
        # Check trajectory for location filling actions
        trajectory = self.get_trajectory()
        
        pickup_filled = False
        dropoff_filled = False
        
        for entry in trajectory:
            action = entry.get('action', {})
            action_type = action.get('action_type', '')
            params = action.get('parameters', {})
            
            # Check for pickup location fill
            if action_type == 'FILL' and 'Pickup location' in str(params.get('selector', '')):
                text = params.get('text', '')
                if 'Current Location' in text or 'current' in text.lower():
                    pickup_filled = True
            
            # Check for Home button click (quick fill for dropoff)
            if action_type == 'CLICK' and 'Home' in str(params.get('selector', '')):
                dropoff_filled = True
        
        # Also verify in HTML states
        step_files = self.find_step_html("step_*")
        for step_file in step_files[3:10]:  # Check early steps
            parser = self.parse_html_file(step_file)
            if parser.location_info.get('pickup') and parser.location_info.get('dropoff'):
                if 'Current Location' in parser.location_info['pickup']:
                    pickup_filled = True
                if '742 Evergreen Terrace' in parser.location_info['dropoff'] or 'Home' in str(parser.location_info['dropoff']):
                    dropoff_filled = True
        
        self.details['pickup_filled'] = pickup_filled
        self.details['dropoff_filled'] = dropoff_filled
        
        if not pickup_filled:
            self.issues.append("Pickup location 'Current Location' was not properly filled")
        if not dropoff_filled:
            self.issues.append("Dropoff location 'Home' was not selected using quick fill")
            
        return pickup_filled and dropoff_filled
    
    def check_checkpoint_2(self) -> bool:
        """Check if Standard service type was selected."""
        trajectory = self.get_trajectory()
        
        standard_selected = False
        
        for entry in trajectory:
            action = entry.get('action', {})
            action_type = action.get('action_type', '')
            params = action.get('parameters', {})
            
            # Check for Standard button click
            if action_type == 'CLICK':
                selector = params.get('selector', '')
                if 'Standard' in selector:
                    standard_selected = True
                    break
        
        # Verify in HTML states
        step_files = self.find_step_html("step_*")
        for step_file in step_files[4:15]:
            parser = self.parse_html_file(step_file)
            if parser.service_type and 'Standard' in parser.service_type:
                standard_selected = True
                break
        
        self.details['standard_selected'] = standard_selected
        
        if not standard_selected:
            self.issues.append("Standard service type was not selected")
            
        return standard_selected
    
    def check_checkpoint_3(self) -> bool:
        """Check if trip was confirmed."""
        trajectory = self.get_trajectory()
        
        confirm_clicked = False
        
        # Look for CONFIRM actions
        for entry in trajectory:
            action = entry.get('action', {})
            action_type = action.get('action_type', '')
            params = action.get('parameters', {})
            
            if action_type == 'CLICK':
                selector = params.get('selector', '')
                if 'CONFIRM' in selector.upper() or 'Confirm' in selector:
                    confirm_clicked = True
                    break
        
        # Check HTML for confirmation page or driver-enroute page
        step_files = self.find_step_html("step_*")
        reached_enroute = False
        
        for step_file in step_files[10:30]:
            parser = self.parse_html_file(step_file)
            if parser.page_class == 'driver-enroute-page':
                reached_enroute = True
                break
            # Check for confirm button in text
            for btn_text in parser.button_texts:
                if 'Confirm' in btn_text and 'Ride' in btn_text:
                    confirm_clicked = True
        
        self.details['confirm_clicked'] = confirm_clicked
        self.details['reached_enroute'] = reached_enroute
        
        if not confirm_clicked:
            self.issues.append("Trip confirmation button was not clicked")
        if not reached_enroute:
            self.issues.append("Did not reach driver-enroute state")
            
        return confirm_clicked and reached_enroute
    
    def check_checkpoint_4(self) -> bool:
        """Check if payment was completed."""
        # The task requires completing the payment
        # However, in the result.json, we can see the agent got stuck at driver-enroute
        # and reached max_steps without completing payment
        
        result_data = self.load_result_json()
        
        # Check if agent reached payment completion
        trajectory = self.get_trajectory()
        
        payment_confirmed = False
        payment_page_reached = False
        
        # Look for payment confirmation actions
        for entry in trajectory:
            action = entry.get('action', {})
            action_type = action.get('action_type', '')
            params = action.get('parameters', {})
            
            if action_type == 'CLICK':
                selector = params.get('selector', '')
                # Look for payment confirmation buttons
                if any(term in selector.upper() for term in ['PAY', 'COMPLETE', 'CONFIRM PAYMENT']):
                    payment_confirmed = True
        
        # Check final state
        final_html = self.find_final_html()
        if final_html:
            parser = self.parse_html_file(final_html)
            
            # If still on driver-enroute page, payment was not completed
            if parser.page_class == 'driver-enroute-page':
                self.issues.append("Agent got stuck on driver-enroute page and did not complete payment")
                self.details['stuck_on_driver_enroute'] = True
            elif parser.page_class == 'confirmed-page':
                payment_page_reached = True
        
        self.details['payment_confirmed'] = payment_confirmed
        self.details['payment_page_reached'] = payment_page_reached
        
        # Check reason for failure
        if result_data.get('reason') == 'max_steps':
            self.issues.append("Agent reached maximum steps (100) without completing the task")
        
        if not payment_confirmed and not payment_page_reached:
            self.issues.append("Payment was not completed - task incomplete")
            
        return payment_confirmed or payment_page_reached
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_locations_filled'] = self.check_checkpoint_1()
            self.checkpoints['cp2_service_type_standard'] = self.check_checkpoint_2()
            self.checkpoints['cp3_trip_confirmed'] = self.check_checkpoint_3()
            self.checkpoints['cp4_payment_completed'] = self.check_checkpoint_4()
            
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
        print("Usage: python transportation_query4.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
