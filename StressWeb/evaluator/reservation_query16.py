#!/usr/bin/env python3
# Evaluator for reservation Query 16

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ReservationModificationParser(HTMLParser):
    """Parse HTML to extract reservation modification information."""

    def __init__(self):
        super().__init__()
        self.in_restaurant_details = False
        self.in_original_details = False
        self.in_modification_form = False
        self.current_tag = None
        self.current_attrs = {}
        
        self.restaurant_name = None
        self.original_date = None
        self.original_time = None
        self.original_party_size = None
        
        self.new_date = None
        self.new_party_size = None
        self.selected_time = None
        
        self.detail_label = None
        self.capture_detail_value = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_attrs = attrs_dict
        
        # Check for restaurant name
        if tag == 'div' and attrs_dict.get('class') == 'restaurant-details':
            self.in_restaurant_details = True
        
        # Check for original reservation details
        if tag == 'div' and attrs_dict.get('class') == 'original-details':
            self.in_original_details = True
        
        # Check for modification form
        if tag == 'div' and attrs_dict.get('class') == 'modification-form-card':
            self.in_modification_form = True
        
        # Capture date input value
        if tag == 'input' and attrs_dict.get('type') == 'date' and self.in_modification_form:
            self.new_date = attrs_dict.get('value', '')
        
        # Capture selected party size
        if tag == 'button' and 'party-size-btn' in attrs_dict.get('class', ''):
            if 'selected' in attrs_dict.get('class', ''):
                # Will capture text in handle_data
                self.capture_detail_value = True
        
        # Capture selected time slot
        if tag == 'button' and 'time-slot-btn' in attrs_dict.get('class', ''):
            if 'selected' in attrs_dict.get('class', ''):
                # Will capture text in handle_data
                self.capture_detail_value = True
        
        # Capture detail labels for original reservation
        if tag == 'span' and attrs_dict.get('class') == 'detail-label' and self.in_original_details:
            self.capture_detail_value = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        # Capture restaurant name
        if self.in_restaurant_details and self.current_tag == 'h2':
            self.restaurant_name = data
        
        # Capture detail labels and values for original reservation
        if self.in_original_details:
            if self.current_tag == 'span' and self.capture_detail_value:
                self.detail_label = data.replace(':', '').strip()
                self.capture_detail_value = False
            elif self.current_tag == 'span' and self.current_attrs.get('class') == 'detail-value':
                if self.detail_label == 'Date':
                    self.original_date = data
                elif self.detail_label == 'Time':
                    self.original_time = data
                elif self.detail_label == 'Party Size':
                    self.original_party_size = data
        
        # Capture selected party size in modification form
        if self.in_modification_form and self.current_tag == 'button':
            if 'party-size-btn' in self.current_attrs.get('class', ''):
                if 'selected' in self.current_attrs.get('class', ''):
                    self.new_party_size = data
            
            # Capture selected time slot
            if 'time-slot-btn' in self.current_attrs.get('class', ''):
                if 'selected' in self.current_attrs.get('class', ''):
                    self.selected_time = data

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_restaurant_details and self.current_attrs.get('class') == 'restaurant-details':
                self.in_restaurant_details = False
            if self.in_original_details and self.current_attrs.get('class') == 'original-details':
                self.in_original_details = False
            if self.in_modification_form and self.current_attrs.get('class') == 'modification-form-card':
                self.in_modification_form = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_reservations": False,
            "cp2_opened_first_reservation": False,
            "cp3_modified_date_to_2030": False,
            "cp4_modified_party_size_to_4": False,
            "cp5_selected_time_close_to_715pm": False,
        }

        self.issues = []
        self.details = {}

    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        # Check for step_100 first since that's the last step
        step_100_files = list(self.result_dir.glob("step_100_*_raw.html"))
        if step_100_files:
            return step_100_files[0]
        
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
        """Checkpoint 1: Navigated to My Reservations page"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked on "My Reservations" link
            for action in action_history:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'My Reservations' in selector or 'reservations' in selector.lower():
                        self.details['navigated_to_reservations'] = True
                        return True
            
            self.issues.append("Did not navigate to My Reservations page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Opened the first restaurant reservation (The Garden Bistro)"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked on "The Garden Bistro" or clicked Modify button
            for action in action_history:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'The Garden Bistro' in selector or 'Modify' in selector:
                        self.details['opened_first_reservation'] = True
                        return True
            
            self.issues.append("Did not open The Garden Bistro reservation")
            return False
        except Exception as e:
            self.issues.append(f"Error checking reservation selection: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Modified date to January 1, 2030"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = ReservationModificationParser()
            parser.feed(html_content)
            
            # Check if date is set to 2030-01-01
            if parser.new_date:
                self.details['new_date'] = parser.new_date
                if parser.new_date == '2030-01-01':
                    return True
                else:
                    self.issues.append(f"Date set to {parser.new_date} instead of 2030-01-01")
                    return False
            else:
                self.issues.append("Date input not found or not set")
                return False
        except Exception as e:
            self.issues.append(f"Error checking date modification: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Modified party size to 4 people"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = ReservationModificationParser()
            parser.feed(html_content)
            
            # Check if party size is 4
            if parser.new_party_size:
                self.details['new_party_size'] = parser.new_party_size
                if '4' in parser.new_party_size:
                    return True
                else:
                    self.issues.append(f"Party size set to {parser.new_party_size} instead of 4")
                    return False
            else:
                self.issues.append("Party size not found or not selected")
                return False
        except Exception as e:
            self.issues.append(f"Error checking party size: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Selected time close to 7:15 PM (7:00 PM or 7:30 PM acceptable)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = ReservationModificationParser()
            parser.feed(html_content)
            
            # Check if time is selected (7:00 PM or 7:30 PM acceptable since 7:15 PM may not be available)
            if parser.selected_time:
                self.details['selected_time'] = parser.selected_time
                time_lower = parser.selected_time.lower()
                if '7:00' in time_lower or '7:30' in time_lower or '7:15' in time_lower:
                    if '7:15' in time_lower:
                        return True
                    else:
                        # Accept 7:00 PM or 7:30 PM as close enough since 7:15 PM may not be available
                        self.issues.append(f"Time set to {parser.selected_time} (acceptable, but task specified 7:15 PM)")
                        return True
                else:
                    self.issues.append(f"Time set to {parser.selected_time} instead of 7:15 PM")
                    return False
            else:
                self.issues.append("Time slot not selected")
                return False
        except Exception as e:
            self.issues.append(f"Error checking time selection: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_reservations'] = self.check_checkpoint_1()
            self.checkpoints['cp2_opened_first_reservation'] = self.check_checkpoint_2()
            self.checkpoints['cp3_modified_date_to_2030'] = self.check_checkpoint_3()
            self.checkpoints['cp4_modified_party_size_to_4'] = self.check_checkpoint_4()
            self.checkpoints['cp5_selected_time_close_to_715pm'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 16,
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
                'query_id': 16,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query16.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
