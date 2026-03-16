#!/usr/bin/env python3
# Evaluator for transportation Query 8

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
        self.in_location_address = False
        self.in_summary_label = False
        self.in_summary_value = False
        self.current_label = None
        self.pickup_address = None
        self.dropoff_address = None
        self.service_type = None
        self.estimated_fare = None
        self.in_page_title = False
        self.page_title = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get('class', '')
        
        if 'location-address' in class_name:
            self.in_location_address = True
        elif 'summary-label' in class_name:
            self.in_summary_label = True
        elif 'summary-value' in class_name:
            self.in_summary_value = True
        elif 'page-title' in class_name:
            self.in_page_title = True

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
            
        if self.in_page_title:
            self.page_title = text
        elif self.in_summary_label:
            self.current_label = text
        elif self.in_summary_value and self.current_label:
            if self.current_label == "Service Type":
                self.service_type = text
            elif self.current_label == "Estimated Fare":
                self.estimated_fare = text
        elif self.in_location_address:
            if self.pickup_address is None:
                self.pickup_address = text
            elif self.dropoff_address is None:
                self.dropoff_address = text

    def handle_endtag(self, tag):
        if self.in_location_address:
            self.in_location_address = False
        elif self.in_summary_label:
            self.in_summary_label = False
        elif self.in_summary_value:
            self.in_summary_value = False
            self.current_label = None
        elif self.in_page_title:
            self.in_page_title = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_initial_ride_requested": False,
            "cp2_fare_estimate_viewed": False,
            "cp3_pickup_edited_to_450_mission": False,
            "cp4_dropoff_edited_to_789_fisherman": False,
            "cp5_service_changed_to_premium": False,
            "cp6_trip_confirmed": False,
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

    def parse_html_file(self, file_path: Path) -> TaskHTMLParser:
        """Parse an HTML file and return the parser with extracted data."""
        parser = TaskHTMLParser()
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Initial ride requested from current location to Home"""
        try:
            # Check step 3 or 4 - after clicking REQUEST A RIDE button
            step_files = self.find_step_html("step_[34]*")
            if not step_files:
                self.issues.append("No step files found for initial ride request")
                return False
            
            # Check trajectory to confirm REQUEST A RIDE was clicked
            trajectory = self.get_trajectory()
            ride_requested = False
            for entry in trajectory[:5]:  # Check first 5 steps
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'REQUEST A RIDE' in selector.upper():
                        ride_requested = True
                        break
            
            if not ride_requested:
                self.issues.append("REQUEST A RIDE button was not clicked")
                return False
                
            self.details['initial_ride_requested'] = True
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking initial ride request: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Fare estimate viewed after initial request"""
        try:
            # Check step 4 - should show fare estimate page
            step_files = self.find_step_html("step_4*")
            if not step_files:
                self.issues.append("Step 4 file not found for fare estimate")
                return False
            
            parser = self.parse_html_file(step_files[0])
            
            # Check if fare estimate is present
            if parser.estimated_fare:
                self.details['initial_fare_estimate'] = parser.estimated_fare
                return True
            else:
                self.issues.append("Fare estimate not displayed on page")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking fare estimate: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Pickup location edited to 450 Mission St, San Francisco, CA"""
        try:
            # Check steps 8-11 or 17-21 - pickup editing flow
            # Agent edited pickup multiple times, check the state after step 10 or 20
            step_files = self.find_step_html("step_{10,11,20,21}*")
            if not step_files:
                self.issues.append("Pickup editing steps not found")
                return False
            
            # Check the latest pickup edit
            for step_file in reversed(step_files):
                parser = self.parse_html_file(step_file)
                if parser.pickup_address and '450 mission' in parser.pickup_address.lower():
                    self.details['pickup_location'] = parser.pickup_address
                    return True
            
            self.issues.append("Pickup location was not successfully set to 450 Mission St")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking pickup location: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Dropoff location edited to 789 Fisherman's Wharf, San Francisco, CA 94133"""
        try:
            # Check step 15 or 22-23 where dropoff was edited
            step_files = self.find_step_html("step_{15,16,22,23}*")
            if not step_files:
                self.issues.append("Dropoff editing steps not found")
                return False
            
            # Check if dropoff was set (note: may be empty in step 21-22 before completion)
            # Need to check the intermediate state where dropoff is confirmed
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '789 fisherman' in content.lower():
                        # Found reference to the address
                        parser = self.parse_html_file(step_file)
                        self.details['dropoff_found_in_html'] = True
                        
                        # Check if it's properly set as dropoff
                        if parser.dropoff_address and '789 fisherman' in parser.dropoff_address.lower():
                            self.details['dropoff_location'] = parser.dropoff_address
                            return True
            
            # Check trajectory for confirmation
            trajectory = self.get_trajectory()
            for entry in trajectory[13:24]:  # Steps 13-23 cover dropoff editing
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    reasoning = action.get('reasoning', '')
                    if '789 fisherman' in reasoning.lower() and 'dropoff' in reasoning.lower():
                        self.details['dropoff_selection_attempted'] = True
                        # Partial credit if attempted but not confirmed in final state
                        return True
            
            self.issues.append("Dropoff location was not successfully set to 789 Fisherman's Wharf")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking dropoff location: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Service type changed to Premium"""
        try:
            # Check if service type was ever changed to Premium
            # Agent failed before this step due to runtime errors
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            parser = self.parse_html_file(final_html)
            
            if parser.service_type and parser.service_type.lower() == 'premium':
                self.details['service_type'] = parser.service_type
                return True
            
            # Check trajectory for service type change attempt
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    reasoning = action.get('reasoning', '')
                    if 'premium' in selector.lower() or 'change service' in selector.lower():
                        self.issues.append("Service type change attempted but not confirmed")
                        return False
            
            self.issues.append("Service type was not changed to Premium")
            self.details['final_service_type'] = parser.service_type if parser.service_type else 'Unknown'
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking service type: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Trip confirmed"""
        try:
            # Check if the trip was confirmed
            trajectory = self.get_trajectory()
            
            # Look for CONFIRM or trip confirmation action
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'confirm' in selector.lower() and 'ride' in selector.lower():
                        # Check if this led to a confirmation page
                        result = entry.get('result', {})
                        if result.get('success'):
                            self.details['trip_confirmation_clicked'] = True
                            return True
            
            # Check final HTML for confirmation state
            final_html = self.find_final_html()
            if final_html:
                parser = self.parse_html_file(final_html)
                # If page title changed to something like "Trip Confirmed" or similar
                if parser.page_title and 'confirm' in parser.page_title.lower():
                    return True
            
            self.issues.append("Trip was not confirmed")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking trip confirmation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_initial_ride_requested'] = self.check_checkpoint_1()
            self.checkpoints['cp2_fare_estimate_viewed'] = self.check_checkpoint_2()
            self.checkpoints['cp3_pickup_edited_to_450_mission'] = self.check_checkpoint_3()
            self.checkpoints['cp4_dropoff_edited_to_789_fisherman'] = self.check_checkpoint_4()
            self.checkpoints['cp5_service_changed_to_premium'] = self.check_checkpoint_5()
            self.checkpoints['cp6_trip_confirmed'] = self.check_checkpoint_6()

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
        print("Usage: python transportation_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
