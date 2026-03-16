#!/usr/bin/env python3
# Evaluator for transportation Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract ride booking information."""

    def __init__(self):
        super().__init__()
        self.in_location_address = False
        self.in_summary_value = False
        self.in_summary_label = False
        self.current_label = None
        self.pickup_location = None
        self.dropoff_location = None
        self.service_type = None
        self.estimated_fare = None
        self.distance = None
        self.duration = None
        self.current_location_type = None
        self.in_location_info = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for location address
        if tag == 'span' and attrs_dict.get('class') == 'location-address':
            self.in_location_address = True
            
        # Check for location label to identify pickup/dropoff
        if tag == 'span' and attrs_dict.get('class') == 'location-label':
            self.in_summary_label = True
            
        # Check for summary values
        if tag == 'span' and attrs_dict.get('class') == 'summary-value':
            self.in_summary_value = True
            
        # Check for summary labels
        if tag == 'span' and attrs_dict.get('class') == 'summary-label':
            self.in_summary_label = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture location label (Pickup/Dropoff)
        if self.in_summary_label and data in ['Pickup', 'Dropoff']:
            self.current_location_type = data
            
        # Capture location addresses
        if self.in_location_address and data:
            if self.current_location_type == 'Pickup' and not self.pickup_location:
                self.pickup_location = data
            elif self.current_location_type == 'Dropoff' and not self.dropoff_location:
                self.dropoff_location = data
                
        # Capture summary labels
        if self.in_summary_label and data in ['Service Type', 'Estimated Fare', 'Distance', 'Duration']:
            self.current_label = data
            
        # Capture summary values
        if self.in_summary_value and self.current_label:
            if self.current_label == 'Service Type' and not self.service_type:
                self.service_type = data
            elif self.current_label == 'Estimated Fare' and not self.estimated_fare:
                self.estimated_fare = data
            elif self.current_label == 'Distance' and not self.distance:
                self.distance = data
            elif self.current_label == 'Duration' and not self.duration:
                self.duration = data

    def handle_endtag(self, tag):
        if tag == 'span':
            self.in_location_address = False
            self.in_summary_value = False
            self.in_summary_label = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_initial_ride_requested": False,
            "cp2_initial_fare_viewed": False,
            "cp3_pickup_edited": False,
            "cp4_updated_fare_viewed": False,
            "cp5_shared_service_selected": False,
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

    def parse_html_for_ride_info(self, html_path: Path) -> Dict:
        """Parse HTML to extract ride information."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = TaskHTMLParser()
        parser.feed(html_content)
        
        return {
            'pickup': parser.pickup_location,
            'dropoff': parser.dropoff_location,
            'service_type': parser.service_type,
            'fare': parser.estimated_fare,
            'distance': parser.distance,
            'duration': parser.duration
        }

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Initial ride requested with correct pickup and dropoff"""
        try:
            # Check step 8 (after requesting ride, before edit)
            step_files = self.find_step_html("step_8_*")
            if not step_files:
                self.issues.append("No step 8 HTML found - initial ride request not found")
                return False
            
            ride_info = self.parse_html_for_ride_info(step_files[0])
            self.details['initial_pickup'] = ride_info['pickup']
            self.details['initial_dropoff'] = ride_info['dropoff']
            
            # Check if initial pickup was "456 Union Square, San Francisco"
            if not ride_info['pickup'] or '456 Union Square' not in ride_info['pickup']:
                self.issues.append(f"Initial pickup location incorrect: {ride_info['pickup']}")
                return False
                
            # Check if dropoff was "Pier 39, San Francisco"
            if not ride_info['dropoff'] or 'Pier 39' not in ride_info['dropoff']:
                self.issues.append(f"Dropoff location incorrect: {ride_info['dropoff']}")
                return False
                
            return True
        except Exception as e:
            self.issues.append(f"Error checking initial ride request: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Initial fare estimate viewed"""
        try:
            # Check step 8 for fare estimate display
            step_files = self.find_step_html("step_8_*")
            if not step_files:
                return False
            
            ride_info = self.parse_html_for_ride_info(step_files[0])
            
            if not ride_info['fare']:
                self.issues.append("Initial fare estimate not found")
                return False
                
            self.details['initial_fare'] = ride_info['fare']
            return True
        except Exception as e:
            self.issues.append(f"Error checking initial fare estimate: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Pickup location edited to Pier 39, San Francisco, CA"""
        try:
            # Check final state or step 25 (after edit confirmation)
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            ride_info = self.parse_html_for_ride_info(final_html)
            self.details['final_pickup'] = ride_info['pickup']
            
            # Check if pickup was changed to "Pier 39, San Francisco, CA"
            if not ride_info['pickup'] or 'Pier 39' not in ride_info['pickup']:
                self.issues.append(f"Pickup not edited correctly: {ride_info['pickup']}")
                return False
                
            return True
        except Exception as e:
            self.issues.append(f"Error checking edited pickup: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Updated fare estimate viewed after pickup edit"""
        try:
            # Check final state for updated fare
            final_html = self.find_final_html()
            if not final_html:
                return False
            
            ride_info = self.parse_html_for_ride_info(final_html)
            
            if not ride_info['fare']:
                self.issues.append("Updated fare estimate not found")
                return False
                
            self.details['final_fare'] = ride_info['fare']
            
            # The fare should be different from initial (since locations changed)
            # According to the example, final fare is $6.00 for 0.5 miles
            return True
        except Exception as e:
            self.issues.append(f"Error checking updated fare: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Shared service type was selected initially"""
        try:
            # Check trajectory for "Shared" selection
            trajectory = self.get_trajectory()
            
            shared_selected = False
            for action in trajectory:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Shared' in selector:
                        shared_selected = True
                        break
            
            if not shared_selected:
                self.issues.append("Shared service type was not selected")
                return False
                
            self.details['service_type_selected'] = 'Shared'
            return True
        except Exception as e:
            self.issues.append(f"Error checking service type: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_initial_ride_requested'] = self.check_checkpoint_1()
            self.checkpoints['cp2_initial_fare_viewed'] = self.check_checkpoint_2()
            self.checkpoints['cp3_pickup_edited'] = self.check_checkpoint_3()
            self.checkpoints['cp4_updated_fare_viewed'] = self.check_checkpoint_4()
            self.checkpoints['cp5_shared_service_selected'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 6,
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
                'query_id': 6,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python transportation_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
