#!/usr/bin/env python3
# Evaluator for transportation Query 3

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
        self.in_input_field = False
        self.in_service_btn = False
        self.in_service_name = False
        self.in_fare_section = False
        self.current_input_value = None
        self.current_service_name = None
        self.current_service_active = False
        
        self.pickup_location = None
        self.dropoff_location = None
        self.selected_service = None
        self.fare_amount = None
        self.fare_visible = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for pickup/dropoff input fields
        if tag == 'input' and attrs_dict.get('placeholder') == 'Pickup location':
            self.pickup_location = attrs_dict.get('value', '')
            
        if tag == 'input' and attrs_dict.get('placeholder') == 'Where to?':
            self.dropoff_location = attrs_dict.get('value', '')
        
        # Check for service type buttons
        if tag == 'button' and 'service-type-btn' in attrs_dict.get('class', ''):
            self.in_service_btn = True
            self.current_service_active = 'active' in attrs_dict.get('class', '')
            self.current_service_name = None
            
        if self.in_service_btn and tag == 'div' and 'service-name' in attrs_dict.get('class', ''):
            self.in_service_name = True
            
        # Check for fare information
        if 'fare' in attrs_dict.get('class', '').lower():
            self.in_fare_section = True

    def handle_data(self, data):
        data = data.strip()
        
        if self.in_service_name and data:
            self.current_service_name = data
            
        if self.in_fare_section and data and '$' in data:
            self.fare_amount = data
            self.fare_visible = True

    def handle_endtag(self, tag):
        if tag == 'button' and self.in_service_btn:
            if self.current_service_active and self.current_service_name:
                self.selected_service = self.current_service_name
            self.in_service_btn = False
            self.current_service_active = False
            
        if tag == 'div' and self.in_service_name:
            self.in_service_name = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_pickup_entered": False,
            "cp2_destination_entered": False,
            "cp3_shared_selected": False,
            "cp4_fare_viewed": False,
            "cp5_not_booked": False,
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

    def parse_html_file(self, html_path: Path) -> TaskHTMLParser:
        """Parse an HTML file and return parser with extracted data."""
        parser = TaskHTMLParser()
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                parser.feed(f.read())
        except Exception as e:
            self.issues.append(f"Error parsing HTML: {str(e)}")
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Pickup location correctly entered."""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No HTML file found")
            return False
        
        parser = self.parse_html_file(html_file)
        
        # Check if pickup location contains "456 Union Square, San Francisco"
        if parser.pickup_location:
            self.details['pickup_location'] = parser.pickup_location
            expected_keywords = ["456", "union", "square", "san francisco"]
            pickup_lower = parser.pickup_location.lower()
            
            if all(kw in pickup_lower for kw in expected_keywords):
                return True
            else:
                self.issues.append(f"Pickup location incorrect: '{parser.pickup_location}'")
                return False
        else:
            self.issues.append("Pickup location not found in HTML")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Destination correctly entered."""
        html_file = self.find_final_html()
        if not html_file:
            return False
        
        parser = self.parse_html_file(html_file)
        
        # Check if destination contains "Pier 39, San Francisco"
        if parser.dropoff_location:
            self.details['dropoff_location'] = parser.dropoff_location
            expected_keywords = ["pier", "39", "san francisco"]
            dropoff_lower = parser.dropoff_location.lower()
            
            if all(kw in dropoff_lower for kw in expected_keywords):
                return True
            else:
                self.issues.append(f"Destination incorrect: '{parser.dropoff_location}'")
                return False
        else:
            self.issues.append("Destination not found in HTML")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Shared ride service type selected."""
        # Need to check intermediate steps since selection might trigger navigation
        all_html_files = self.find_step_html("step_*")
        if not all_html_files:
            html_file = self.find_final_html()
            if html_file:
                all_html_files = [html_file]
        
        # Check all HTML files to see if Shared was ever selected
        for html_file in all_html_files:
            parser = self.parse_html_file(html_file)
            if parser.selected_service:
                self.details['selected_service'] = parser.selected_service
                if parser.selected_service.lower() == 'shared':
                    return True
        
        # If we reached here, check final state
        html_file = self.find_final_html()
        if html_file:
            parser = self.parse_html_file(html_file)
            if parser.selected_service:
                self.details['selected_service'] = parser.selected_service
                if parser.selected_service.lower() == 'shared':
                    return True
                else:
                    self.issues.append(f"Wrong service selected: '{parser.selected_service}' instead of 'Shared'")
            else:
                self.issues.append("No service type selected")
        
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Fare estimate viewed."""
        # Check trajectory for navigation to fare/booking page
        trajectory = self.get_trajectory()
        
        # Look for URLs that indicate fare viewing
        fare_related_urls = []
        for entry in trajectory:
            url = entry.get('url', '')
            if 'fare' in url.lower() or 'booking' in url.lower() or 'service' in url.lower():
                fare_related_urls.append(url)
        
        if fare_related_urls:
            self.details['fare_pages_visited'] = fare_related_urls
        
        # Also check HTML files for fare information
        all_html_files = self.find_step_html("step_*")
        if not all_html_files:
            html_file = self.find_final_html()
            if html_file:
                all_html_files = [html_file]
        
        for html_file in all_html_files:
            parser = self.parse_html_file(html_file)
            if parser.fare_visible and parser.fare_amount:
                self.details['fare_amount'] = parser.fare_amount
                return True
        
        # Check if agent navigated beyond the initial page
        if fare_related_urls:
            return True
        
        self.issues.append("Fare estimate not viewed - agent did not navigate to fare page")
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Ride not booked (no confirmation action taken)."""
        trajectory = self.get_trajectory()
        
        # Check for booking confirmation indicators
        for entry in trajectory:
            url = entry.get('url', '')
            # If we see "active" or "confirmation" in URL, ride was booked
            if 'active' in url.lower() or 'confirmation' in url.lower() or 'trip-active' in url.lower():
                self.issues.append(f"Ride was booked - found URL: {url}")
                return False
        
        # Check action history for "confirm" or "book" actions
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        for action_entry in action_history:
            action = action_entry.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            # Check if agent tried to confirm/book
            if 'confirm' in reasoning and 'trip' in reasoning:
                self.issues.append("Agent attempted to confirm trip")
                return False
            if 'book' in reasoning:
                self.issues.append("Agent attempted to book ride")
                return False
        
        # If we didn't find booking evidence, this checkpoint passes
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_pickup_entered'] = self.check_checkpoint_1()
            self.checkpoints['cp2_destination_entered'] = self.check_checkpoint_2()
            self.checkpoints['cp3_shared_selected'] = self.check_checkpoint_3()
            self.checkpoints['cp4_fare_viewed'] = self.check_checkpoint_4()
            self.checkpoints['cp5_not_booked'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 3,
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
                'query_id': 3,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python transportation_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
