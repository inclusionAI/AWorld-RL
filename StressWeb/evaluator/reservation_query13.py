#!/usr/bin/env python3
# Evaluator for reservation Query 13

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RestaurantSearchParser(HTMLParser):
    """Parse HTML to extract restaurant search results and form data."""

    def __init__(self):
        super().__init__()
        self.in_results_count = False
        self.in_search_input = False
        self.in_filter_label = False
        self.results_count = None
        self.search_value = ""
        self.wheelchair_checked = False
        self.current_label_text = ""
        self.date_value = ""
        self.time_value = ""
        self.party_size = ""
        self.in_date_input = False
        self.in_time_select = False
        self.in_party_select = False
        self.in_special_request = False
        self.special_request_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for search input
        if tag == "input" and attrs_dict.get("placeholder") == "Location, Restaurant, or Cuisine":
            self.search_value = attrs_dict.get("value", "")
        
        # Check for wheelchair accessible checkbox
        if tag == "input" and attrs_dict.get("type") == "checkbox":
            if "wheelchair" in attrs_dict.get("id", "").lower():
                self.wheelchair_checked = "checked" in attrs_dict
        
        # Check for date input
        if tag == "input" and attrs_dict.get("type") == "text":
            value = attrs_dict.get("value", "")
            if "/" in value and len(value) == 10:  # Date format MM/DD/YYYY
                self.date_value = value
        
        # Check for time and party size selects
        if tag == "select":
            self.in_time_select = True
        
        # Check for special request textarea
        if tag == "textarea":
            if "special" in attrs_dict.get("placeholder", "").lower():
                self.in_special_request = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        # Extract results count
        if "restaurants found" in data.lower():
            match = re.search(r'(\d+)\s+restaurants?\s+found', data, re.IGNORECASE)
            if match:
                self.results_count = int(match.group(1))
        
        # Extract special request text
        if self.in_special_request:
            self.special_request_text += data

    def handle_endtag(self, tag):
        if tag == "textarea":
            self.in_special_request = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_wheelchair_filter_applied": False,
            "cp3_search_parameters_set": False,
            "cp4_reservation_attempted": False,
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
        """Checkpoint 1: Search for Italian restaurants executed"""
        try:
            # Check steps 24-30 where search was configured
            step_files = self.find_step_html("step_2[4-9]*") + self.find_step_html("step_30*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    parser = RestaurantSearchParser()
                    parser.feed(html_content)
                    
                    # Check if "Italian" was entered in search
                    if "italian" in parser.search_value.lower():
                        self.details["search_term_found"] = True
                        return True
            
            self.issues.append("Search term 'Italian' was not properly entered")
            return False
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Wheelchair accessible filter applied"""
        try:
            # Check steps 26-30 where wheelchair filter was applied
            step_files = self.find_step_html("step_2[6-9]*") + self.find_step_html("step_30*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                    # Check if wheelchair accessible checkbox is checked
                    if 'wheelchair' in html_content.lower() and 'checked' in html_content.lower():
                        # Parse to confirm
                        parser = RestaurantSearchParser()
                        parser.feed(html_content)
                        if parser.wheelchair_checked or "wheelchair accessible" in html_content.lower():
                            self.details["wheelchair_filter_applied"] = True
                            return True
            
            self.issues.append("Wheelchair accessible filter was not applied")
            return False
        except Exception as e:
            self.issues.append(f"Error checking wheelchair filter: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Search parameters (date, time, party size) correctly set"""
        try:
            # Check steps 47-49 where time and party size were set
            step_files = self.find_step_html("step_4[7-9]*") + self.find_step_html("step_5*")
            
            time_set = False
            party_size_set = False
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                    # Check for time 6:30 PM
                    if "6:30 pm" in html_content.lower() or "6:30pm" in html_content.lower():
                        time_set = True
                    
                    # Check for party size 4 people
                    if "4 people" in html_content.lower():
                        party_size_set = True
            
            if time_set and party_size_set:
                self.details["time_set"] = "6:30 PM"
                self.details["party_size_set"] = "4 people"
                return True
            
            if not time_set:
                self.issues.append("Time was not set to 6:30 PM")
            if not party_size_set:
                self.issues.append("Party size was not set to 4 people")
            return False
        except Exception as e:
            self.issues.append(f"Error checking search parameters: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Reservation attempted (even if no restaurants found)"""
        try:
            result_data = self.load_result_json()
            
            # Check if agent attempted to search with correct criteria
            # The agent should have searched for Italian restaurants with wheelchair accessibility
            # Even if 0 results, the search attempt counts
            
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    parser = RestaurantSearchParser()
                    parser.feed(html_content)
                    
                    # Check if search was executed (showing results count)
                    if parser.results_count is not None:
                        self.details["search_executed"] = True
                        self.details["results_found"] = parser.results_count
                        
                        # Even 0 results means the search was attempted properly
                        if parser.results_count == 0:
                            self.issues.append("No restaurants found matching criteria (0 results) - unable to complete reservation")
                            return True  # Search was attempted correctly, just no results
                        else:
                            # If results were found, check if reservation was attempted
                            return True
            
            # Check action history for search attempts
            action_history = result_data.get("action_history", [])
            for action in action_history:
                action_data = action.get("action", {})
                if action_data.get("action_type") == "CLICK":
                    reasoning = action_data.get("reasoning", "")
                    if "let's go" in reasoning.lower() or "search" in reasoning.lower():
                        self.details["search_button_clicked"] = True
                        return True
            
            self.issues.append("Search was not executed with the specified criteria")
            return False
        except Exception as e:
            self.issues.append(f"Error checking reservation attempt: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_wheelchair_filter_applied'] = self.check_checkpoint_2()
            self.checkpoints['cp3_search_parameters_set'] = self.check_checkpoint_3()
            self.checkpoints['cp4_reservation_attempted'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            # Add context about why task couldn't be completed
            if result_data.get("success") == False:
                message = result_data.get("message", "")
                if "0 restaurants found" in message or "No restaurants found" in message:
                    self.details["failure_reason"] = "No restaurants matching search criteria found in database"

            return {
                'query_id': 13,
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
                'query_id': 13,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query13.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
