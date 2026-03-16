#!/usr/bin/env python3
# Evaluator for calendar Query 14

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EventHTMLParser(HTMLParser):
    """Parse HTML to extract event information."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.current_event = {}
        self.in_event = False
        self.current_tag = None
        self.current_attrs = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for event divs with data-date attribute
        if tag == 'div' and 'data-date' in attrs_dict:
            self.in_event = True
            self.current_event = {
                'date': attrs_dict.get('data-date', ''),
                'title': '',
                'is_all_day': False
            }
        
        # Look for all-day indicators
        if self.in_event and 'class' in attrs_dict:
            if 'all-day' in attrs_dict['class'] or 'allday' in attrs_dict['class'].lower():
                self.current_event['is_all_day'] = True
        
        self.current_tag = tag
        self.current_attrs = attrs_dict

    def handle_data(self, data):
        if self.in_event and data.strip():
            text = data.strip()
            if text and not self.current_event.get('title'):
                self.current_event['title'] = text

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_event:
            if self.current_event.get('title'):
                self.events.append(self.current_event.copy())
            self.in_event = False
            self.current_event = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints
        self.checkpoints = {
            "cp1_new_button_clicked": False,
            "cp2_event_title_filled": False,
            "cp3_all_day_checkbox_checked": False,
            "cp4_date_set_to_2027": False,
            "cp5_event_created": False,
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
        """Checkpoint 1: New button was clicked"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if first action was clicking the New button
            if action_history and len(action_history) > 0:
                first_action = action_history[0]
                if first_action.get('action', {}).get('action_type') == 'CLICK':
                    selector = first_action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'New' in selector:
                        self.details['new_button_clicked'] = True
                        return True
            
            self.issues.append("New button was not clicked")
            return False
        except Exception as e:
            self.issues.append(f"Error checking New button: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Event title 'All Day' was filled"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if "All Day" was typed in the title field
            for action in action_history[:10]:  # Check first 10 actions
                if action.get('action', {}).get('action_type') == 'FILL':
                    text = action.get('action', {}).get('parameters', {}).get('text', '')
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if text == "All Day" and 'title' in selector.lower():
                        self.details['event_title'] = text
                        return True
            
            self.issues.append("Event title 'All Day' was not filled")
            return False
        except Exception as e:
            self.issues.append(f"Error checking event title: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: All-day event checkbox was checked"""
        try:
            # Check step 4 HTML where checkbox was clicked
            step_files = self.find_step_html("step_4_*")
            if not step_files:
                step_files = self.find_step_html("step_3_*")
            
            if step_files:
                with open(step_files[0], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for checked checkbox with all-day
                if 'type="checkbox"' in html_content and 'checked' in html_content:
                    self.details['all_day_checkbox'] = 'checked'
                    return True
                
                # Also check action history
                result_data = self.load_result_json()
                action_history = result_data.get('action_history', [])
                for action in action_history[:10]:
                    if action.get('action', {}).get('action_type') == 'CLICK':
                        selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                        if 'checkbox' in selector.lower():
                            return True
            
            self.issues.append("All-day checkbox was not checked")
            return False
        except Exception as e:
            self.issues.append(f"Error checking all-day checkbox: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Date was set to 2027-01-25"""
        try:
            # Check step 7-8 where date was entered
            step_files = self.find_step_html("step_[78]_*")
            
            if step_files:
                for step_file in step_files:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        
                    # Look for 2027-01-25 or 01/25/2027
                    if '2027-01-25' in html_content or '01/25/2027' in html_content:
                        self.details['date_set'] = '2027-01-25'
                        return True
            
            # Also check action history
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            for action in action_history[:15]:
                if action.get('action', {}).get('action_type') in ['FILL', 'TYPE']:
                    text = action.get('action', {}).get('parameters', {}).get('text', '')
                    if '2027' in text and '25' in text:
                        self.details['date_input'] = text
                        return True
            
            self.issues.append("Date was not set to 2027-01-25")
            return False
        except Exception as e:
            self.issues.append(f"Error checking date: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Event was created (Create button clicked and event exists)"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if Create Event button was clicked
            create_clicked = False
            for action in action_history[:15]:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Create' in selector and 'Event' in selector:
                        create_clicked = True
                        break
            
            if not create_clicked:
                self.issues.append("Create Event button was not clicked")
                return False
            
            # Check if event exists in any step HTML after creation
            step_files = self.find_step_html("step_[89]*")
            if not step_files:
                step_files = self.find_step_html("step_1*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Look for the event with title "All Day" and date 2027-01-25
                if 'All Day' in html_content and '2027-01-25' in html_content:
                    self.details['event_created'] = True
                    return True
            
            self.issues.append("Event 'All Day' on 2027-01-25 was not found in calendar")
            return False
        except Exception as e:
            self.issues.append(f"Error checking event creation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_new_button_clicked'] = self.check_checkpoint_1()
            self.checkpoints['cp2_event_title_filled'] = self.check_checkpoint_2()
            self.checkpoints['cp3_all_day_checkbox_checked'] = self.check_checkpoint_3()
            self.checkpoints['cp4_date_set_to_2027'] = self.check_checkpoint_4()
            self.checkpoints['cp5_event_created'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 14,
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
                'query_id': 14,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query14.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
