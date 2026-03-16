#!/usr/bin/env python3
# Evaluator for calendar Query 15

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class AgendaHTMLParser(HTMLParser):
    """Parse HTML to extract agenda events information."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.current_event = {}
        self.in_event_title = False
        self.in_event_time = False
        self.in_event_location = False
        self.in_event_description = False
        self.in_date_title = False
        self.current_date = None
        self.in_detail_location = False
        self.in_detail_value = False
        self.found_location_label = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'h3' and any(c[1] == 'date-title' for c in attrs):
            self.in_date_title = True
        elif tag == 'h4' and any(c[1] == 'event-title-agenda' for c in attrs):
            self.in_event_title = True
            self.current_event = {'date': self.current_date}
        elif tag == 'span' and any(c[1] == 'event-time-agenda' for c in attrs):
            self.in_event_time = True
        elif tag == 'div' and any(c[1] == 'event-location-agenda' for c in attrs):
            self.in_event_location = True
        elif tag == 'div' and any(c[1] == 'event-description-agenda' for c in attrs):
            self.in_event_description = True
        # For event detail view
        elif tag == 'div' and 'detail-label' in attrs_dict.get('class', ''):
            self.in_detail_value = False
        elif tag == 'div' and 'detail-value' in attrs_dict.get('class', ''):
            if self.found_location_label:
                self.in_detail_location = True
                self.found_location_label = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_date_title:
            self.current_date = data
            self.in_date_title = False
        elif self.in_event_title:
            self.current_event['title'] = data
            self.in_event_title = False
        elif self.in_event_time:
            self.current_event['time'] = data
            self.in_event_time = False
        elif self.in_event_location:
            # Remove the icon text
            if not data.startswith('fa-'):
                self.current_event['location'] = data
        elif self.in_event_description:
            self.current_event['description'] = data
            self.in_event_description = False
            self.events.append(self.current_event)
            self.current_event = {}
        elif data == '📍 LOCATION':
            self.found_location_label = True
        elif self.in_detail_location:
            self.current_event['location'] = data
            self.in_detail_location = False

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_event_location:
                self.in_event_location = False
                # If no description follows, save event
                if self.current_event and 'title' in self.current_event:
                    if 'description' not in self.current_event:
                        self.events.append(self.current_event)
                        self.current_event = {}
            elif self.in_detail_location:
                self.in_detail_location = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_agenda": False,
            "cp2_identified_second_to_last_event": False,
            "cp3_opened_event_for_editing": False,
            "cp4_location_changed_to_home": False,
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

    def parse_agenda_events(self, html_file: Path) -> List[Dict]:
        """Parse agenda events from HTML file."""
        parser = AgendaHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            parser.feed(html_content)
        return parser.events

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Agenda view"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if there was a click on Agenda
            for action in action_history:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Agenda' in selector:
                        self.details['agenda_navigation'] = 'Found Agenda navigation'
                        return True
            
            self.issues.append("Did not navigate to Agenda view")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Agenda navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Identified second-to-last event (Product Demo)"""
        try:
            # Find agenda view HTML files (before clicking on event)
            agenda_files = []
            for step_file in sorted(self.result_dir.glob("step_*_raw.html")):
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'agenda-view' in content and 'Product Demo' in content and 'Family Dinner' in content:
                        agenda_files.append(step_file)
            
            if not agenda_files:
                self.issues.append("Could not find agenda view with all events")
                return False
            
            # Parse the last agenda view to verify event order
            events = self.parse_agenda_events(agenda_files[-1])
            
            # Find Product Demo and Family Dinner positions
            product_demo_idx = -1
            family_dinner_idx = -1
            
            for idx, event in enumerate(events):
                if 'Product Demo' in event.get('title', ''):
                    product_demo_idx = idx
                if 'Family Dinner' in event.get('title', ''):
                    family_dinner_idx = idx
            
            if product_demo_idx == -1:
                self.issues.append("Product Demo event not found in agenda")
                return False
            
            if family_dinner_idx == -1:
                self.issues.append("Family Dinner event not found in agenda")
                return False
            
            # Check if Family Dinner is the last event and Product Demo is second-to-last
            if family_dinner_idx == len(events) - 1 and product_demo_idx == len(events) - 2:
                self.details['second_to_last_event'] = 'Product Demo (correct)'
                self.details['last_event'] = 'Family Dinner'
                return True
            else:
                self.issues.append(f"Product Demo is not second-to-last event. Found at position {product_demo_idx+1}/{len(events)}, expected {len(events)-1}/{len(events)}")
                return False
                
        except Exception as e:
            self.issues.append(f"Error identifying second-to-last event: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Opened Product Demo event for editing"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if Product Demo was clicked
            product_demo_clicked = False
            edit_clicked = False
            
            for action in action_history:
                action_type = action.get('action', {}).get('action_type')
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                
                if action_type == 'CLICK' and 'Product Demo' in selector:
                    product_demo_clicked = True
                
                if action_type == 'CLICK' and 'Edit' in selector:
                    edit_clicked = True
            
            if product_demo_clicked and edit_clicked:
                self.details['event_editing'] = 'Product Demo opened for editing'
                return True
            elif product_demo_clicked:
                self.issues.append("Product Demo was clicked but not opened for editing")
                return False
            else:
                self.issues.append("Product Demo was not clicked")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking event editing: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Location changed to 'home'"""
        try:
            # Check the final HTML state
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML not found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if we're viewing Product Demo event detail
            if 'Product Demo' not in html_content:
                self.issues.append("Final state is not showing Product Demo event")
                return False
            
            # Parse location from detail view
            parser = AgendaHTMLParser()
            parser.feed(html_content)
            
            # Look for location in HTML - use regex for more reliable extraction
            location_pattern = r'📍 LOCATION</div><div class="detail-value">([^<]+)</div>'
            match = re.search(location_pattern, html_content)
            
            if match:
                location = match.group(1).strip()
                self.details['final_location'] = location
                
                if location.lower() == 'home':
                    self.details['location_change'] = 'Successfully changed to home'
                    return True
                else:
                    self.issues.append(f"Location is '{location}', expected 'home'")
                    return False
            else:
                # Try alternative pattern
                if '<div class="detail-value">home</div>' in html_content:
                    self.details['final_location'] = 'home'
                    self.details['location_change'] = 'Successfully changed to home'
                    return True
                
                self.issues.append("Could not find location field in Product Demo event")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking location change: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_agenda'] = self.check_checkpoint_1()
            self.checkpoints['cp2_identified_second_to_last_event'] = self.check_checkpoint_2()
            self.checkpoints['cp3_opened_event_for_editing'] = self.check_checkpoint_3()
            self.checkpoints['cp4_location_changed_to_home'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 15,
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
                'query_id': 15,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query15.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
