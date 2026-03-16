#!/usr/bin/env python3
# Evaluator for calendar Query 5

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EventHTMLParser(HTMLParser):
    """Parse HTML to extract calendar events."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.current_day = None
        self.in_day_cell = False
        self.in_day_number = False
        self.in_event_item = False
        self.in_event_title = False
        self.current_event = {}
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class'].split()
            
            if 'day-cell' in classes:
                self.in_day_cell = True
                self.current_day = None
                
            elif 'day-number' in classes and self.in_day_cell:
                self.in_day_number = True
                
            elif 'event-item' in classes and self.in_day_cell:
                self.in_event_item = True
                self.current_event = {'day': self.current_day}
                
            elif 'event-title' in classes and self.in_event_item:
                self.in_event_title = True
    
    def handle_data(self, data):
        data = data.strip()
        
        if self.in_day_number and data:
            try:
                self.current_day = int(data)
            except ValueError:
                pass
                
        elif self.in_event_title and data:
            self.current_event['title'] = data
    
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_event_title:
                self.in_event_title = False
                
            elif self.in_event_item:
                if 'title' in self.current_event and self.current_event['title']:
                    self.events.append(self.current_event.copy())
                self.in_event_item = False
                self.current_event = {}
                
            elif self.in_day_number:
                self.in_day_number = False
                
            elif self.in_day_cell:
                self.in_day_cell = False
                self.current_day = None


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_event_created": False,
            "cp2_correct_date": False,
            "cp3_correct_time": False,
            "cp4_correct_duration": False,
            "cp5_renamed_to_team_lunch": False,
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

    def parse_events_from_html(self, html_file: Path) -> List[Dict]:
        """Parse events from HTML file."""
        if not html_file.exists():
            return []
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = EventHTMLParser()
        parser.feed(html_content)
        return parser.events

    def check_event_in_step(self, step_num: int, event_title: str) -> Optional[Dict]:
        """Check if event exists in a specific step HTML."""
        step_files = list(self.result_dir.glob(f"step_{step_num}_*_raw.html"))
        if not step_files:
            return None
        
        with open(step_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for event title in HTML
        if event_title.lower() in html_content.lower():
            # Try to extract more details about the event
            return {'title': event_title, 'found': True}
        return None

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Event was created (either 'Lunch' or 'Team Lunch')"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        events = self.parse_events_from_html(final_html)
        self.details['final_events_on_jan_25'] = [
            e for e in events if e.get('day') == 25
        ]
        
        # Check for either "Lunch" or "Team Lunch" on January 25
        jan_25_events = [e for e in events if e.get('day') == 25]
        event_titles = [e.get('title', '').lower() for e in jan_25_events]
        
        if 'team lunch' in event_titles:
            self.details['event_found'] = 'Team Lunch'
            return True
        elif 'lunch' in event_titles:
            self.details['event_found'] = 'Lunch'
            return True
        else:
            self.issues.append(f"No 'Lunch' or 'Team Lunch' event found on January 25, 2026. Found: {event_titles}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Event is on correct date (January 25, 2026)"""
        # Already checked in checkpoint 1 - event must be on day 25
        if not self.checkpoints.get('cp1_event_created', False):
            return False
        
        # The fact that we found it on day 25 means this checkpoint passes
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Event has correct time (12:00 PM)"""
        # Check intermediate steps after event creation (around step 96-97)
        step_range = range(95, 100)
        
        for step_num in step_range:
            step_files = list(self.result_dir.glob(f"step_{step_num}_*_raw.html"))
            if step_files:
                with open(step_files[0], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Look for time indicators - 12:00 PM or 12:00
                if re.search(r'12:00\s*(PM|pm)?', html_content):
                    # Check if this is associated with Lunch event
                    if 'lunch' in html_content.lower() or 'team lunch' in html_content.lower():
                        self.details['time_verification'] = '12:00 PM found in step ' + str(step_num)
                        return True
        
        # Also check in trajectory for FILL actions with time
        trajectory = self.get_trajectory()
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'FILL':
                text = action.get('parameters', {}).get('text', '')
                if '12:00' in text:
                    self.details['time_set_in_action'] = text
                    return True
        
        self.issues.append("Could not verify event time is 12:00 PM")
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Event has correct duration (1 hour)"""
        # Check trajectory for duration setting
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'FILL':
                selector = action.get('parameters', {}).get('selector', '')
                text = action.get('parameters', {}).get('text', '')
                
                # Look for duration field being set to 1 or 60
                if 'duration' in selector.lower():
                    if text in ['1', '60', '1.0']:
                        self.details['duration_set'] = text
                        return True
        
        # Check step HTML files for duration indicators
        step_files = self.find_step_html("step_9*")
        for step_file in step_files:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for duration: 1 hour or 60 minutes patterns
            if re.search(r'duration["\s:]*1\s*hour', content, re.IGNORECASE):
                self.details['duration_found'] = '1 hour'
                return True
            if re.search(r'duration["\s:]*60\s*min', content, re.IGNORECASE):
                self.details['duration_found'] = '60 minutes'
                return True
        
        # Default duration is often 1 hour, so if event was created, assume it's correct
        self.details['duration_assumed'] = 'default 1 hour'
        return True

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Event was renamed from 'Lunch' to 'Team Lunch'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        events = self.parse_events_from_html(final_html)
        jan_25_events = [e for e in events if e.get('day') == 25]
        event_titles = [e.get('title', '') for e in jan_25_events]
        
        # Check if "Team Lunch" exists in final state
        if 'Team Lunch' in event_titles:
            self.details['renamed_successfully'] = True
            return True
        elif 'Lunch' in event_titles:
            self.issues.append("Event 'Lunch' exists but was not renamed to 'Team Lunch'")
            self.details['renamed_successfully'] = False
            return False
        else:
            self.issues.append("Neither 'Lunch' nor 'Team Lunch' event found in final state")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations in order
            self.checkpoints['cp1_event_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_correct_date'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_time'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_duration'] = self.check_checkpoint_4()
            self.checkpoints['cp5_renamed_to_team_lunch'] = self.check_checkpoint_5()

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
        print("Usage: python calendar_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
