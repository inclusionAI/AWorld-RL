#!/usr/bin/env python3
# Evaluator for calendar Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract calendar information."""

    def __init__(self):
        super().__init__()
        self.current_month = None
        self.current_view = None
        self.events = []
        self.in_mini_calendar = False
        self.in_mini_month_title = False
        self.in_event_title = False
        self.current_event = {}
        self.in_header_date = False
        self.view_active = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect mini calendar month
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'mini-month-title' in classes:
                self.in_mini_month_title = True
            elif 'header-date' in classes:
                self.in_header_date = True
            elif 'event-title' in classes:
                self.in_event_title = True
                self.current_event = {}
            elif 'view-btn' in classes and 'active' in classes:
                # Next data will be the active view
                self.view_active = True

    def handle_data(self, data):
        data = data.strip()
        if self.in_mini_month_title and data:
            self.current_month = data
            self.in_mini_month_title = False
        elif self.in_header_date and data:
            if not self.current_month:
                self.current_month = data
            self.in_header_date = False
        elif self.in_event_title and data:
            self.events.append(data)
            self.in_event_title = False
        elif self.view_active and data:
            self.current_view = data
            self.view_active = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task:
        # Task: Use the small calendar on the left to navigate to February 5, 2026, 
        # and create a "Gym Session" event at 6:00 PM lasting 1.5 hours with category Health.
        self.checkpoints = {
            "cp1_navigated_to_february": False,
            "cp2_navigated_to_feb_5": False,
            "cp3_gym_session_created": False,
            "cp4_correct_time_6pm": False,
            "cp5_correct_duration_90min": False,
            "cp6_health_category": False,
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

    def parse_html_file(self, html_file: Path) -> TaskHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_navigation_to_february(self) -> bool:
        """Check if agent navigated to February 2026 at any point."""
        all_steps = self.find_step_html("step_*")
        
        for step_file in all_steps:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'February 2026' in content:
                    self.details['navigated_to_february'] = True
                    self.details['february_detected_at'] = step_file.name
                    return True
        
        self.issues.append("Never navigated to February 2026")
        self.details['navigated_to_february'] = False
        return False

    def check_navigation_to_feb_5(self) -> bool:
        """Check if agent navigated to or focused on February 5, 2026."""
        all_steps = self.find_step_html("step_*")
        
        for step_file in all_steps:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for February 2026 context and date selection
                if 'February 2026' in content:
                    # Look for date "5" selection or data-date="2026-02-05"
                    if 'data-date="2026-02-05"' in content or \
                       ('selected' in content and re.search(r'title="[^"]*2/5/2026[^"]*"', content)):
                        self.details['navigated_to_feb_5'] = True
                        self.details['feb_5_detected_at'] = step_file.name
                        return True
        
        # If we got to February but not specifically Feb 5
        if self.details.get('navigated_to_february'):
            self.issues.append("Navigated to February but not specifically to Feb 5, 2026")
        else:
            self.issues.append("Never navigated to February 5, 2026")
        
        self.details['navigated_to_feb_5'] = False
        return False

    def check_gym_session_created(self) -> bool:
        """Check if 'Gym Session' event was created."""
        all_steps = self.find_step_html("step_*")
        final_html = self.find_final_html()
        
        # Check in all steps for the event
        files_to_check = all_steps + ([final_html] if final_html else [])
        
        for step_file in files_to_check:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'Gym Session' in content:
                    self.details['gym_session_created'] = True
                    self.details['gym_session_detected_at'] = step_file.name
                    return True
        
        self.issues.append("Event 'Gym Session' was not created")
        self.details['gym_session_created'] = False
        return False

    def check_event_time_6pm(self) -> bool:
        """Check if event is at 6:00 PM (18:00)."""
        all_steps = self.find_step_html("step_*")
        final_html = self.find_final_html()
        
        files_to_check = all_steps + ([final_html] if final_html else [])
        
        for step_file in files_to_check:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for Gym Session event with time information
                # Common patterns: "18:00", "6:00 PM", "18"
                if 'Gym Session' in content:
                    # Check for time patterns near Gym Session
                    # Look for data-start-time, start time in JSON, or visible time display
                    time_patterns = [
                        r'"startTime"\s*:\s*"18:00"',
                        r'"startTime"\s*:\s*"6:00 PM"',
                        r'data-start-time="18:00"',
                        r'data-start-time="18"',
                        r'>18:00<',
                        r'>6:00 PM<',
                    ]
                    
                    for pattern in time_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            self.details['event_time_6pm'] = True
                            self.details['time_detected_at'] = step_file.name
                            return True
        
        if self.details.get('gym_session_created'):
            self.issues.append("Gym Session event created but not at 6:00 PM")
        else:
            self.issues.append("Cannot verify time - event not created")
        
        self.details['event_time_6pm'] = False
        return False

    def check_event_duration_90min(self) -> bool:
        """Check if event duration is 1.5 hours (90 minutes)."""
        all_steps = self.find_step_html("step_*")
        final_html = self.find_final_html()
        
        files_to_check = all_steps + ([final_html] if final_html else [])
        
        for step_file in files_to_check:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'Gym Session' in content:
                    # Look for duration patterns
                    duration_patterns = [
                        r'"duration"\s*:\s*90',
                        r'"duration"\s*:\s*"90"',
                        r'data-duration="90"',
                        r'"endTime"\s*:\s*"19:30"',  # 6pm + 1.5h = 7:30pm
                        r'"endTime"\s*:\s*"7:30 PM"',
                        r'1\.5\s*hours?',
                        r'90\s*min',
                    ]
                    
                    for pattern in duration_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            self.details['event_duration_90min'] = True
                            self.details['duration_detected_at'] = step_file.name
                            return True
        
        if self.details.get('gym_session_created'):
            self.issues.append("Gym Session created but duration not 1.5 hours")
        else:
            self.issues.append("Cannot verify duration - event not created")
        
        self.details['event_duration_90min'] = False
        return False

    def check_health_category(self) -> bool:
        """Check if event has Health category."""
        all_steps = self.find_step_html("step_*")
        final_html = self.find_final_html()
        
        files_to_check = all_steps + ([final_html] if final_html else [])
        
        for step_file in files_to_check:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'Gym Session' in content:
                    # Look for Health category association
                    # Check for category field, color associated with Health (red: rgb(239, 68, 68))
                    category_patterns = [
                        r'"category"\s*:\s*"Health"',
                        r'data-category="Health"',
                        r'Gym Session.*?rgb\(239,\s*68,\s*68\)',  # Health category color
                    ]
                    
                    for pattern in category_patterns:
                        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                            self.details['health_category'] = True
                            self.details['category_detected_at'] = step_file.name
                            return True
        
        if self.details.get('gym_session_created'):
            self.issues.append("Gym Session created but not categorized as Health")
        else:
            self.issues.append("Cannot verify category - event not created")
        
        self.details['health_category'] = False
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_february'] = self.check_navigation_to_february()
            self.checkpoints['cp2_navigated_to_feb_5'] = self.check_navigation_to_feb_5()
            self.checkpoints['cp3_gym_session_created'] = self.check_gym_session_created()
            self.checkpoints['cp4_correct_time_6pm'] = self.check_event_time_6pm()
            self.checkpoints['cp5_correct_duration_90min'] = self.check_event_duration_90min()
            self.checkpoints['cp6_health_category'] = self.check_health_category()

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
        print("Usage: python calendar_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
