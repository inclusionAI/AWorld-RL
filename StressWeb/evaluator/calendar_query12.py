#!/usr/bin/env python3
# Evaluator for calendar Query 12

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class CalendarEventParser(HTMLParser):
    """Parse HTML to extract calendar events from the page state."""

    def __init__(self):
        super().__init__()
        self.in_script = False
        self.script_content = ''
        self.events = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False

    def handle_data(self, data):
        if self.in_script and 'window.__INITIAL_STATE__' in data:
            self.script_content += data

    def extract_events(self):
        """Extract events from the captured script content."""
        if not self.script_content:
            return {}
        
        try:
            # Find the JSON object in the script
            start = self.script_content.find('{')
            end = self.script_content.rfind('}') + 1
            if start == -1 or end == 0:
                return {}
            
            state = json.loads(self.script_content[start:end])
            self.events = state.get('events', {})
            return self.events
        except (json.JSONDecodeError, Exception):
            return {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_email_review_created": False,
            "cp2_team_sync_created": False,
            "cp3_focus_work_created": False,
            "cp4_all_on_jan_31": False,
            "cp5_all_work_category": False,
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

    def parse_events_from_html(self, html_path: Path) -> Dict:
        """Parse events from HTML file."""
        if not html_path.exists():
            return {}
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = CalendarEventParser()
        parser.feed(html_content)
        return parser.extract_events()

    def get_jan_31_events(self, events: Dict) -> List[Dict]:
        """Filter events for January 31, 2026."""
        jan_31_events = []
        for event_id, event in events.items():
            if event.get('start_date') == '2026-01-31':
                jan_31_events.append(event)
        
        # Sort by start time
        jan_31_events.sort(key=lambda x: x.get('start_time', ''))
        return jan_31_events

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Email Review event created (9:00-9:30)."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        events = self.parse_events_from_html(final_html)
        jan_31_events = self.get_jan_31_events(events)
        
        # Look for Email Review event
        email_review = None
        for event in jan_31_events:
            if event.get('title') == 'Email Review':
                email_review = event
                break
        
        if not email_review:
            self.issues.append("Email Review event not found")
            return False
        
        # Check time
        if email_review.get('start_time') != '09:00':
            self.issues.append(f"Email Review start time is {email_review.get('start_time')}, expected 09:00")
            return False
        
        if email_review.get('end_time') != '09:30':
            self.issues.append(f"Email Review end time is {email_review.get('end_time')}, expected 09:30")
            return False
        
        self.details['email_review'] = email_review
        return True

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Team Sync event created (9:30-10:00)."""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        events = self.parse_events_from_html(final_html)
        jan_31_events = self.get_jan_31_events(events)
        
        # Look for Team Sync event
        team_sync = None
        for event in jan_31_events:
            if event.get('title') == 'Team Sync':
                team_sync = event
                break
        
        if not team_sync:
            self.issues.append("Team Sync event not found")
            return False
        
        # Check time
        if team_sync.get('start_time') != '09:30':
            self.issues.append(f"Team Sync start time is {team_sync.get('start_time')}, expected 09:30")
            return False
        
        if team_sync.get('end_time') != '10:00':
            self.issues.append(f"Team Sync end time is {team_sync.get('end_time')}, expected 10:00")
            return False
        
        self.details['team_sync'] = team_sync
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Focus Work event created (10:00-11:00)."""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        events = self.parse_events_from_html(final_html)
        jan_31_events = self.get_jan_31_events(events)
        
        # Look for Focus Work event
        focus_work = None
        for event in jan_31_events:
            if event.get('title') == 'Focus Work':
                focus_work = event
                break
        
        if not focus_work:
            self.issues.append("Focus Work event not found")
            return False
        
        # Check time
        if focus_work.get('start_time') != '10:00':
            self.issues.append(f"Focus Work start time is {focus_work.get('start_time')}, expected 10:00")
            return False
        
        if focus_work.get('end_time') != '11:00':
            self.issues.append(f"Focus Work end time is {focus_work.get('end_time')}, expected 11:00")
            return False
        
        self.details['focus_work'] = focus_work
        return True

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: All three events are on January 31, 2026."""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        events = self.parse_events_from_html(final_html)
        jan_31_events = self.get_jan_31_events(events)
        
        # Count events with the required titles on Jan 31
        required_titles = {'Email Review', 'Team Sync', 'Focus Work'}
        found_titles = set()
        
        for event in jan_31_events:
            title = event.get('title')
            if title in required_titles:
                found_titles.add(title)
        
        self.details['jan_31_event_count'] = len(jan_31_events)
        self.details['found_titles'] = list(found_titles)
        
        if len(found_titles) != 3:
            self.issues.append(f"Only {len(found_titles)} of 3 required events found on Jan 31")
            return False
        
        return True

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: All three events have Work category."""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        events = self.parse_events_from_html(final_html)
        jan_31_events = self.get_jan_31_events(events)
        
        required_titles = {'Email Review', 'Team Sync', 'Focus Work'}
        work_category_id = 101  # Work category ID
        
        all_have_work_category = True
        
        for event in jan_31_events:
            title = event.get('title')
            if title in required_titles:
                category_id = event.get('category_id')
                if category_id != work_category_id:
                    self.issues.append(f"{title} does not have Work category (category_id={category_id})")
                    all_have_work_category = False
        
        return all_have_work_category

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_email_review_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_team_sync_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_focus_work_created'] = self.check_checkpoint_3()
            self.checkpoints['cp4_all_on_jan_31'] = self.check_checkpoint_4()
            self.checkpoints['cp5_all_work_category'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 12,
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
                'query_id': 12,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
