#!/usr/bin/env python3
# Evaluator for calendar Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EventHTMLParser(HTMLParser):
    """Parse HTML to extract calendar events and current view state."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.current_event = {}
        self.in_event_title = False
        self.in_event_time = False
        self.capture_text = False
        self.current_view = None
        self.current_date_display = None
        self.in_header_date = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for active view button
        if tag == 'button':
            classes = attrs_dict.get('class', '')
            if 'view-btn' in classes and 'active' in classes:
                # Next data will be the view name
                self.capture_text = True
                self.view_type = 'view'
        
        # Check for header date display
        if 'class' in attrs_dict and 'header-date' in attrs_dict.get('class', ''):
            self.in_header_date = True
        
        # Check for event items in week view
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict.get('class', '')
            if 'week-event' in classes or 'event-block' in classes:
                self.current_event = {'attrs': classes}
            elif 'event-title' in classes and self.current_event:
                self.in_event_title = True
            elif 'event-time' in classes and self.current_event:
                self.in_event_time = True
                
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.capture_text and hasattr(self, 'view_type') and self.view_type == 'view':
            self.current_view = data
            self.capture_text = False
            delattr(self, 'view_type')
            
        if self.in_header_date:
            self.current_date_display = data
            
        if self.in_event_title:
            self.current_event['title'] = data
        elif self.in_event_time:
            self.current_event['time'] = data
    
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_event_title:
                self.in_event_title = False
            elif self.in_event_time:
                self.in_event_time = False
            elif self.current_event and ('title' in self.current_event or 'time' in self.current_event):
                if 'title' in self.current_event:
                    self.events.append(self.current_event.copy())
                self.current_event = {}
        
        if tag == 'h2' and self.in_header_date:
            self.in_header_date = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for this task
        self.checkpoints = {
            "cp1_switched_to_week_view": False,
            "cp2_navigated_to_february": False,
            "cp3_found_team_standup_feb2": False,
            "cp4_event_moved_to_feb5_11am": False,
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

    def check_switched_to_week_view(self) -> bool:
        """Check if agent switched to week view."""
        # Check in trajectory if Week button was clicked
        trajectory = self.get_trajectory()
        
        for step in trajectory:
            action = step.get('action', {})
            params = action.get('parameters', {})
            selector = params.get('selector', '')
            
            if 'Week' in selector or 'week' in selector.lower():
                self.details['week_view_clicked'] = True
                return True
                
        # Also check intermediate steps to see if week view was reached
        step_files = self.find_step_html("step_*")
        for step_file in step_files[:20]:  # Check first 20 steps
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'class="week-view"' in content or 'Week view' in content:
                    if 'view-btn active' in content:
                        # Check if Week is the active button
                        parser = EventHTMLParser()
                        parser.feed(content)
                        if parser.current_view and 'Week' in parser.current_view:
                            self.details['reached_week_view'] = True
                            return True
        
        self.issues.append("Did not switch to week view")
        return False

    def check_navigated_to_february(self) -> bool:
        """Check if agent navigated to February 2026."""
        # Check trajectory for navigation to February
        step_files = self.find_step_html("step_*")
        
        for step_file in step_files:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for February 2026 in the header
                if 'February 2026' in content or 'Feb 2026' in content:
                    self.details['reached_february'] = True
                    return True
        
        self.issues.append("Did not navigate to February 2026")
        return False

    def check_found_team_standup_feb2(self) -> bool:
        """Check if Team Standup event on Feb 2 was found and opened for editing."""
        # Check trajectory for clicking on Team Standup event
        trajectory = self.get_trajectory()
        
        found_event_interaction = False
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            if 'team standup' in reasoning and ('february 2' in reasoning or 'feb 2' in reasoning):
                found_event_interaction = True
                self.details['interacted_with_team_standup'] = True
                break
        
        # Also check if edit modal was opened
        step_files = self.find_step_html("step_*")
        for step_file in step_files[20:]:  # Check after initial navigation
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for edit modal with Team Standup
                if 'modal' in content.lower() and 'Team Standup' in content:
                    self.details['opened_edit_modal'] = True
                    return True
        
        if found_event_interaction:
            return True
        
        self.issues.append("Did not find or interact with Team Standup event on Feb 2")
        return False

    def check_event_moved_to_feb5_11am(self) -> bool:
        """Check if Team Standup event was moved to Feb 5 at 11:00 AM."""
        # Check final HTML and late-stage intermediate steps for the moved event
        step_files = self.find_step_html("step_*")
        final_html = self.find_final_html()
        
        files_to_check = []
        if final_html:
            files_to_check.append(final_html)
        # Also check last 20 steps
        files_to_check.extend(step_files[-20:])
        
        for html_file in files_to_check:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Look for Team Standup event on Feb 5 with 11:00 AM time
                # The event should appear in context of February 5
                if 'Team Standup' in content:
                    # Parse to find event details
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'Team Standup' in line:
                            # Check surrounding context for date and time
                            context = '\n'.join(lines[max(0, i-10):min(len(lines), i+10)])
                            
                            # Check for Feb 5 or February 5
                            has_feb5 = ('feb 5' in context.lower() or 'february 5' in context.lower() 
                                       or 'feb-5' in context.lower() or '2/5' in context
                                       or 'Feb 5' in context or 'February 5' in context)
                            
                            # Check for 11:00 AM or 11 AM
                            has_11am = ('11:00 am' in context.lower() or '11 am' in context.lower() 
                                       or '11:00am' in context.lower() or '11am' in context.lower())
                            
                            if has_feb5 and has_11am:
                                self.details['team_standup_moved'] = True
                                self.details['new_date'] = 'February 5, 2026'
                                self.details['new_time'] = '11:00 AM'
                                return True
        
        # Check trajectory for save actions
        trajectory = self.get_trajectory()
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            if 'save' in reasoning and 'team standup' in reasoning:
                self.details['attempted_save'] = True
        
        self.issues.append("Team Standup event was not successfully moved to Feb 5 at 11:00 AM")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations in order
            self.checkpoints['cp1_switched_to_week_view'] = self.check_switched_to_week_view()
            self.checkpoints['cp2_navigated_to_february'] = self.check_navigated_to_february()
            self.checkpoints['cp3_found_team_standup_feb2'] = self.check_found_team_standup_feb2()
            self.checkpoints['cp4_event_moved_to_feb5_11am'] = self.check_event_moved_to_feb5_11am()

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
        print("Usage: python calendar_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
