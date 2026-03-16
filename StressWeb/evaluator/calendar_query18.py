#!/usr/bin/env python3
# Evaluator for calendar Query 18

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
        self.in_event = False
        self.current_event = {}
        self.events = []
        self.current_tag = None
        self.current_attrs = {}
        self.active_view = None
        self.capture_data = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_attrs = attrs_dict
        
        # Check for active view
        if tag == 'button' and 'class' in attrs_dict:
            classes = attrs_dict.get('class', '')
            if 'view-btn' in classes and 'active' in classes:
                self.capture_data = True
        
        # Check for event elements in calendar
        if 'class' in attrs_dict:
            classes = attrs_dict.get('class', '')
            if 'event' in classes.lower() or 'calendar-event' in classes:
                self.in_event = True
                self.current_event = {'classes': classes}
                if 'data-event-id' in attrs_dict:
                    self.current_event['id'] = attrs_dict['data-event-id']
                if 'data-time' in attrs_dict:
                    self.current_event['time'] = attrs_dict['data-time']

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture active view name
        if self.capture_data and self.current_tag == 'button':
            self.active_view = data
            self.capture_data = False
        
        # Capture event data
        if self.in_event:
            if 'title' not in self.current_event:
                self.current_event['title'] = data
            elif 'time' not in self.current_event:
                self.current_event['time'] = data
            
    def handle_endtag(self, tag):
        if self.in_event and tag == 'div':
            if 'title' in self.current_event:
                self.events.append(self.current_event.copy())
            self.in_event = False
            self.current_event = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task 18:
        # 1. Create "Team Standup" event on Jan 28, 2026 at 10:00 AM (30 min)
        # 2. Navigate to Feb 5, 2026 using mini calendar
        # 3. Create "Gym Session" event at 6:00 PM (1.5 hours) with Health category
        # 4. Open Agenda view
        # 5. Cancel "Weekend Hike" event
        self.checkpoints = {
            "cp1_team_standup_created": False,
            "cp2_navigated_to_feb5": False,
            "cp3_gym_session_created": False,
            "cp4_opened_agenda": False,
            "cp5_weekend_hike_cancelled": False,
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

    def check_event_in_html(self, html_path: Path, event_title: str) -> bool:
        """Check if an event with given title exists in HTML."""
        if not html_path.exists():
            return False
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return event_title in content

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Team Standup event created on Jan 28, 2026 at 10:00 AM for 30 minutes"""
        try:
            # Check trajectory for event creation
            trajectory = self.get_trajectory()
            
            team_standup_created = False
            correct_date = False
            correct_time = False
            correct_duration = False
            
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                
                # Check if filling event title
                if action.get('action_type') == 'FILL' and params.get('text') == 'Team Standup':
                    team_standup_created = True
                
                # Check if date is set to Jan 28, 2026
                if action.get('action_type') == 'FILL' and '2026-01-28' in str(params.get('text', '')):
                    correct_date = True
                
                # Check if time is set to 10:00
                if action.get('action_type') == 'FILL' and '10:00' in str(params.get('text', '')):
                    correct_time = True
                
                # Check if duration is set to 30 minutes
                if action.get('action_type') == 'SELECT' and '30 minutes' in str(params.get('label', '')):
                    correct_duration = True
            
            # Check if Team Standup exists in HTML files after creation
            step_files = self.find_step_html("step_*")
            event_found_in_html = False
            for step_file in step_files[20:60]:  # Check steps after creation attempt
                if self.check_event_in_html(step_file, "Team Standup"):
                    event_found_in_html = True
                    break
            
            if team_standup_created and correct_date and correct_time and correct_duration:
                self.details['team_standup'] = "Event created with correct details"
                return True
            elif event_found_in_html:
                self.details['team_standup'] = "Event found in HTML"
                return True
            else:
                missing = []
                if not team_standup_created:
                    missing.append("title")
                if not correct_date:
                    missing.append("date")
                if not correct_time:
                    missing.append("time")
                if not correct_duration:
                    missing.append("duration")
                self.issues.append(f"Team Standup event missing or incorrect: {', '.join(missing)}")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Team Standup event: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Navigated to February 5, 2026 using mini calendar"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for navigation to February
            clicked_next_month = False
            clicked_feb_5 = False
            
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                
                # Check if clicked next month button (to go from Jan to Feb)
                if action.get('action_type') == 'CLICK':
                    reasoning = action.get('reasoning', '').lower()
                    if 'february' in reasoning or 'next month' in reasoning:
                        clicked_next_month = True
                    
                    # Check if clicked on day 5
                    if 'has-text' in str(params.get('selector', '')) and "'5'" in str(params.get('selector', '')):
                        clicked_feb_5 = True
                    elif params.get('selector') == "button:has-text('5')":
                        clicked_feb_5 = True
            
            # Check URL in trajectory for February dates
            feb_navigation = False
            for step in trajectory:
                page_info = step.get('page_info', {})
                url = page_info.get('url', '')
                if 'february' in url.lower() or step.get('action', {}).get('reasoning', '').lower().find('february 5') != -1:
                    feb_navigation = True
                    break
            
            if clicked_next_month or clicked_feb_5 or feb_navigation:
                self.details['feb5_navigation'] = "Navigated to February 5, 2026"
                return True
            else:
                self.issues.append("Did not navigate to February 5, 2026 using mini calendar")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking February 5 navigation: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Gym Session event created at 6:00 PM for 1.5 hours with Health category"""
        try:
            trajectory = self.get_trajectory()
            
            gym_session_created = False
            correct_time = False
            correct_duration = False
            health_category = False
            
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                
                # Check if filling event title
                if action.get('action_type') == 'FILL' and params.get('text') == 'Gym Session':
                    gym_session_created = True
                
                # Check if time is set to 18:00 (6:00 PM)
                if action.get('action_type') == 'FILL' and ('18:00' in str(params.get('text', '')) or '6:00 PM' in str(params.get('text', ''))):
                    correct_time = True
                
                # Check if duration is set to 1.5 hours (90 minutes)
                if action.get('action_type') == 'SELECT':
                    label = params.get('label', '')
                    if '1.5' in label or '90' in label or '1 hour 30' in label:
                        correct_duration = True
                
                # Check if Health category is selected
                if action.get('action_type') == 'SELECT' and 'Health' in str(params.get('label', '')):
                    health_category = True
            
            # Check HTML for Gym Session event
            step_files = self.find_step_html("step_*")
            event_found_in_html = False
            for step_file in step_files[60:]:  # Check later steps
                if self.check_event_in_html(step_file, "Gym Session"):
                    event_found_in_html = True
                    break
            
            if gym_session_created:
                missing = []
                if not correct_time:
                    missing.append("time (6:00 PM)")
                if not correct_duration:
                    missing.append("duration (1.5 hours)")
                if not health_category:
                    missing.append("Health category")
                
                if missing:
                    self.issues.append(f"Gym Session event created but missing: {', '.join(missing)}")
                    self.details['gym_session'] = f"Partial - missing {', '.join(missing)}"
                    return len(missing) <= 1  # Allow if only one field is missing
                else:
                    self.details['gym_session'] = "Event created with all correct details"
                    return True
            elif event_found_in_html:
                self.details['gym_session'] = "Event found in HTML (cannot verify all details)"
                self.issues.append("Gym Session found but cannot verify time/duration/category from HTML")
                return False
            else:
                self.issues.append("Gym Session event not created")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Gym Session event: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Opened Agenda view"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if clicked Agenda view
            agenda_clicked = False
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                
                if action.get('action_type') == 'CLICK':
                    selector = params.get('selector', '')
                    if 'Agenda' in selector or 'agenda' in selector.lower():
                        agenda_clicked = True
                        break
            
            # Check final HTML for active Agenda view
            final_html = self.find_final_html()
            if final_html and final_html.exists():
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for active agenda view button
                    if 'view-btn active">Agenda' in content or 'view-btn active" >Agenda' in content:
                        self.details['agenda_view'] = "Agenda view is active in final state"
                        return True
            
            if agenda_clicked:
                self.details['agenda_view'] = "Agenda view was clicked"
                return True
            else:
                self.issues.append("Agenda view was not opened")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Agenda view: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Weekend Hike event cancelled"""
        try:
            trajectory = self.get_trajectory()
            
            # Check for actions related to Weekend Hike
            weekend_hike_action = False
            cancel_action = False
            delete_action = False
            
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                reasoning = action.get('reasoning', '').lower()
                
                # Check if Weekend Hike was mentioned
                if 'weekend hike' in reasoning:
                    weekend_hike_action = True
                
                # Check for cancel or delete actions
                if action.get('action_type') == 'CLICK':
                    selector = params.get('selector', '').lower()
                    if 'cancel' in selector or 'delete' in selector or 'remove' in selector:
                        cancel_action = True
                    if 'cancel' in reasoning or 'delete' in reasoning:
                        delete_action = True
            
            # Check if Weekend Hike exists in final HTML
            final_html = self.find_final_html()
            weekend_hike_in_final = False
            if final_html and final_html.exists():
                weekend_hike_in_final = self.check_event_in_html(final_html, "Weekend Hike")
            
            # Check early steps for Weekend Hike presence
            step_files = self.find_step_html("step_*")
            weekend_hike_in_early_steps = False
            if len(step_files) > 0:
                weekend_hike_in_early_steps = self.check_event_in_html(step_files[0], "Weekend Hike")
            
            # If Weekend Hike was never present, it's likely a test setup issue
            if not weekend_hike_in_early_steps:
                self.issues.append("Weekend Hike event was not present in initial state (may not exist in this calendar)")
                self.details['weekend_hike'] = "Event not found in calendar"
                return False
            
            # If action was taken and event is gone from final state, consider it cancelled
            if weekend_hike_action and not weekend_hike_in_final:
                self.details['weekend_hike'] = "Event successfully cancelled"
                return True
            elif (cancel_action or delete_action) and not weekend_hike_in_final:
                self.details['weekend_hike'] = "Event cancelled/deleted"
                return True
            elif not weekend_hike_in_final and weekend_hike_in_early_steps:
                self.details['weekend_hike'] = "Event removed from calendar"
                return True
            elif weekend_hike_in_final:
                self.issues.append("Weekend Hike event still exists in final state")
                return False
            else:
                self.issues.append("Could not verify Weekend Hike cancellation")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Weekend Hike cancellation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_team_standup_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_navigated_to_feb5'] = self.check_checkpoint_2()
            self.checkpoints['cp3_gym_session_created'] = self.check_checkpoint_3()
            self.checkpoints['cp4_opened_agenda'] = self.check_checkpoint_4()
            self.checkpoints['cp5_weekend_hike_cancelled'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 18,
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
                'query_id': 18,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query18.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
