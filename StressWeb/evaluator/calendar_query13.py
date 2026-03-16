#!/usr/bin/env python3
# Evaluator for calendar Query 13

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser
from datetime import datetime


class CalendarEventParser(HTMLParser):
    """Parse HTML to extract calendar event information."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.current_event = {}
        self.in_event = False
        self.current_tag = None
        self.current_attrs = {}
        self.capture_data = False
        self.data_buffer = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for event containers - these might be divs with specific classes or data attributes
        if tag == 'div':
            classes = attrs_dict.get('class', '')
            # Look for event-related classes
            if any(keyword in classes for keyword in ['event', 'calendar-event', 'schedule-item']):
                self.in_event = True
                self.current_event = {'attrs': attrs_dict}
        
        if self.in_event:
            self.current_tag = tag
            self.current_attrs = attrs_dict

    def handle_data(self, data):
        data = data.strip()
        if data and self.in_event:
            self.data_buffer.append(data)

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_event:
            # Store the event data
            if self.data_buffer:
                self.current_event['text'] = ' '.join(self.data_buffer)
                self.events.append(self.current_event)
            self.in_event = False
            self.current_event = {}
            self.data_buffer = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for this task
        self.checkpoints = {
            "cp1_week_view_opened": False,
            "cp2_lunch_break_found": False,
            "cp3_start_time_changed_to_11am": False,
            "cp4_duration_changed_to_90min": False,
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

    def extract_lunch_break_event_info(self, html_content: str) -> List[Dict]:
        """Extract Lunch Break event information from HTML."""
        events = []
        
        # Look for "Lunch Break" or "Lunch" events with time information
        # Pattern to find event divs with title and time
        patterns = [
            r'<div[^>]*>.*?Lunch Break.*?</div>',
            r'<div[^>]*>.*?Lunch\s*</div>',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, html_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                event_html = match.group(0)
                
                # Try to extract time information
                time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM)'
                time_matches = re.findall(time_pattern, event_html, re.IGNORECASE)
                
                event_info = {
                    'html': event_html,
                    'times': time_matches
                }
                events.append(event_info)
        
        # Also search for specific data in the HTML text
        # Look for January 29 events
        lines = html_content.split('\n')
        for i, line in enumerate(lines):
            if 'Lunch Break' in line or ('>Lunch<' in line and i > 0):
                # Check surrounding lines for time and date info
                context = '\n'.join(lines[max(0, i-5):min(len(lines), i+5)])
                time_matches = re.findall(r'(\d{1,2}):(\d{2})\s*(AM|PM)', context, re.IGNORECASE)
                if time_matches:
                    events.append({
                        'context': context,
                        'times': time_matches,
                        'line': line
                    })
        
        return events

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Week view was opened."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if there's a click action on Week button
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Week' in selector or 'week' in selector.lower():
                        self.details['week_view_clicked'] = True
                        return True
            
            # Also check if week view is visible in final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for week view indicators
                    if 'week' in content.lower() and any(day in content for day in ['MON', 'TUE', 'WED', 'THU', 'FRI']):
                        self.details['week_view_visible'] = True
                        return True
            
            self.issues.append("Week view was not opened")
            return False
        except Exception as e:
            self.issues.append(f"Error checking week view: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Lunch Break event on January 29 was found."""
        try:
            # Check trajectory for searches or clicks on Lunch Break
            trajectory = self.get_trajectory()
            found_lunch_break = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Lunch Break' in selector or 'Lunch' in selector:
                        found_lunch_break = True
                        self.details['lunch_break_clicked'] = True
                        break
            
            if found_lunch_break:
                return True
            
            self.issues.append("Lunch Break event was not found or clicked")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Lunch Break event: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Start time was changed to 11:00 AM."""
        try:
            # Check if the final HTML shows Lunch Break at 11:00 AM
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for Lunch Break event with 11:00 AM time
            # Search for patterns like "11:00 AM" near "Lunch Break"
            lunch_break_pattern = r'Lunch\s*Break'
            matches = list(re.finditer(lunch_break_pattern, content, re.IGNORECASE))
            
            if not matches:
                # Try just "Lunch" on January 29
                lunch_pattern = r'>Lunch<'
                matches = list(re.finditer(lunch_pattern, content))
            
            if matches:
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 500)
                    end = min(len(content), match.end() + 500)
                    context = content[start:end]
                    
                    # Look for time patterns
                    time_pattern = r'11:00\s*AM'
                    if re.search(time_pattern, context, re.IGNORECASE):
                        self.details['start_time_11am_found'] = True
                        return True
                    
                    # Also check for 11:00 in 24-hour format
                    if '11:00' in context and ('12:00' not in context or context.index('11:00') < context.index('12:00')):
                        self.details['start_time_11am_found'] = True
                        return True
            
            self.issues.append("Start time was not changed to 11:00 AM")
            return False
        except Exception as e:
            self.issues.append(f"Error checking start time: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Duration was changed to 90 minutes."""
        try:
            # Check if the final HTML shows Lunch Break with 90 minute duration
            # This means it should end at 12:30 PM if starting at 11:00 AM
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for Lunch Break event
            lunch_break_pattern = r'Lunch\s*Break'
            matches = list(re.finditer(lunch_break_pattern, content, re.IGNORECASE))
            
            if not matches:
                lunch_pattern = r'>Lunch<'
                matches = list(re.finditer(lunch_pattern, content))
            
            if matches:
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 800)
                    end = min(len(content), match.end() + 800)
                    context = content[start:end]
                    
                    # Look for time range patterns that indicate 90 minutes
                    # 11:00 AM - 12:30 PM would be 90 minutes
                    duration_patterns = [
                        r'11:00\s*AM\s*-\s*12:30\s*PM',
                        r'11:00\s*-\s*12:30',
                        r'90\s*min',
                        r'1\.5\s*hour',
                        r'1\s*hour\s*30\s*min'
                    ]
                    
                    for pattern in duration_patterns:
                        if re.search(pattern, context, re.IGNORECASE):
                            self.details['duration_90min_found'] = True
                            return True
            
            self.issues.append("Duration was not changed to 90 minutes")
            return False
        except Exception as e:
            self.issues.append(f"Error checking duration: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Check if agent claimed failure
            agent_success = result_data.get('success', False)
            if not agent_success:
                self.details['agent_failed'] = True
                self.details['agent_reason'] = result_data.get('reason', 'Unknown')
                self.details['agent_message'] = result_data.get('message', '')

            # Run checkpoint validations
            self.checkpoints['cp1_week_view_opened'] = self.check_checkpoint_1()
            self.checkpoints['cp2_lunch_break_found'] = self.check_checkpoint_2()
            self.checkpoints['cp3_start_time_changed_to_11am'] = self.check_checkpoint_3()
            self.checkpoints['cp4_duration_changed_to_90min'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 13,
                'query': result_data.get('query', ''),
                'overall_success': overall_success,
                'success_rate': success_rate,
                'checkpoints': self.checkpoints,
                'checkpoints_passed': f"{passed_count}/{total_count}",
                'issues': self.issues,
                'details': self.details,
                'agent_claimed_success': agent_success,
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
        print("Usage: python calendar_query13.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
