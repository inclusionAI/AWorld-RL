#!/usr/bin/env python3
# Evaluator for calendar Query 9

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
        self.current_date = None
        self.in_day_view = False
        self.events = []
        self.current_event = {}
        self.in_event = False
        self.capture_text = False
        self.text_buffer = []
        self.in_header = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in day view
        if tag == 'div' and 'class' in attrs_dict and 'day-view' in attrs_dict['class']:
            self.in_day_view = True
        
        # Capture the current date from header
        if tag == 'h2' and 'class' in attrs_dict and 'header-date' in attrs_dict['class']:
            self.in_header = True
            self.capture_text = True
            self.text_buffer = []
        
        # Detect event elements (they typically have event-related classes)
        if tag == 'div' and 'class' in attrs_dict:
            class_str = attrs_dict['class']
            if 'event' in class_str.lower() and 'event-card' in class_str:
                self.in_event = True
                self.current_event = {'classes': class_str, 'style': attrs_dict.get('style', '')}
        
        # Capture event title
        if self.in_event and tag == 'div' and 'class' in attrs_dict and 'event-title' in attrs_dict['class']:
            self.capture_text = True
            self.text_buffer = []
        
        # Capture event time
        if self.in_event and tag == 'div' and 'class' in attrs_dict and 'event-time' in attrs_dict['class']:
            self.capture_text = True
            self.text_buffer = []
    
    def handle_data(self, data):
        if self.capture_text:
            self.text_buffer.append(data.strip())
        
        if self.in_header:
            text = data.strip()
            if text:
                self.current_date = text
    
    def handle_endtag(self, tag):
        if tag == 'h2' and self.in_header:
            self.in_header = False
            self.capture_text = False
        
        if self.in_event and tag == 'div':
            if self.capture_text:
                text = ' '.join(self.text_buffer)
                if 'event-title' in str(self.text_buffer):
                    self.current_event['title'] = text
                elif any(t for t in ['AM', 'PM'] if t in text):
                    self.current_event['time'] = text
                self.capture_text = False
                self.text_buffer = []
            
            # Check if event div is closing
            if 'event-card' in str(self.current_event.get('classes', '')):
                if self.current_event:
                    self.events.append(self.current_event.copy())
                    self.current_event = {}
                self.in_event = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_day_view": False,
            "cp2_navigated_to_january_28": False,
            "cp3_found_events_7am_to_4pm": False,
            "cp4_applied_blue_labels": False,
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

    def parse_html_for_date_and_view(self, html_path: Path) -> Tuple[Optional[str], bool]:
        """Parse HTML to extract current date and check if in day view."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for day view
            in_day_view = 'class="day-view"' in content or '<div class="day-view">' in content
            
            # Extract date from header
            date_match = re.search(r'<h2[^>]*class="[^"]*header-date[^"]*"[^>]*>([^<]+)</h2>', content)
            current_date = date_match.group(1).strip() if date_match else None
            
            return current_date, in_day_view
        except Exception as e:
            return None, False

    def parse_html_for_events(self, html_path: Path) -> List[Dict]:
        """Parse HTML to extract events with their labels."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            events = []
            # Look for event elements - they usually have event-card class and contain time info
            event_pattern = re.compile(
                r'<div[^>]*class="[^"]*event-card[^"]*"[^>]*>.*?</div>',
                re.DOTALL
            )
            
            for event_match in event_pattern.finditer(content):
                event_html = event_match.group(0)
                
                # Extract time
                time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', event_html, re.IGNORECASE)
                time_str = time_match.group(1) if time_match else None
                
                # Extract title
                title_match = re.search(r'<div[^>]*class="[^"]*event-title[^"]*"[^>]*>([^<]+)</div>', event_html)
                title = title_match.group(1).strip() if title_match else None
                
                # Check for blue label (bg-blue class or blue color in style)
                has_blue_label = 'bg-blue' in event_html or 'background.*blue' in event_html.lower()
                
                if time_str:
                    events.append({
                        'time': time_str,
                        'title': title,
                        'has_blue_label': has_blue_label,
                        'html': event_html[:200]  # Store snippet for debugging
                    })
            
            return events
        except Exception as e:
            self.issues.append(f"Error parsing events: {str(e)}")
            return []

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to day view"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        _, in_day_view = self.parse_html_for_date_and_view(final_html)
        
        if in_day_view:
            self.details['day_view'] = 'Yes'
            return True
        else:
            self.issues.append("Not in day view")
            self.details['day_view'] = 'No'
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Navigated to January 28, 2026"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        current_date, _ = self.parse_html_for_date_and_view(final_html)
        self.details['current_date'] = current_date or 'Unknown'
        
        # Check if on January 28, 2026
        if current_date and 'January 28, 2026' in current_date:
            return True
        else:
            self.issues.append(f"Not on January 28, 2026 (current: {current_date})")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Found events that start between 7:00 AM and 4:00 PM"""
        # We need to check if the agent was ever on January 28 to see events
        all_steps = self.find_step_html("step_*")
        
        events_found = []
        for step_file in all_steps:
            current_date, in_day_view = self.parse_html_for_date_and_view(step_file)
            
            if current_date and 'January 28, 2026' in current_date and in_day_view:
                events = self.parse_html_for_events(step_file)
                if events:
                    events_found.extend(events)
        
        # Filter events between 7:00 AM and 4:00 PM
        def time_in_range(time_str: str) -> bool:
            if not time_str:
                return False
            
            try:
                # Parse time
                match = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM)', time_str, re.IGNORECASE)
                if not match:
                    return False
                
                hour = int(match.group(1))
                minute = int(match.group(2))
                period = match.group(3).upper()
                
                # Convert to 24-hour format
                if period == 'PM' and hour != 12:
                    hour += 12
                elif period == 'AM' and hour == 12:
                    hour = 0
                
                # Check if between 7:00 AM (7) and 4:00 PM (16)
                # "starts between 7:00 AM and 4:00 PM" means >= 7:00 and < 16:00
                if hour >= 7 and hour < 16:
                    return True
                elif hour == 16 and minute == 0:
                    # 4:00 PM exactly - interpret as included based on task wording
                    return True
                return False
            except:
                return False
        
        target_events = [e for e in events_found if time_in_range(e['time'])]
        
        self.details['events_in_time_range'] = len(target_events)
        self.details['sample_events'] = [{'time': e['time'], 'title': e['title']} for e in target_events[:5]]
        
        if target_events:
            return True
        else:
            self.issues.append("No events found between 7:00 AM and 4:00 PM on January 28")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Applied blue labels to all events between 7:00 AM and 4:00 PM"""
        # Check all steps where agent was on January 28
        all_steps = self.find_step_html("step_*")
        
        events_found = []
        for step_file in all_steps:
            current_date, in_day_view = self.parse_html_for_date_and_view(step_file)
            
            if current_date and 'January 28, 2026' in current_date and in_day_view:
                events = self.parse_html_for_events(step_file)
                if events:
                    events_found = events  # Use latest state
        
        # Filter events between 7:00 AM and 4:00 PM
        def time_in_range(time_str: str) -> bool:
            if not time_str:
                return False
            
            try:
                match = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM)', time_str, re.IGNORECASE)
                if not match:
                    return False
                
                hour = int(match.group(1))
                minute = int(match.group(2))
                period = match.group(3).upper()
                
                if period == 'PM' and hour != 12:
                    hour += 12
                elif period == 'AM' and hour == 12:
                    hour = 0
                
                if hour >= 7 and hour < 16:
                    return True
                elif hour == 16 and minute == 0:
                    return True
                return False
            except:
                return False
        
        target_events = [e for e in events_found if time_in_range(e['time'])]
        
        if not target_events:
            self.issues.append("No events found to apply labels to")
            return False
        
        # Check if all target events have blue labels
        events_with_blue = [e for e in target_events if e['has_blue_label']]
        
        self.details['total_target_events'] = len(target_events)
        self.details['events_with_blue_label'] = len(events_with_blue)
        
        if len(events_with_blue) == len(target_events) and len(target_events) > 0:
            return True
        else:
            self.issues.append(f"Only {len(events_with_blue)}/{len(target_events)} events have blue labels")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_day_view'] = self.check_checkpoint_1()
            self.checkpoints['cp2_navigated_to_january_28'] = self.check_checkpoint_2()
            self.checkpoints['cp3_found_events_7am_to_4pm'] = self.check_checkpoint_3()
            self.checkpoints['cp4_applied_blue_labels'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 9,
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
                'query_id': 9,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
