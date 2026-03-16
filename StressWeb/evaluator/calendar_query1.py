#!/usr/bin/env python3
# Evaluator for calendar Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class WeekViewParser:
    """Parse week view HTML to extract event information."""
    
    def __init__(self, html_content: str):
        self.html = html_content
        self.week_columns = []
        self.events_by_time_and_column = {}
        self._parse_week_structure()
    
    def _parse_week_structure(self):
        """Parse week header to get column layout (SUN-SAT with dates)."""
        week_header = re.search(
            r'<div class="week-header">(.*?)</div>\s*<div class="time-grid-container">',
            self.html,
            re.DOTALL
        )
        if week_header:
            header_html = week_header.group(1)
            self.week_columns = re.findall(
                r'<div class="day-column-header">.*?<div class="day-name">(\w+)</div>.*?<div class="day-date[^>]*">(\d+)</div>',
                header_html,
                re.DOTALL
            )
    
    def get_events_at_time(self, time_label: str) -> Dict[int, List[Dict[str, str]]]:
        """
        Get all events at a specific time across all columns.
        
        Args:
            time_label: Time label like "9 AM", "10 AM", etc.
        
        Returns:
            Dict mapping column index to list of events with 'title' and 'time'
        """
        # Find the time section
        next_hour = int(time_label.split()[0]) + 1
        if 'AM' in time_label and next_hour == 12:
            next_label = '12 PM'
        elif 'PM' in time_label and next_hour == 12:
            next_label = '12 AM'
        elif 'AM' in time_label:
            next_label = f'{next_hour} AM'
        else:
            next_label = f'{next_hour % 12} PM' if next_hour != 12 else '12 AM'
        
        pattern = rf'<div class="time-label">{re.escape(time_label)}</div>(.*?)<div class="time-label">{re.escape(next_label)}</div>'
        time_section = re.search(pattern, self.html, re.DOTALL)
        
        if not time_section:
            return {}
        
        section = time_section.group(1)
        # Extract all time-slot divs
        slots = re.findall(
            r'<div class="time-slot[^>]*">(.*?)</div>(?=\s*(?:<div class="time-slot|<div class="time-label"))',
            section,
            re.DOTALL
        )
        
        events_by_column = {}
        for col_idx, slot in enumerate(slots):
            events = []
            # Find all events in this slot
            event_blocks = re.findall(
                r'<div class="slot-event[^>]*">.*?<div class="event-title-small">([^<]+)</div>.*?<div class="event-time">([^<]+)</div>',
                slot,
                re.DOTALL
            )
            for title, time in event_blocks:
                events.append({'title': title.strip(), 'time': time.strip()})
            
            if events:
                events_by_column[col_idx] = events
        
        return events_by_column
    
    def find_event_by_title(self, title: str) -> List[Dict[str, any]]:
        """
        Find all events matching the given title.
        
        Returns:
            List of dicts with 'title', 'time', 'day', 'date', 'column'
        """
        results = []
        # Search through all time labels
        for hour in range(12):
            for period in ['AM', 'PM']:
                time_label = f'{hour if hour > 0 else 12} {period}'
                events_by_col = self.get_events_at_time(time_label)
                
                for col_idx, events in events_by_col.items():
                    for event in events:
                        if title.lower() in event['title'].lower():
                            if col_idx < len(self.week_columns):
                                day, date = self.week_columns[col_idx]
                                results.append({
                                    'title': event['title'],
                                    'time': event['time'],
                                    'day': day,
                                    'date': int(date),
                                    'column': col_idx
                                })
        
        return results
    
    def get_column_for_date(self, date: int) -> Optional[int]:
        """Get column index for a specific date."""
        for col_idx, (day, col_date) in enumerate(self.week_columns):
            if int(col_date) == date:
                return col_idx
        return None


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_week_view_active": False,
            "cp2_event_created": False,
            "cp3_correct_date": False,
            "cp4_correct_time": False,
            "cp5_correct_duration": False,
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

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def check_checkpoint_1(self, html_content: str) -> bool:
        """Checkpoint 1: Week view is active."""
        # Check if week view button is active
        week_active = re.search(
            r'<button class="view-btn active">Week</button>',
            html_content
        )
        
        if not week_active:
            self.issues.append("Week view is not active in final state")
            return False
        
        self.details['view_mode'] = 'week'
        return True

    def check_checkpoint_2(self, parser: WeekViewParser) -> bool:
        """Checkpoint 2: Event titled 'Team Standup' exists."""
        events = parser.find_event_by_title("Team Standup")
        
        if not events:
            self.issues.append("No event titled 'Team Standup' found in the calendar")
            return False
        
        self.details['team_standup_events'] = events
        return True

    def check_checkpoint_3(self, parser: WeekViewParser) -> bool:
        """Checkpoint 3: Event is on the correct date (January 28, 2026)."""
        events = parser.find_event_by_title("Team Standup")
        
        if not events:
            self.issues.append("Cannot verify date - no 'Team Standup' event found")
            return False
        
        # Check if any event is on date 28
        correct_date_events = [e for e in events if e['date'] == 28]
        
        if not correct_date_events:
            actual_dates = [f"{e['day']} {e['date']}" for e in events]
            self.issues.append(
                f"'Team Standup' event is not on date 28. Found on: {', '.join(actual_dates)}"
            )
            return False
        
        self.details['event_date'] = '28 (WED)'
        return True

    def check_checkpoint_4(self, parser: WeekViewParser) -> bool:
        """Checkpoint 4: Event starts at 10:00 AM."""
        events = parser.find_event_by_title("Team Standup")
        
        if not events:
            self.issues.append("Cannot verify start time - no 'Team Standup' event found")
            return False
        
        # Filter events on date 28
        date_28_events = [e for e in events if e['date'] == 28]
        
        if not date_28_events:
            self.issues.append("Cannot verify start time - no event on date 28")
            return False
        
        # Check if start time is 10:00 AM
        correct_time_events = [
            e for e in date_28_events 
            if e['time'].startswith('10:00 AM')
        ]
        
        if not correct_time_events:
            actual_times = [e['time'] for e in date_28_events]
            self.issues.append(
                f"'Team Standup' event does not start at 10:00 AM. Found: {', '.join(actual_times)}"
            )
            return False
        
        self.details['event_start_time'] = '10:00 AM'
        return True

    def check_checkpoint_5(self, parser: WeekViewParser) -> bool:
        """Checkpoint 5: Event duration is 30 minutes (ends at 10:30 AM)."""
        events = parser.find_event_by_title("Team Standup")
        
        if not events:
            self.issues.append("Cannot verify duration - no 'Team Standup' event found")
            return False
        
        # Filter events on date 28 starting at 10:00 AM
        target_events = [
            e for e in events 
            if e['date'] == 28 and e['time'].startswith('10:00 AM')
        ]
        
        if not target_events:
            self.issues.append("Cannot verify duration - no event at correct date/time")
            return False
        
        # Check if end time is 10:30 AM (30 minutes duration)
        correct_duration = False
        for event in target_events:
            # Expected format: "10:00 AM - 10:30 AM"
            if '10:30 AM' in event['time']:
                correct_duration = True
                self.details['event_duration'] = '30 minutes'
                break
        
        if not correct_duration:
            actual_times = [e['time'] for e in target_events]
            self.issues.append(
                f"'Team Standup' event duration is not 30 minutes. Found: {', '.join(actual_times)}"
            )
            return False
        
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Find final HTML
            final_html_path = self.find_final_html()
            if not final_html_path:
                raise FileNotFoundError("No final HTML state file found")

            with open(final_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Parse the week view
            parser = WeekViewParser(html_content)

            # Run checkpoint validations
            self.checkpoints['cp1_week_view_active'] = self.check_checkpoint_1(html_content)
            self.checkpoints['cp2_event_created'] = self.check_checkpoint_2(parser)
            self.checkpoints['cp3_correct_date'] = self.check_checkpoint_3(parser)
            self.checkpoints['cp4_correct_time'] = self.check_checkpoint_4(parser)
            self.checkpoints['cp5_correct_duration'] = self.check_checkpoint_5(parser)

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 1,
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
                'query_id': 1,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
