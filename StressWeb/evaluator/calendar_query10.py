#!/usr/bin/env python3
# Evaluator for calendar Query 10

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EventHTMLParser(HTMLParser):
    """Parse HTML to extract event information from week view."""
    def __init__(self):
        super().__init__()
        self.events = []
        self.current_event = None
        self.in_event = False
        self.in_title = False
        self.in_time = False
        self.current_text = ""
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect event container
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'slot-event' in classes:
                self.in_event = True
                self.current_event = {
                    'title': '',
                    'time': '',
                    'style': attrs_dict.get('style', '')
                }
            elif self.in_event and 'event-title-small' in classes:
                self.in_title = True
            elif self.in_event and 'event-time' in classes:
                self.in_time = True
                
    def handle_data(self, data):
        if self.in_title:
            self.current_event['title'] = data.strip()
        elif self.in_time:
            self.current_event['time'] = data.strip()
            
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_title:
                self.in_title = False
            elif self.in_time:
                self.in_time = False
            elif self.in_event and self.current_event and self.current_event['title']:
                self.events.append(self.current_event.copy())
                self.current_event = None
                self.in_event = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_week_view_loaded": False,
            "cp2_conflicting_events_identified": False,
            "cp3_conflicting_events_deleted": False,
        }
        
        self.issues = []
        self.details = {}
    
    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        # Try to find the last step file
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        return step_files[-1] if step_files else None
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def parse_events_from_html(self, html_path: Path) -> List[Dict]:
        """Parse events from HTML file."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = EventHTMLParser()
            parser.feed(html_content)
            return parser.events
        except Exception as e:
            self.issues.append(f"Error parsing HTML: {str(e)}")
            return []
    
    def identify_conflicts(self, events: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Identify time conflicts between events."""
        conflicts = []
        
        def parse_time(time_str: str) -> Tuple[int, int]:
            """Parse time string like '9:00 AM' to (hour, minute)."""
            parts = time_str.split()
            time_parts = parts[0].split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            
            if 'PM' in parts[-1] and hour != 12:
                hour += 12
            elif 'AM' in parts[-1] and hour == 12:
                hour = 0
                
            return (hour, minute)
        
        def get_time_range(event: Dict) -> Tuple[Tuple[int, int], Tuple[int, int]]:
            """Extract start and end time from event."""
            time_str = event['time']
            if ' - ' in time_str:
                start_str, end_str = time_str.split(' - ')
                return (parse_time(start_str), parse_time(end_str))
            return ((0, 0), (0, 0))
        
        def times_overlap(range1: Tuple, range2: Tuple) -> bool:
            """Check if two time ranges overlap."""
            start1, end1 = range1
            start2, end2 = range2
            return start1 < end2 and start2 < end1
        
        # Group events by day (this is simplified - assumes same day)
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                try:
                    range1 = get_time_range(event1)
                    range2 = get_time_range(event2)
                    
                    if times_overlap(range1, range2):
                        conflicts.append((event1, event2))
                except Exception as e:
                    continue
        
        return conflicts
    
    def check_checkpoint_1(self) -> bool:
        """Check if week view was loaded."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No HTML file found")
            return False
        
        try:
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for week view indicators
            if 'week-view' in html_content or 'week-grid' in html_content:
                self.details['week_view'] = 'Week view loaded'
                return True
            else:
                self.issues.append("Week view not found in final state")
                return False
        except Exception as e:
            self.issues.append(f"Error checking week view: {str(e)}")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if conflicting events were identified (by checking action history)."""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent attempted to delete events
            delete_attempts = [a for a in action_history if 
                             a.get('action', {}).get('parameters', {}).get('selector', '').find('Delete') >= 0 or
                             'delete' in str(a.get('action', {}).get('reasoning', '')).lower()]
            
            if len(delete_attempts) > 0:
                self.details['delete_attempts'] = len(delete_attempts)
                return True
            else:
                self.issues.append("No delete actions attempted")
                return False
        except Exception as e:
            self.issues.append(f"Error checking delete attempts: {str(e)}")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if conflicting events were actually deleted."""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        # Parse events from final state
        final_events = self.parse_events_from_html(final_html)
        self.details['final_event_count'] = len(final_events)
        
        # Identify remaining conflicts
        conflicts = self.identify_conflicts(final_events)
        self.details['remaining_conflicts'] = len(conflicts)
        
        if len(conflicts) > 0:
            conflict_details = []
            for e1, e2 in conflicts[:5]:  # Show first 5 conflicts
                conflict_details.append({
                    'event1': {'title': e1['title'], 'time': e1['time']},
                    'event2': {'title': e2['title'], 'time': e2['time']}
                })
            self.details['conflict_examples'] = conflict_details
            self.issues.append(f"Found {len(conflicts)} remaining time conflicts")
            return False
        
        # Check specific known conflicts from the task
        # Based on the action history, agent was trying to delete:
        # - GYM (conflicts with Team Standup on Mon)
        # - Code Review (conflicts with Focus Time on Thu)
        
        remaining_titles = [e['title'] for e in final_events]
        
        # Count how many overlapping events remain in critical time slots
        morning_events = [e for e in final_events if '9:00 AM' in e['time'] or '10:00 AM' in e['time']]
        
        # If there are multiple events in the same time slot, that's a problem
        # Simple heuristic: if we still have many conflicts, task failed
        if len(conflicts) == 0:
            self.details['status'] = 'All conflicts resolved'
            return True
        else:
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_week_view_loaded'] = self.check_checkpoint_1()
            self.checkpoints['cp2_conflicting_events_identified'] = self.check_checkpoint_2()
            self.checkpoints['cp3_conflicting_events_deleted'] = self.check_checkpoint_3()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query10.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
