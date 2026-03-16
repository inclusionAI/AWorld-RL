#!/usr/bin/env python3
# Evaluator for calendar Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class EventDetailParser(HTMLParser):
    """Parse HTML to extract event detail information."""

    def __init__(self):
        super().__init__()
        self.in_detail_title = False
        self.in_detail_value = False
        self.in_detail_label = False
        self.current_label = ""
        self.event_title = ""
        self.event_date = ""
        self.event_time = ""
        self.event_duration = ""
        self.event_category = ""
        self.is_detail_view = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in event detail view
        if tag == "div" and attrs_dict.get("class") == "event-detail":
            self.is_detail_view = True
        
        # Check for detail title
        if tag == "h2" and "detail-title" in attrs_dict.get("class", ""):
            self.in_detail_title = True
        
        # Check for detail label
        if tag == "div" and "detail-label" in attrs_dict.get("class", ""):
            self.in_detail_label = True
        
        # Check for detail value
        if tag == "div" and "detail-value" in attrs_dict.get("class", ""):
            self.in_detail_value = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_detail_title:
            self.event_title = data
        elif self.in_detail_label:
            self.current_label = data
        elif self.in_detail_value:
            if "DATE" in self.current_label:
                self.event_date = data
            elif "TIME" in self.current_label:
                self.event_time = data
            elif "DURATION" in self.current_label:
                self.event_duration = data
            elif "CATEGORY" in self.current_label:
                self.event_category = data

    def handle_endtag(self, tag):
        if tag == "h2":
            self.in_detail_title = False
        elif tag == "div":
            self.in_detail_label = False
            self.in_detail_value = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_found_gym_event": False,
            "cp2_earliest_event_identified": False,
            "cp3_viewing_event_details": False,
            "cp4_correct_event_displayed": False,
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

    def parse_html_file(self, html_path: Path) -> EventDetailParser:
        """Parse HTML file and return parser with extracted data."""
        parser = EventDetailParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Agent found at least one GYM event"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked on a GYM event
            for action in action_history:
                action_obj = action.get('action', {})
                if action_obj.get('action_type') == 'CLICK':
                    params = action_obj.get('parameters', {})
                    selector = params.get('selector', '')
                    if 'GYM' in selector.upper():
                        self.details['gym_event_clicked'] = True
                        return True
            
            self.issues.append("Agent did not click on any GYM event")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Agent searched for the earliest GYM event (navigated to previous months)"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent navigated to previous months
            previous_month_clicks = 0
            for action in action_history:
                action_obj = action.get('action', {})
                reasoning = action_obj.get('reasoning', '').lower()
                
                # Check for navigation to previous months
                if 'previous' in reasoning and ('month' in reasoning or 'earlier' in reasoning):
                    previous_month_clicks += 1
            
            if previous_month_clicks > 0:
                self.details['previous_month_navigations'] = previous_month_clicks
                return True
            else:
                self.issues.append("Agent did not search for earlier GYM events by navigating to previous months")
                return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Final state shows event detail view"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html_file(final_html)
            
            if parser.is_detail_view:
                self.details['is_detail_view'] = True
                return True
            else:
                self.issues.append("Final state is not showing event detail view")
                return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Event details show the correct earliest GYM event (January 25, 2026)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html_file(final_html)
            
            # Check if title contains GYM
            title_has_gym = 'GYM' in parser.event_title.upper()
            
            # Check if date is January 25, 2026
            date_match = 'January 25, 2026' in parser.event_date or 'Jan 25, 2026' in parser.event_date or '1/25/2026' in parser.event_date
            
            # Also check for Sunday since January 25, 2026 is a Sunday
            if 'Sunday' in parser.event_date and '25' in parser.event_date and '2026' in parser.event_date:
                date_match = True
            
            self.details['event_title'] = parser.event_title
            self.details['event_date'] = parser.event_date
            self.details['event_time'] = parser.event_time
            self.details['event_duration'] = parser.event_duration
            
            if not title_has_gym:
                self.issues.append(f"Event title does not contain 'GYM': {parser.event_title}")
                return False
            
            if not date_match:
                self.issues.append(f"Event date is not January 25, 2026: {parser.event_date}")
                return False
            
            return True
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_found_gym_event'] = self.check_checkpoint_1()
            self.checkpoints['cp2_earliest_event_identified'] = self.check_checkpoint_2()
            self.checkpoints['cp3_viewing_event_details'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_event_displayed'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 3,
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
                'query_id': 3,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
