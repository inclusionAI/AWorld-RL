#!/usr/bin/env python3
# Evaluator for calendar Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EventDetailParser(HTMLParser):
    """Parse HTML to extract event detail information."""

    def __init__(self):
        super().__init__()
        self.in_detail_value = False
        self.in_detail_label = False
        self.in_detail_title = False
        self.current_label = ""
        self.event_details = {}
        self.event_title = ""
        self.detail_values = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div":
            if attrs_dict.get("class") == "detail-label":
                self.in_detail_label = True
            elif attrs_dict.get("class") == "detail-value":
                self.in_detail_value = True
        elif tag == "h2":
            if attrs_dict.get("class") == "detail-title":
                self.in_detail_title = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_detail_title:
            self.event_title = data
        elif self.in_detail_label:
            self.current_label = data
        elif self.in_detail_value and self.current_label:
            self.detail_values.append(data)

    def handle_endtag(self, tag):
        if tag == "div":
            if self.in_detail_label:
                self.in_detail_label = False
            elif self.in_detail_value:
                if self.current_label and self.detail_values:
                    self.event_details[self.current_label] = " ".join(self.detail_values)
                    self.detail_values = []
                    self.current_label = ""
                self.in_detail_value = False
        elif tag == "h2" and self.in_detail_title:
            self.in_detail_title = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_event_found": False,
            "cp2_date_changed_to_jan_29": False,
            "cp3_time_changed_to_11am": False,
            "cp4_event_details_correct": False,
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

    def parse_event_details(self, html_path: Path) -> Tuple[str, Dict[str, str]]:
        """Parse event details from HTML."""
        parser = EventDetailParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser.event_title, parser.event_details

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Team Standup event found"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False

            event_title, event_details = self.parse_event_details(final_html)
            
            # Check if we're viewing Team Standup event
            if "Team Standup" in event_title:
                self.details["event_title"] = event_title
                return True
            else:
                self.issues.append(f"Team Standup event not found in final state. Found: {event_title}")
                return False
        except Exception as e:
            self.issues.append(f"Error checking event: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Date changed to January 29, 2026"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            event_title, event_details = self.parse_event_details(final_html)
            
            date_value = event_details.get("📅 DATE", "")
            self.details["final_date"] = date_value
            
            # Check for January 29, 2026
            if "January 29, 2026" in date_value or "29" in date_value and "January" in date_value and "2026" in date_value:
                # Also verify it's Thursday
                if "Thursday" in date_value or "Thu" in date_value:
                    return True
                else:
                    # Still accept if date is correct even without day name
                    return "29" in date_value and "January" in date_value and "2026" in date_value
            else:
                self.issues.append(f"Date not changed to January 29, 2026. Found: {date_value}")
                return False
        except Exception as e:
            self.issues.append(f"Error checking date: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Time changed to 11:00 AM"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            event_title, event_details = self.parse_event_details(final_html)
            
            time_value = event_details.get("🕐 TIME", "")
            self.details["final_time"] = time_value
            
            # Check for 11:00 AM start time
            if "11:00 AM" in time_value or "11:00AM" in time_value:
                return True
            else:
                self.issues.append(f"Time not changed to 11:00 AM. Found: {time_value}")
                return False
        except Exception as e:
            self.issues.append(f"Error checking time: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Event details are correct (30 min duration, end time 11:30 AM)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            event_title, event_details = self.parse_event_details(final_html)
            
            time_value = event_details.get("🕐 TIME", "")
            duration_value = event_details.get("⏱️ DURATION", "")
            
            self.details["final_duration"] = duration_value
            
            # Check for 30 minutes duration
            duration_ok = "30 minutes" in duration_value or "30 min" in duration_value
            
            # Check end time is 11:30 AM
            end_time_ok = "11:30 AM" in time_value or "11:30AM" in time_value
            
            if duration_ok and end_time_ok:
                return True
            else:
                if not duration_ok:
                    self.issues.append(f"Duration not correct. Expected 30 minutes, found: {duration_value}")
                if not end_time_ok:
                    self.issues.append(f"End time not correct. Expected 11:30 AM in: {time_value}")
                return False
        except Exception as e:
            self.issues.append(f"Error checking event details: {str(e)}")
            return False

    def verify_original_event(self) -> bool:
        """Verify the original event was on Jan 27 at 9:00 AM before editing"""
        try:
            # Check step 2 which should show the event detail before editing
            step_files = self.find_step_html("step_2_*")
            if not step_files:
                return True  # Can't verify but don't fail
            
            event_title, event_details = self.parse_event_details(step_files[0])
            
            if "Team Standup" not in event_title:
                return True  # Different event shown, skip verification
            
            original_date = event_details.get("📅 DATE", "")
            original_time = event_details.get("🕐 TIME", "")
            
            self.details["original_date"] = original_date
            self.details["original_time"] = original_time
            
            # Verify it was January 27 at 9:00 AM
            date_ok = "January 27" in original_date and "2026" in original_date
            time_ok = "9:00 AM" in original_time
            
            return date_ok and time_ok
        except Exception as e:
            # If we can't verify, don't fail the evaluation
            return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = {}
            if self.result_file.exists():
                with open(self.result_file, 'r') as f:
                    result_data = json.load(f)

            # Verify original event state
            original_correct = self.verify_original_event()
            if not original_correct:
                self.issues.append("Original event was not at 9:00 AM on January 27, 2026")

            # Run checkpoint validations
            self.checkpoints['cp1_event_found'] = self.check_checkpoint_1()
            self.checkpoints['cp2_date_changed_to_jan_29'] = self.check_checkpoint_2()
            self.checkpoints['cp3_time_changed_to_11am'] = self.check_checkpoint_3()
            self.checkpoints['cp4_event_details_correct'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 7,
                'query': result_data.get('query', 'In the week view, find the "Team Standup" event at 9:00 AM on January 27, 2026, and change it to 11:00 AM on January 29.'),
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
                'query_id': 7,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
