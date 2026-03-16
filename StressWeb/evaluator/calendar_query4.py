#!/usr/bin/env python3
# Evaluator for calendar Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract calendar view information."""

    def __init__(self):
        super().__init__()
        self.current_view = None  # week, month, day, agenda
        self.date_range = None  # For week view: "Feb 8 - Feb 14, 2026"
        self.month_year = None  # For month view: "May 2026"
        self.in_header_date = False
        self.in_month_header = False
        self.in_view_btn = False
        self.current_btn_text = ""
        self.is_active_btn = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for active view button
        if tag == "button" and "view-btn" in attrs_dict.get("class", ""):
            self.in_view_btn = True
            self.is_active_btn = "active" in attrs_dict.get("class", "")
            self.current_btn_text = ""
            
        # Check for header date (week view date range)
        if tag == "h2" and "header-date" in attrs_dict.get("class", ""):
            self.in_header_date = True
            
        # Check for month header (month view)
        if tag == "h2" and "month-header-title" in attrs_dict.get("class", ""):
            self.in_month_header = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_view_btn:
            self.current_btn_text = data
            
        if self.in_header_date:
            self.date_range = data
            
        if self.in_month_header:
            self.month_year = data

    def handle_endtag(self, tag):
        if tag == "button" and self.in_view_btn:
            if self.is_active_btn and self.current_btn_text:
                self.current_view = self.current_btn_text.lower()
            self.in_view_btn = False
            self.is_active_btn = False
            self.current_btn_text = ""
            
        if tag == "h2" and self.in_header_date:
            self.in_header_date = False
            
        if tag == "h2" and self.in_month_header:
            self.in_month_header = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        # Task: View the week view from Feb 8 to Feb 14, then switch to the month view of May 2026
        self.checkpoints = {
            "cp1_initial_week_view": False,  # Started in week view or switched to it
            "cp2_navigated_to_feb_8_14": False,  # Navigated to week of Feb 8-14, 2026
            "cp3_switched_to_month_view": False,  # Switched from week to month view
            "cp4_navigated_to_may_2026": False,  # Navigated to May 2026 in month view
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

    def parse_html_file(self, html_path: Path) -> TaskHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            parser.feed(f.read())
        return parser

    def extract_step_number(self, filename: str) -> int:
        """Extract step number from filename like step_42_timestamp_raw.html"""
        match = re.search(r'step_(\d+)_', filename)
        if match:
            return int(match.group(1))
        return -1

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Started in week view or switched to it"""
        try:
            # Check initial state and early steps
            step_files = self.find_step_html("step_*")
            if not step_files:
                self.issues.append("No step files found")
                return False
            
            # Check first few steps for week view
            for step_file in step_files[:5]:
                parser = self.parse_html_file(step_file)
                if parser.current_view == "week":
                    self.details["initial_view"] = "week"
                    return True
            
            self.issues.append("Did not switch to week view initially")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Navigated to week of Feb 8-14, 2026"""
        try:
            step_files = self.find_step_html("step_*")
            
            # Check all intermediate steps for Feb 8-14 date range
            for step_file in step_files:
                parser = self.parse_html_file(step_file)
                if parser.current_view == "week" and parser.date_range:
                    # Check if date range matches Feb 8-14
                    if "Feb 8" in parser.date_range and "Feb 14" in parser.date_range:
                        step_num = self.extract_step_number(step_file.name)
                        self.details["feb_8_14_reached_at_step"] = step_num
                        self.details["date_range_feb"] = parser.date_range
                        return True
            
            # Check what date range was shown in final state
            final_html = self.find_final_html()
            if final_html:
                parser = self.parse_html_file(final_html)
                if parser.date_range:
                    self.details["final_date_range"] = parser.date_range
                    self.issues.append(f"Did not reach Feb 8-14 week. Final date range: {parser.date_range}")
                else:
                    self.issues.append("Did not reach Feb 8-14 week. No date range found in final state")
            else:
                self.issues.append("Did not reach Feb 8-14 week. No HTML files found")
                
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Switched to month view"""
        try:
            step_files = self.find_step_html("step_*")
            
            # Look for transition from week to month view
            prev_view = None
            for step_file in step_files:
                parser = self.parse_html_file(step_file)
                if prev_view == "week" and parser.current_view == "month":
                    step_num = self.extract_step_number(step_file.name)
                    self.details["switched_to_month_at_step"] = step_num
                    return True
                prev_view = parser.current_view
            
            # Check final state
            final_html = self.find_final_html()
            if final_html:
                parser = self.parse_html_file(final_html)
                if parser.current_view:
                    self.details["final_view"] = parser.current_view
                    self.issues.append(f"Did not switch to month view. Final view: {parser.current_view}")
                else:
                    self.issues.append("Did not switch to month view. Could not determine final view")
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Navigated to May 2026 in month view"""
        try:
            step_files = self.find_step_html("step_*")
            
            # Check all steps for May 2026 in month view
            for step_file in step_files:
                parser = self.parse_html_file(step_file)
                if parser.current_view == "month":
                    # Check month header or date range for May 2026
                    if parser.month_year and "May" in parser.month_year and "2026" in parser.month_year:
                        step_num = self.extract_step_number(step_file.name)
                        self.details["may_2026_reached_at_step"] = step_num
                        self.details["month_year"] = parser.month_year
                        return True
            
            # Check final state
            final_html = self.find_final_html()
            if final_html:
                parser = self.parse_html_file(final_html)
                if parser.current_view == "month":
                    month_info = parser.month_year if parser.month_year else "unknown"
                    self.details["final_month"] = month_info
                    self.issues.append(f"Did not reach May 2026. Final month: {month_info}")
                else:
                    self.issues.append(f"Did not reach May 2026. Not in month view in final state")
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_initial_week_view'] = self.check_checkpoint_1()
            self.checkpoints['cp2_navigated_to_feb_8_14'] = self.check_checkpoint_2()
            self.checkpoints['cp3_switched_to_month_view'] = self.check_checkpoint_3()
            self.checkpoints['cp4_navigated_to_may_2026'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 4,
                'query': result_data.get('query', 'View the week view from Feb 8 to Feb 14, then switch to the month view of May 2026.'),
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
                'query_id': 4,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query4.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
