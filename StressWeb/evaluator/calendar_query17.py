#!/usr/bin/env python3
# Evaluator for calendar Query 17

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class CalendarHTMLParser(HTMLParser):
    """Parse HTML to extract calendar state information."""

    def __init__(self):
        super().__init__()
        self.in_header_date = False
        self.header_date_text = ""
        self.active_view = ""
        self.in_view_btn = False
        self.current_btn_text = ""
        self.current_btn_active = False
        self.day_view_date = ""
        self.in_day_header = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get('class', '')
        
        if 'header-date' in classes:
            self.in_header_date = True
        elif 'view-btn' in classes:
            self.in_view_btn = True
            self.current_btn_text = ""
            self.current_btn_active = 'active' in classes
        elif 'day-view-header' in classes or 'day-header' in classes:
            self.in_day_header = True
            
    def handle_data(self, data):
        text = data.strip()
        if self.in_header_date:
            self.header_date_text += text
        elif self.in_view_btn:
            self.current_btn_text += text
        elif self.in_day_header:
            self.day_view_date += text
            
    def handle_endtag(self, tag):
        if self.in_view_btn:
            if self.current_btn_active:
                self.active_view = self.current_btn_text
            self.in_view_btn = False
        if self.in_header_date:
            self.in_header_date = False
        if self.in_day_header:
            self.in_day_header = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for task: "View the day view for June 21, 2026"
        self.checkpoints = {
            "cp1_navigated_to_june_2026": False,
            "cp2_switched_to_day_view": False,
            "cp3_viewing_june_21_2026": False,
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

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def parse_html_file(self, html_file: Path) -> CalendarHTMLParser:
        """Parse an HTML file and return the parser with extracted data."""
        parser = CalendarHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            parser.feed(html_content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to June 2026"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML state found")
            return False

        parser = self.parse_html_file(final_html)
        date_text = parser.header_date_text.lower()
        
        # Check if we're viewing June 2026
        if 'june' in date_text and '2026' in date_text:
            self.details['final_month'] = 'June 2026'
            return True
        else:
            self.details['final_month'] = parser.header_date_text or "Unknown"
            self.issues.append(f"Did not navigate to June 2026. Final month: {self.details['final_month']}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Switched to Day view"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        parser = self.parse_html_file(final_html)
        active_view = parser.active_view.lower()
        
        if 'day' in active_view:
            self.details['final_view'] = 'Day'
            return True
        else:
            self.details['final_view'] = parser.active_view or "Unknown"
            self.issues.append(f"Did not switch to Day view. Active view: {self.details['final_view']}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Viewing June 21, 2026 specifically"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        # Must have both June 2026 and Day view
        if not (self.checkpoints.get('cp1_navigated_to_june_2026') and 
                self.checkpoints.get('cp2_switched_to_day_view')):
            self.issues.append("Cannot verify June 21 without being in Day view of June 2026")
            return False

        parser = self.parse_html_file(final_html)
        
        # Parse HTML to check for June 21 indicators
        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for various patterns that indicate June 21, 2026
        # Could be "June 21", "21 June", "Saturday, June 21", etc.
        patterns = [
            r'june\s+21',
            r'21\s+june',
            r'saturday,?\s+june\s+21',
            r'jun\s+21',
            r'21\s+jun',
        ]
        
        found_june_21 = False
        for pattern in patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                found_june_21 = True
                break
        
        # Also check the day_view_date from parser
        if parser.day_view_date:
            self.details['day_view_date'] = parser.day_view_date
            if '21' in parser.day_view_date and ('june' in parser.day_view_date.lower() or 'jun' in parser.day_view_date.lower()):
                found_june_21 = True
        
        # Check header date for day number
        if '21' in parser.header_date_text and 'june' in parser.header_date_text.lower():
            found_june_21 = True
            self.details['viewing_date'] = 'June 21, 2026'
        
        if found_june_21:
            self.details['viewing_date'] = 'June 21, 2026'
            return True
        else:
            self.issues.append("Not specifically viewing June 21, 2026")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_june_2026'] = self.check_checkpoint_1()
            self.checkpoints['cp2_switched_to_day_view'] = self.check_checkpoint_2()
            self.checkpoints['cp3_viewing_june_21_2026'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 17,
                'query': result_data.get('query', 'View the day view for June 21, 2026.'),
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
                'query_id': 17,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query17.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
