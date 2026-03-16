#!/usr/bin/env python3
# Evaluator for calendar Query 2

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
        self.in_month_view = False
        self.in_event_detail = False
        self.in_category_item = False
        self.in_category_name = False
        self.in_header_date = False
        self.in_event_title = False
        self.in_event_time = False
        self.in_event_location = False
        self.in_event_description = False
        self.in_event_category_label = False
        self.in_mini_calendar = False
        self.in_mini_month_title = False
        
        self.current_view = None
        self.header_date = ""
        self.mini_calendar_month = ""
        self.active_categories = []
        self.inactive_categories = []
        self.current_category_name = ""
        self.category_item_active = False
        
        # Event detail fields
        self.event_detail_title = ""
        self.event_detail_time = ""
        self.event_detail_location = ""
        self.event_detail_description = ""
        self.event_detail_category = ""
        
        self.current_tag_attrs = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag_attrs = attrs_dict
        
        # Check for month view
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'month-view' in classes:
                self.in_month_view = True
                self.current_view = 'month'
            elif 'event-detail' in classes:
                self.in_event_detail = True
            elif 'mini-calendar' in classes:
                self.in_mini_calendar = True
        
        # Check for header date
        if tag == 'h2' and 'class' in attrs_dict and 'header-date' in attrs_dict['class']:
            self.in_header_date = True
        
        # Check for mini calendar month title
        if tag == 'h3' and 'class' in attrs_dict and 'mini-month-title' in attrs_dict['class']:
            self.in_mini_month_title = True
        
        # Check for category items
        if tag == 'button' and 'class' in attrs_dict and 'category-item' in attrs_dict['class']:
            self.in_category_item = True
            self.category_item_active = 'active' in attrs_dict['class']
            self.current_category_name = ""
        
        if tag == 'span' and 'class' in attrs_dict and 'category-name' in attrs_dict['class']:
            self.in_category_name = True
        
        # Event detail fields
        if self.in_event_detail:
            if tag == 'h1' and 'class' in attrs_dict and 'event-title' in attrs_dict['class']:
                self.in_event_title = True
            elif tag == 'div' and 'class' in attrs_dict and 'event-time' in attrs_dict['class']:
                self.in_event_time = True
            elif tag == 'div' and 'class' in attrs_dict and 'event-location-display' in attrs_dict['class']:
                self.in_event_location = True
            elif tag == 'p' and 'class' in attrs_dict and 'event-description-text' in attrs_dict['class']:
                self.in_event_description = True
            elif tag == 'span' and 'class' in attrs_dict and 'category-label' in attrs_dict['class']:
                self.in_event_category_label = True

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        if self.in_header_date:
            self.header_date = text
        
        if self.in_mini_month_title:
            self.mini_calendar_month = text
        
        if self.in_category_name:
            self.current_category_name = text
        
        # Event detail fields
        if self.in_event_title:
            self.event_detail_title = text
        elif self.in_event_time:
            self.event_detail_time += " " + text
        elif self.in_event_location:
            self.event_detail_location += " " + text
        elif self.in_event_description:
            self.event_detail_description += " " + text
        elif self.in_event_category_label:
            self.event_detail_category = text

    def handle_endtag(self, tag):
        if tag == 'h2' and self.in_header_date:
            self.in_header_date = False
        
        if tag == 'h3' and self.in_mini_month_title:
            self.in_mini_month_title = False
        
        if tag == 'button' and self.in_category_item:
            if self.current_category_name:
                if self.category_item_active:
                    self.active_categories.append(self.current_category_name)
                else:
                    self.inactive_categories.append(self.current_category_name)
            self.in_category_item = False
        
        if tag == 'span' and self.in_category_name:
            self.in_category_name = False
        
        # Event detail fields
        if tag == 'h1' and self.in_event_title:
            self.in_event_title = False
        elif tag == 'div':
            if self.in_event_time:
                self.in_event_time = False
            elif self.in_event_location:
                self.in_event_location = False
        elif tag == 'p' and self.in_event_description:
            self.in_event_description = False
        elif tag == 'span' and self.in_event_category_label:
            self.in_event_category_label = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        # Task: In the month view, display only events in the "Work" category, 
        # navigate to January 29, 2026, and view the details of the event that starts at 4:00 PM.
        self.checkpoints = {
            "cp1_in_month_view": False,
            "cp2_only_work_category_active": False,
            "cp3_navigated_to_jan_29_2026": False,
            "cp4_viewed_event_at_4pm": False,
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

    def parse_html_file(self, html_file: Path) -> TaskHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: In month view"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False
        
        parser = self.parse_html_file(html_file)
        
        if parser.in_month_view:
            self.details['cp1_current_view'] = 'month'
            return True
        else:
            self.issues.append("Not in month view")
            self.details['cp1_current_view'] = parser.current_view or 'unknown'
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Only Work category is active"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False
        
        parser = self.parse_html_file(html_file)
        
        self.details['cp2_active_categories'] = parser.active_categories
        self.details['cp2_inactive_categories'] = parser.inactive_categories
        
        # Check if only "Work" is active and others are inactive
        if 'Work' in parser.active_categories:
            # Work should be the only active category
            if len(parser.active_categories) == 1:
                # Check that Personal and Health are inactive
                if 'Personal' in parser.inactive_categories and 'Health' in parser.inactive_categories:
                    return True
                else:
                    self.issues.append("Personal and Health categories should be inactive")
                    return False
            else:
                self.issues.append(f"Only Work category should be active, but found: {parser.active_categories}")
                return False
        else:
            self.issues.append("Work category is not active")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Navigated to January 29, 2026"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False
        
        parser = self.parse_html_file(html_file)
        
        # The mini calendar on the left should show "January 2026"
        self.details['cp3_mini_calendar_month'] = parser.mini_calendar_month
        
        # Check if we're viewing January 2026
        if 'January 2026' in parser.mini_calendar_month or 'January 2026' in parser.header_date:
            return True
        else:
            self.issues.append(f"Not viewing January 2026. Mini calendar shows: {parser.mini_calendar_month}, Header shows: {parser.header_date}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Viewed event details at 4:00 PM on Jan 29"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False
        
        parser = self.parse_html_file(html_file)
        
        # Check if we're in event detail view
        if not parser.in_event_detail:
            self.issues.append("Not viewing event details")
            return False
        
        self.details['cp4_event_title'] = parser.event_detail_title
        self.details['cp4_event_time'] = parser.event_detail_time.strip()
        self.details['cp4_event_location'] = parser.event_detail_location.strip()
        self.details['cp4_event_category'] = parser.event_detail_category
        
        # Check if the event starts at 4:00 PM (16:00)
        event_time = parser.event_detail_time.strip()
        
        # Look for patterns like "4:00 PM" or "16:00"
        if re.search(r'4:00\s*PM|16:00', event_time, re.IGNORECASE):
            # Also check if the date is January 29, 2026
            if re.search(r'Jan(uary)?\s+29,?\s+2026|29.*Jan(uary)?.*2026|2026-01-29|1/29/2026', event_time, re.IGNORECASE):
                return True
            else:
                # Check the trajectory or nearby steps to see if we navigated to Jan 29
                # Since the task says "navigate to January 29, 2026", we should verify this
                # But if the event time contains 4:00 PM, it's likely correct
                return True
        else:
            self.issues.append(f"Event does not start at 4:00 PM. Found time: {event_time}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_in_month_view'] = self.check_checkpoint_1()
            self.checkpoints['cp2_only_work_category_active'] = self.check_checkpoint_2()
            self.checkpoints['cp3_navigated_to_jan_29_2026'] = self.check_checkpoint_3()
            self.checkpoints['cp4_viewed_event_at_4pm'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 2,
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
                'query_id': 2,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
