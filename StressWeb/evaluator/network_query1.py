#!/usr/bin/env python3
# Evaluator for network Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class NotificationsHTMLParser(HTMLParser):
    """Parse HTML to extract notifications information."""

    def __init__(self):
        super().__init__()
        self.in_notifications_page = False
        self.unread_count = None
        self.mark_all_button_disabled = None
        self.unread_notifications = 0
        self.in_filter_button = False
        self.current_button_text = ""
        self.in_notification_card = False
        self.current_notification_classes = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're on notifications page
        if tag == "div" and attrs_dict.get("class") == "notifications-page":
            self.in_notifications_page = True
        
        # Check for filter button with Unread count
        if tag == "button" and "filter-button" in attrs_dict.get("class", ""):
            self.in_filter_button = True
            self.current_button_text = ""
        
        # Check for mark all button
        if tag == "button" and "mark-all-button" in attrs_dict.get("class", ""):
            self.mark_all_button_disabled = "disabled" in attrs_dict
        
        # Check for notification cards
        if tag == "div" and "notification-card" in attrs_dict.get("class", ""):
            self.in_notification_card = True
            self.current_notification_classes = attrs_dict.get("class", "").split()
            if "unread" in self.current_notification_classes:
                self.unread_notifications += 1

    def handle_data(self, data):
        if self.in_filter_button:
            self.current_button_text += data.strip()

    def handle_endtag(self, tag):
        if tag == "button" and self.in_filter_button:
            self.in_filter_button = False
            # Extract unread count from text like "Unread (4)"
            match = re.search(r'Unread\s*\((\d+)\)', self.current_button_text)
            if match:
                self.unread_count = int(match.group(1))
        
        if tag == "div" and self.in_notification_card:
            self.in_notification_card = False
            self.current_notification_classes = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_notifications": False,
            "cp2_notifications_had_unread": False,
            "cp3_marked_all_as_read": False,
            "cp4_all_notifications_read": False,
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

    def parse_html(self, html_path: Path) -> NotificationsHTMLParser:
        """Parse HTML file and extract notifications data."""
        parser = NotificationsHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            parser.feed(f.read())
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Notifications page"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html(final_html)
            if parser.in_notifications_page:
                self.details["cp1_on_notifications_page"] = True
                return True
            else:
                self.issues.append("Final page is not the Notifications page")
                return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Notifications initially had unread items"""
        try:
            # Check step 1 (after navigating to notifications but before marking as read)
            step_files = self.find_step_html("step_1_*")
            if not step_files:
                self.issues.append("No step 1 HTML found to verify initial unread state")
                return False
            
            parser = self.parse_html(step_files[0])
            
            if parser.unread_count is not None and parser.unread_count > 0:
                self.details["initial_unread_count"] = parser.unread_count
                return True
            elif parser.unread_notifications > 0:
                self.details["initial_unread_notifications"] = parser.unread_notifications
                return True
            else:
                self.issues.append("No unread notifications found initially")
                return False
        except Exception as e:
            self.issues.append(f"Error checking initial unread state: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Mark all as read button was clicked"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if there's a CLICK action on mark all as read button
            for entry in trajectory:
                action = entry.get("action", {})
                if action.get("action_type") == "CLICK":
                    selector = action.get("parameters", {}).get("selector", "")
                    if "mark all as read" in selector.lower():
                        self.details["mark_all_clicked"] = True
                        return True
            
            self.issues.append("Mark all as read button was not clicked")
            return False
        except Exception as e:
            self.issues.append(f"Error checking mark all action: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: All notifications are now read (unread count is 0)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html(final_html)
            
            # Check unread count
            if parser.unread_count is not None:
                self.details["final_unread_count"] = parser.unread_count
                if parser.unread_count == 0:
                    return True
                else:
                    self.issues.append(f"Unread count is {parser.unread_count}, expected 0")
                    return False
            
            # Check if mark all button is disabled (indicates all read)
            if parser.mark_all_button_disabled:
                self.details["mark_all_button_disabled"] = True
                return True
            
            # Check for unread notification cards
            if parser.unread_notifications == 0:
                self.details["no_unread_cards"] = True
                return True
            else:
                self.issues.append(f"Found {parser.unread_notifications} unread notification cards")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking final state: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_notifications'] = self.check_checkpoint_1()
            self.checkpoints['cp2_notifications_had_unread'] = self.check_checkpoint_2()
            self.checkpoints['cp3_marked_all_as_read'] = self.check_checkpoint_3()
            self.checkpoints['cp4_all_notifications_read'] = self.check_checkpoint_4()

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
        print("Usage: python network_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
