#!/usr/bin/env python3
# Evaluator for email Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class EmailHTMLParser(HTMLParser):
    """Parse HTML to extract email information."""

    def __init__(self):
        super().__init__()
        self.in_message_item = False
        self.in_sender = False
        self.in_subject = False
        self.current_sender = None
        self.current_subject = None
        self.emails = []
        self.in_label_view = False
        self.current_label = None
        self.in_label_title = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_val = attrs_dict.get('class', '')
        
        if 'message-item' in class_val:
            self.in_message_item = True
            self.current_sender = None
            self.current_subject = None
        elif 'sender' in class_val and self.in_message_item:
            self.in_sender = True
        elif 'subject' in class_val and self.in_message_item:
            self.in_subject = True
        elif 'label-title' in class_val:
            self.in_label_title = True

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_message_item:
            if self.current_sender and self.current_subject:
                self.emails.append({
                    'sender': self.current_sender.strip(),
                    'subject': self.current_subject.strip()
                })
            self.in_message_item = False
        elif self.in_sender:
            self.in_sender = False
        elif self.in_subject:
            self.in_subject = False
        elif self.in_label_title:
            self.in_label_title = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_sender:
            self.current_sender = data
        elif self.in_subject:
            self.current_subject = data
        elif self.in_label_title:
            self.current_label = data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_email_found": False,
            "cp2_work_label_applied": False,
            "cp3_email_in_work_view": False,
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

    def parse_html_for_emails(self, html_path: Path) -> List[Dict]:
        """Parse HTML file and extract email list."""
        parser = EmailHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser.emails

    def check_work_label_view(self, html_path: Path) -> tuple[bool, List[Dict]]:
        """Check if HTML shows Work label view and return emails."""
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if we're in the Work label view
        if 'class="label-view-main"' in content or 'label-title">Work<' in content:
            parser = EmailHTMLParser()
            parser.feed(content)
            return True, parser.emails
        return False, []

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Email from updates@automation.io with subject 'Training session invitation' exists."""
        try:
            # Check in search results (step 13 or later)
            search_steps = self.find_step_html("step_1[3-9]*")
            if not search_steps:
                search_steps = self.find_step_html("step_[2-9]*")
            
            for step_file in search_steps:
                emails = self.parse_html_for_emails(step_file)
                for email in emails:
                    if email['sender'] == 'updates@automation.io' and \
                       'Training session invitation' in email['subject']:
                        self.details['target_email_found'] = True
                        return True
            
            self.issues.append("Target email 'Training session invitation' from updates@automation.io not found")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking for target email: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Work label was clicked/applied (check trajectory)."""
        try:
            trajectory = self.get_trajectory()
            
            # Look for actions that involve clicking "Work" or applying label
            for entry in trajectory:
                action = entry.get('action', {})
                selector = action.get('parameters', {}).get('selector', '')
                reasoning = action.get('reasoning', '')
                
                if 'Work' in selector or 'Work' in reasoning:
                    self.details['work_label_clicked'] = True
                    return True
            
            self.issues.append("No action to apply Work label found in trajectory")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Work label action: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Email is in Work label view."""
        try:
            # Check steps where Work label view might be shown (step 57-59)
            work_view_steps = self.find_step_html("step_5[7-9]*") + self.find_step_html("step_6*")
            
            for step_file in work_view_steps:
                is_work_view, emails = self.check_work_label_view(step_file)
                if is_work_view:
                    self.details['work_view_found'] = True
                    self.details['emails_in_work_view'] = len(emails)
                    
                    # Check if our target email is in the Work view
                    for email in emails:
                        if email['sender'] == 'updates@automation.io' and \
                           'Training session invitation' in email['subject']:
                            self.details['target_email_in_work_view'] = True
                            return True
                    
                    # Work view found but email not there
                    self.issues.append(f"Work label view found with {len(emails)} emails, but target email not present")
                    return False
            
            # Also check final state
            final_html = self.find_final_html()
            if final_html:
                is_work_view, emails = self.check_work_label_view(final_html)
                if is_work_view:
                    for email in emails:
                        if email['sender'] == 'updates@automation.io' and \
                           'Training session invitation' in email['subject']:
                            self.details['target_email_in_work_view'] = True
                            return True
            
            self.issues.append("Target email not found in Work label view")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Work label view: {str(e)}")
            return False

    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data from result.json."""
        if not self.result_file.exists():
            return []

        with open(self.result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            return result_data.get('action_history', [])

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_email_found'] = self.check_checkpoint_1()
            self.checkpoints['cp2_work_label_applied'] = self.check_checkpoint_2()
            self.checkpoints['cp3_email_in_work_view'] = self.check_checkpoint_3()

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
        print("Usage: python email_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
