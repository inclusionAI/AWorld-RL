#!/usr/bin/env python3
# Evaluator for email Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class SentEmailParser(HTMLParser):
    """Parse HTML to extract sent emails information."""

    def __init__(self):
        super().__init__()
        self.emails = []
        self.current_email = {}
        self.in_message_item = False
        self.in_subject = False
        self.in_recipient = False
        self.current_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "div" and attrs_dict.get("class") == "message-item":
            self.in_message_item = True
            self.current_email = {}
        elif self.in_message_item:
            if tag == "div" and attrs_dict.get("class") == "subject":
                self.in_subject = True
                self.current_text = ""
            elif tag == "span" and attrs_dict.get("class") == "recipient":
                self.in_recipient = True
                self.current_text = ""

    def handle_data(self, data):
        if self.in_subject or self.in_recipient:
            self.current_text += data.strip()

    def handle_endtag(self, tag):
        if tag == "div":
            if self.in_subject:
                self.current_email["subject"] = self.current_text
                self.in_subject = False
            elif self.in_message_item and "subject" in self.current_email:
                self.emails.append(self.current_email.copy())
                self.in_message_item = False
        elif tag == "span" and self.in_recipient:
            self.current_email["recipient"] = self.current_text
            self.in_recipient = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_sent": False,
            "cp2_first_email_deleted": False,
            "cp3_third_email_deleted": False,
            "cp4_second_email_archived": False,
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

    def find_initial_html(self) -> Optional[Path]:
        """Find initial HTML state file."""
        initial_files = list(self.result_dir.glob("initial_*_raw.html"))
        if initial_files:
            return initial_files[0]
        return None

    def parse_sent_emails(self, html_path: Path) -> List[Dict]:
        """Parse sent emails from HTML."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = SentEmailParser()
            parser.feed(html_content)
            return parser.emails
        except Exception as e:
            self.issues.append(f"Error parsing HTML: {str(e)}")
            return []

    def check_sent_folder_displayed(self, html_path: Path) -> bool:
        """Check if Sent folder is displayed."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            # Check for active Sent nav link
            return 'class="nav-link active"' in html_content and '>Sent<' in html_content
        except Exception:
            return False

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Sent folder"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        is_in_sent = self.check_sent_folder_displayed(final_html)
        if is_in_sent:
            self.details["sent_folder_displayed"] = True
        else:
            self.issues.append("Sent folder is not displayed")
            self.details["sent_folder_displayed"] = False
        
        return is_in_sent

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: First email (Weekly Report) deleted"""
        initial_html = self.find_initial_html()
        final_html = self.find_final_html()
        
        if not initial_html or not final_html:
            self.issues.append("Missing initial or final HTML")
            return False
        
        # Parse emails
        initial_emails = self.parse_sent_emails(initial_html)
        final_emails = self.parse_sent_emails(final_html)
        
        self.details["initial_email_count"] = len(initial_emails)
        self.details["final_email_count"] = len(final_emails)
        
        if len(initial_emails) == 0:
            self.issues.append("No initial emails found")
            return False
        
        # First email should be "Weekly Report"
        first_email = initial_emails[0] if initial_emails else {}
        self.details["first_email_subject"] = first_email.get("subject", "")
        
        # Check if first email is deleted (not present in final)
        first_email_still_exists = any(
            email.get("subject") == "Weekly Report" and 
            email.get("recipient") == first_email.get("recipient")
            for email in final_emails
        )
        
        if first_email_still_exists:
            self.issues.append("First email 'Weekly Report' was not deleted")
            self.details["first_email_deleted"] = False
            return False
        
        self.details["first_email_deleted"] = True
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Third email (Project Proposal) deleted"""
        initial_html = self.find_initial_html()
        final_html = self.find_final_html()
        
        if not initial_html or not final_html:
            return False
        
        initial_emails = self.parse_sent_emails(initial_html)
        final_emails = self.parse_sent_emails(final_html)
        
        if len(initial_emails) < 3:
            self.issues.append("Less than 3 initial emails found")
            return False
        
        # Third email should be "Project Proposal"
        third_email = initial_emails[2]
        self.details["third_email_subject"] = third_email.get("subject", "")
        
        # Check if third email is deleted
        third_email_still_exists = any(
            email.get("subject") == "Project Proposal" and 
            email.get("recipient") == third_email.get("recipient")
            for email in final_emails
        )
        
        if third_email_still_exists:
            self.issues.append("Third email 'Project Proposal' was not deleted")
            self.details["third_email_deleted"] = False
            return False
        
        self.details["third_email_deleted"] = True
        return True

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Second email (Follow-up on Meeting) archived"""
        initial_html = self.find_initial_html()
        final_html = self.find_final_html()
        
        if not initial_html or not final_html:
            return False
        
        initial_emails = self.parse_sent_emails(initial_html)
        final_emails = self.parse_sent_emails(final_html)
        
        if len(initial_emails) < 2:
            self.issues.append("Less than 2 initial emails found")
            return False
        
        # Second email should be "Follow-up on Meeting"
        second_email = initial_emails[1]
        self.details["second_email_subject"] = second_email.get("subject", "")
        
        # Check if second email is NOT in Sent folder (archived)
        second_email_in_sent = any(
            email.get("subject") == "Follow-up on Meeting" and 
            email.get("recipient") == second_email.get("recipient")
            for email in final_emails
        )
        
        if second_email_in_sent:
            self.issues.append("Second email 'Follow-up on Meeting' was not archived (still in Sent)")
            self.details["second_email_archived"] = False
            return False
        
        self.details["second_email_archived"] = True
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_sent'] = self.check_checkpoint_1()
            self.checkpoints['cp2_first_email_deleted'] = self.check_checkpoint_2()
            self.checkpoints['cp3_third_email_deleted'] = self.check_checkpoint_3()
            self.checkpoints['cp4_second_email_archived'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 8,
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
                'query_id': 8,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
