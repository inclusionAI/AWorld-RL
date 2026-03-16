#!/usr/bin/env python3
# Evaluator for email Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class SentEmailParser(HTMLParser):
    """Parse HTML to extract sent emails information."""

    def __init__(self):
        super().__init__()
        self.in_sent_section = False
        self.in_message_list = False
        self.in_message_item = False
        self.in_subject = False
        self.in_recipient = False
        self.current_email = {}
        self.emails = []
        self.email_position = 0
        self.in_toolbar = False
        self.selection_count_text = None
        self.sent_folder_count = None
        self.in_count_span = False
        self.in_sent_link = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in the Sent section
        if tag == 'main':
            for attr_name, attr_value in attrs:
                if attr_name == 'class' and 'sent-main' in attr_value:
                    self.in_sent_section = True
        
        # Check for message list
        if self.in_sent_section and tag == 'div':
            if attrs_dict.get('class') == 'message-list':
                self.in_message_list = True
        
        # Check for individual message items
        if self.in_message_list and tag == 'div':
            if 'message-item' in attrs_dict.get('class', ''):
                self.in_message_item = True
                self.email_position += 1
                self.current_email = {'position': self.email_position}
        
        # Check for subject
        if self.in_message_item and tag == 'div':
            if attrs_dict.get('class') == 'subject':
                self.in_subject = True
        
        # Check for recipient
        if self.in_message_item and tag == 'span':
            if attrs_dict.get('class') == 'recipient':
                self.in_recipient = True
        
        # Check for toolbar
        if self.in_sent_section and tag == 'div':
            if attrs_dict.get('class') == 'toolbar':
                self.in_toolbar = True
        
        # Check for selection count
        if self.in_toolbar and tag == 'span':
            if attrs_dict.get('class') == 'selection-count':
                self.in_count_span = True
        
        # Check for Sent nav link
        if tag == 'a':
            class_val = attrs_dict.get('class', '')
            if 'nav-link' in class_val and 'active' in class_val:
                self.in_sent_link = True

    def handle_endtag(self, tag):
        if tag == 'main':
            self.in_sent_section = False
        if tag == 'div':
            if self.in_message_item:
                if self.current_email:
                    self.emails.append(self.current_email)
                    self.current_email = {}
                self.in_message_item = False
                self.in_subject = False
                self.in_recipient = False
            if self.in_message_list:
                self.in_message_list = False
            if self.in_toolbar:
                self.in_toolbar = False
        if tag == 'span':
            self.in_count_span = False
            self.in_recipient = False
        if tag == 'a':
            self.in_sent_link = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_subject:
            self.current_email['subject'] = data
            self.in_subject = False
        
        if self.in_recipient:
            self.current_email['recipient'] = data
            self.in_recipient = False
        
        if self.in_count_span:
            self.selection_count_text = data
        
        if self.in_sent_link:
            # Looking for count in format "Sent 89"
            if data.isdigit():
                self.sent_folder_count = int(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for the task
        self.checkpoints = {
            "cp1_navigated_to_sent": False,
            "cp2_first_email_deleted": False,
            "cp3_third_email_deleted": False,
            "cp4_only_two_emails_deleted": False,
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

    def parse_sent_emails(self, html_file: Path) -> Tuple[List[Dict], int]:
        """Parse HTML and extract sent emails."""
        parser = SentEmailParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        
        return parser.emails, parser.sent_folder_count or 0

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Sent folder"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            # Check if we're in the Sent folder by looking for "sent-main" class
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'sent-main' in content or 'class="sent-main"' in content:
                    self.details['navigated_to_sent'] = True
                    return True
                else:
                    self.issues.append("Agent did not navigate to Sent folder")
                    return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: First email was deleted"""
        try:
            initial_html = self.find_initial_html()
            final_html = self.find_final_html()
            
            if not initial_html or not final_html:
                self.issues.append("Missing initial or final HTML")
                return False
            
            initial_emails, initial_count = self.parse_sent_emails(initial_html)
            final_emails, final_count = self.parse_sent_emails(final_html)
            
            self.details['initial_sent_count'] = initial_count
            self.details['final_sent_count'] = final_count
            self.details['initial_emails_parsed'] = len(initial_emails)
            self.details['final_emails_parsed'] = len(final_emails)
            
            if len(initial_emails) == 0:
                self.issues.append("Could not parse initial emails")
                return False
            
            # Check if first email is missing in final state
            first_email = initial_emails[0] if initial_emails else None
            if first_email:
                self.details['first_email_subject'] = first_email.get('subject', 'Unknown')
                # Check if this email is no longer present
                first_email_found = any(
                    e.get('subject') == first_email.get('subject') and 
                    e.get('position') == 1 
                    for e in final_emails
                )
                if not first_email_found:
                    return True
                else:
                    self.issues.append(f"First email '{first_email.get('subject')}' was not deleted")
                    return False
            return False
        except Exception as e:
            self.issues.append(f"Error checking first email deletion: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Third email was deleted"""
        try:
            initial_html = self.find_initial_html()
            final_html = self.find_final_html()
            
            if not initial_html or not final_html:
                return False
            
            initial_emails, _ = self.parse_sent_emails(initial_html)
            final_emails, _ = self.parse_sent_emails(final_html)
            
            if len(initial_emails) < 3:
                self.issues.append("Not enough initial emails to check third email")
                return False
            
            # Check if third email is missing in final state
            third_email = initial_emails[2]  # Index 2 is the third email
            self.details['third_email_subject'] = third_email.get('subject', 'Unknown')
            
            # Check if this email is no longer at position 3
            third_email_found = any(
                e.get('subject') == third_email.get('subject') and 
                e.get('position') == 3
                for e in final_emails
            )
            if not third_email_found:
                return True
            else:
                self.issues.append(f"Third email '{third_email.get('subject')}' was not deleted")
                return False
        except Exception as e:
            self.issues.append(f"Error checking third email deletion: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Only two emails were deleted (not more)"""
        try:
            initial_html = self.find_initial_html()
            final_html = self.find_final_html()
            
            if not initial_html or not final_html:
                return False
            
            initial_emails, initial_count = self.parse_sent_emails(initial_html)
            final_emails, final_count = self.parse_sent_emails(final_html)
            
            # Check using folder count
            expected_final_count = initial_count - 2
            if final_count == expected_final_count:
                return True
            elif final_count > expected_final_count:
                self.issues.append(f"Too few emails deleted: expected {initial_count - final_count} deleted, should be 2")
                return False
            elif final_count < expected_final_count:
                self.issues.append(f"Too many emails deleted: {initial_count - final_count} emails deleted instead of 2")
                return False
            
            return False
        except Exception as e:
            self.issues.append(f"Error checking email count: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_sent'] = self.check_checkpoint_1()
            self.checkpoints['cp2_first_email_deleted'] = self.check_checkpoint_2()
            self.checkpoints['cp3_third_email_deleted'] = self.check_checkpoint_3()
            self.checkpoints['cp4_only_two_emails_deleted'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 7,
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
                'query_id': 7,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
