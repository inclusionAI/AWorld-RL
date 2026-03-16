#!/usr/bin/env python3
# Evaluator for email Query 4

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
        self.in_sent_folder = False
        self.sent_emails = []
        self.current_email = {}
        self.current_tag = None
        self.in_nav_link = False
        self.nav_link_text = ""
        self.nav_link_class = ""
        self.in_message_item = False
        self.in_message_header = False
        self.in_subject = False
        self.in_recipient = False
        self.capture_text = False
        self.current_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for nav links to determine current folder
        if tag == 'a' and 'nav-link' in attrs_dict.get('class', ''):
            self.in_nav_link = True
            self.nav_link_class = attrs_dict.get('class', '')
            self.nav_link_text = ""
        
        # Check for message items in Sent folder
        if tag == 'div' and 'message-item' in attrs_dict.get('class', ''):
            self.in_message_item = True
            self.current_email = {}
        
        # Check for message header (recipient and time)
        if self.in_message_item and tag == 'div' and 'message-header' in attrs_dict.get('class', ''):
            self.in_message_header = True
        
        # Check for recipient
        if self.in_message_header and tag == 'span' and 'recipient' in attrs_dict.get('class', ''):
            self.in_recipient = True
            self.current_text = ""
        
        # Check for subject
        if self.in_message_item and tag == 'div' and 'subject' in attrs_dict.get('class', ''):
            self.in_subject = True
            self.current_text = ""

    def handle_data(self, data):
        if self.in_nav_link:
            self.nav_link_text += data.strip()
        elif self.in_recipient:
            self.current_text += data.strip()
        elif self.in_subject:
            self.current_text += data.strip()

    def handle_endtag(self, tag):
        if tag == 'a' and self.in_nav_link:
            if 'Sent' in self.nav_link_text and 'active' in self.nav_link_class:
                self.in_sent_folder = True
            self.in_nav_link = False
        
        if tag == 'span' and self.in_recipient:
            self.current_email['recipient'] = self.current_text
            self.in_recipient = False
        
        if tag == 'div' and self.in_subject:
            self.current_email['subject'] = self.current_text
            self.in_subject = False
        
        if tag == 'div' and self.in_message_item:
            if self.current_email:
                self.sent_emails.append(self.current_email)
            self.in_message_item = False
            self.in_message_header = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_email_opened": False,
            "cp2_reply_clicked": False,
            "cp3_acknowledged_typed": False,
            "cp4_reply_sent": False,
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
        """Find intermediate step HTML files.

        Args:
            step_pattern: Pattern like "step_*", "step_10_*", "step_[12]*"
        Returns:
            List of matching HTML files sorted by step number
        """
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

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Email from tom@friend.com titled 'Lunch Plans' was opened"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if there was an action to click on email with "Lunch Plans" and "tom@friend.com"
            for action_item in action_history:
                action = action_item.get('action', {})
                params = action.get('parameters', {})
                selector = params.get('selector', '')
                
                if 'Lunch Plans' in selector and 'tom@friend.com' in selector:
                    result = action_item.get('result', {})
                    if result.get('success'):
                        self.details['email_opened'] = True
                        return True
            
            self.issues.append("Email titled 'Lunch Plans' from tom@friend.com was not opened")
            return False
        except Exception as e:
            self.issues.append(f"Error checking if email was opened: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Reply button was clicked"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if there was an action to click Reply button
            for action_item in action_history:
                action = action_item.get('action', {})
                if action.get('action_type') == 'CLICK':
                    params = action.get('parameters', {})
                    selector = params.get('selector', '')
                    
                    if 'Reply' in selector:
                        result = action_item.get('result', {})
                        if result.get('success'):
                            self.details['reply_clicked'] = True
                            return True
            
            self.issues.append("Reply button was not clicked")
            return False
        except Exception as e:
            self.issues.append(f"Error checking if Reply was clicked: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: 'Acknowledged' was typed in the compose area"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if "Acknowledged" was typed
            for action_item in action_history:
                action = action_item.get('action', {})
                if action.get('action_type') == 'TYPE':
                    params = action.get('parameters', {})
                    text = params.get('text', '')
                    
                    if 'Acknowledged' in text:
                        result = action_item.get('result', {})
                        if result.get('success'):
                            self.details['acknowledged_typed'] = True
                            return True
            
            self.issues.append("'Acknowledged' was not typed in the reply")
            return False
        except Exception as e:
            self.issues.append(f"Error checking if 'Acknowledged' was typed: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Reply was sent and appears in Sent folder"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if we're in Sent folder and the reply exists
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            if not parser.in_sent_folder:
                self.issues.append("Not viewing Sent folder in final state")
                return False
            
            # Look for the reply email with subject "Re: Lunch Plans" or "Lunch PlansAcknowledged"
            # and recipient "tom@friend.com"
            for email in parser.sent_emails:
                recipient = email.get('recipient', '')
                subject = email.get('subject', '')
                
                if 'tom@friend.com' in recipient:
                    if 'Lunch Plans' in subject and 'Acknowledged' in subject:
                        self.details['reply_in_sent'] = True
                        self.details['reply_subject'] = subject
                        self.details['reply_recipient'] = recipient
                        return True
                    elif 'Re: Lunch Plans' in subject:
                        # Also check if body contains Acknowledged
                        self.details['reply_in_sent'] = True
                        self.details['reply_subject'] = subject
                        self.details['reply_recipient'] = recipient
                        return True
            
            self.issues.append("Reply to tom@friend.com with 'Lunch Plans' not found in Sent folder")
            return False
        except Exception as e:
            self.issues.append(f"Error checking if reply was sent: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_email_opened'] = self.check_checkpoint_1()
            self.checkpoints['cp2_reply_clicked'] = self.check_checkpoint_2()
            self.checkpoints['cp3_acknowledged_typed'] = self.check_checkpoint_3()
            self.checkpoints['cp4_reply_sent'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 4,
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
                'query_id': 4,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query4.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
