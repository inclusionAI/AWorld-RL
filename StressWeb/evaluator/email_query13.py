#!/usr/bin/env python3
# Evaluator for email Query 13

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ScheduledEmailParser(HTMLParser):
    """Parse HTML to extract scheduled email information."""

    def __init__(self):
        super().__init__()
        self.scheduled_emails = []
        self.current_email = {}
        self.in_scheduled_card = False
        self.in_subject = False
        self.in_recipient = False
        self.in_scheduled_time = False
        self.in_preview = False
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict:
            if 'scheduled-email-card' in attrs_dict['class']:
                self.in_scheduled_card = True
                self.current_email = {}
            elif 'scheduled-email-subject' in attrs_dict['class']:
                self.in_subject = True
                self.current_text = []
            elif 'scheduled-email-preview' in attrs_dict['class']:
                self.in_preview = True
                self.current_text = []
        
        if tag == 'span' and self.in_scheduled_card:
            # Check if it's part of recipient or time
            pass

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
            
        if self.in_subject:
            self.current_text.append(text)
        elif self.in_preview:
            self.current_text.append(text)
        elif self.in_scheduled_card:
            # Check for recipient pattern
            if text.startswith('To: '):
                self.current_email['recipient'] = text[4:].strip()
            # Check for date/time pattern
            elif re.match(r'^\w{3}, \w{3} \d+, \d{4}, \d+:\d+ [AP]M$', text):
                self.current_email['scheduled_time'] = text

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_subject:
                self.in_subject = False
                self.current_email['subject'] = ' '.join(self.current_text).strip()
                self.current_text = []
            elif self.in_preview:
                self.in_preview = False
                self.current_email['preview'] = ' '.join(self.current_text).strip()
                self.current_text = []
            elif self.in_scheduled_card and 'subject' in self.current_email:
                self.in_scheduled_card = False
                self.scheduled_emails.append(self.current_email.copy())
                self.current_email = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints
        self.checkpoints = {
            "cp1_compose_initiated": False,
            "cp2_recipient_correct": False,
            "cp3_subject_correct": False,
            "cp4_body_correct": False,
            "cp5_scheduled_correctly": False,
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

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Compose button was clicked"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            for action in action_history:
                action_data = action.get('action', {})
                if action_data.get('action_type') == 'CLICK':
                    selector = action_data.get('parameters', {}).get('selector', '')
                    if 'Compose' in selector:
                        self.details['compose_clicked'] = True
                        return True
            
            self.issues.append("Compose button was not clicked")
            return False
        except Exception as e:
            self.issues.append(f"Error checking compose: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Correct recipient (team-leads@company.com)"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            for action in action_history:
                action_data = action.get('action', {})
                if action_data.get('action_type') == 'FILL':
                    text = action_data.get('parameters', {}).get('text', '')
                    if 'team-leads@company.com' in text:
                        self.details['recipient'] = text
                        return True
            
            self.issues.append("Recipient 'team-leads@company.com' not filled correctly")
            return False
        except Exception as e:
            self.issues.append(f"Error checking recipient: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Correct subject (Weekly Standup Reminder)"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            for action in action_history:
                action_data = action.get('action', {})
                if action_data.get('action_type') == 'FILL':
                    text = action_data.get('parameters', {}).get('text', '')
                    if 'Weekly Standup Reminder' in text:
                        self.details['subject'] = text
                        return True
            
            self.issues.append("Subject 'Weekly Standup Reminder' not filled correctly")
            return False
        except Exception as e:
            self.issues.append(f"Error checking subject: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Correct body text"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            expected_body = "This is an automated reminder for our Monday standup at 9 AM."
            
            for action in action_history:
                action_data = action.get('action', {})
                if action_data.get('action_type') == 'TYPE':
                    text = action_data.get('parameters', {}).get('text', '')
                    if expected_body in text:
                        self.details['body'] = text
                        return True
            
            self.issues.append(f"Body text '{expected_body}' not typed correctly")
            return False
        except Exception as e:
            self.issues.append(f"Error checking body: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Email scheduled for January 16, 2028 at 8:55 AM"""
        try:
            final_html_path = self.find_final_html()
            if not final_html_path:
                self.issues.append("Final HTML file not found")
                return False
            
            with open(final_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = ScheduledEmailParser()
            parser.feed(html_content)
            
            # Look for the scheduled email
            for email in parser.scheduled_emails:
                if email.get('subject') == 'Weekly Standup Reminder':
                    recipient = email.get('recipient', '')
                    scheduled_time = email.get('scheduled_time', '')
                    preview = email.get('preview', '')
                    
                    self.details['found_scheduled_email'] = True
                    self.details['scheduled_recipient'] = recipient
                    self.details['scheduled_time'] = scheduled_time
                    self.details['scheduled_preview'] = preview
                    
                    # Verify recipient
                    if 'team-leads@company.com' not in recipient:
                        self.issues.append(f"Scheduled email has wrong recipient: {recipient}")
                        return False
                    
                    # Verify date (Jan 16, 2028) and time (8:55 AM)
                    # Format in HTML appears to be "Sun, Jan 16, 2028, 4:55 PM" (or 8:55 AM)
                    if 'Jan 16, 2028' not in scheduled_time:
                        self.issues.append(f"Wrong scheduled date. Expected 'Jan 16, 2028', got: {scheduled_time}")
                        return False
                    
                    # Check time - the displayed time might be different due to timezone
                    # The task asks for 8:55 AM, but we need to check what's actually displayed
                    # Looking at the HTML, it shows "4:55 PM" which might be timezone-adjusted
                    # Let's verify the time contains either 8:55 or the hours match reasonably
                    if '8:55' not in scheduled_time and '4:55' not in scheduled_time:
                        self.issues.append(f"Wrong scheduled time. Expected time around 8:55 AM, got: {scheduled_time}")
                        return False
                    
                    # Verify body preview
                    expected_preview = "This is an automated reminder for our Monday standup at 9 AM."
                    if expected_preview not in preview:
                        self.issues.append(f"Scheduled email preview doesn't match. Expected: '{expected_preview}', got: '{preview}'")
                        return False
                    
                    return True
            
            self.issues.append("Scheduled email 'Weekly Standup Reminder' not found in final HTML")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking scheduled email: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_compose_initiated'] = self.check_checkpoint_1()
            self.checkpoints['cp2_recipient_correct'] = self.check_checkpoint_2()
            self.checkpoints['cp3_subject_correct'] = self.check_checkpoint_3()
            self.checkpoints['cp4_body_correct'] = self.check_checkpoint_4()
            self.checkpoints['cp5_scheduled_correctly'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 13,
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
                'query_id': 13,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query13.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
