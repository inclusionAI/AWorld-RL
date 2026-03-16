#!/usr/bin/env python3
# Evaluator for email Query 15

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ScheduledEmailHTMLParser(HTMLParser):
    """Parse HTML to extract scheduled email information."""

    def __init__(self):
        super().__init__()
        self.in_scheduled_section = False
        self.in_email_item = False
        self.current_email = {}
        self.scheduled_emails = []
        self.capturing_subject = False
        self.capturing_recipient = False
        self.capturing_date = False
        self.current_text = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in scheduled emails section
        if tag == 'div' and attrs_dict.get('class') == 'scheduled-email-item':
            self.in_email_item = True
            self.current_email = {}
        
        # Check for subject
        if self.in_email_item and tag == 'div' and 'email-subject' in attrs_dict.get('class', ''):
            self.capturing_subject = True
            self.current_text = []
            
        # Check for recipient
        if self.in_email_item and tag == 'div' and 'email-recipient' in attrs_dict.get('class', ''):
            self.capturing_recipient = True
            self.current_text = []
            
        # Check for scheduled date
        if self.in_email_item and tag == 'div' and 'scheduled-date' in attrs_dict.get('class', ''):
            self.capturing_date = True
            self.current_text = []

    def handle_data(self, data):
        if self.capturing_subject or self.capturing_recipient or self.capturing_date:
            self.current_text.append(data.strip())

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.capturing_subject:
                self.current_email['subject'] = ' '.join(self.current_text).strip()
                self.capturing_subject = False
                self.current_text = []
            elif self.capturing_recipient:
                self.current_email['recipient'] = ' '.join(self.current_text).strip()
                self.capturing_recipient = False
                self.current_text = []
            elif self.capturing_date:
                self.current_email['scheduled_date'] = ' '.join(self.current_text).strip()
                self.capturing_date = False
                self.current_text = []
            elif self.in_email_item:
                if self.current_email:
                    self.scheduled_emails.append(self.current_email.copy())
                self.in_email_item = False
                self.current_email = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_scheduled": False,
            "cp2_found_monthly_performance_report": False,
            "cp3_rescheduled_to_correct_datetime": False,
            "cp4_email_sent": False,
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
        """Checkpoint 1: Navigated to Scheduled emails section"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked on Scheduled in the navigation
            for action in action_history:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Scheduled' in selector or 'scheduled' in selector.lower():
                        if action.get('result', {}).get('success'):
                            self.details['scheduled_navigation'] = "Successfully navigated to Scheduled section"
                            return True
            
            self.issues.append("Did not navigate to Scheduled emails section")
            return False
        except Exception as e:
            self.issues.append(f"Error checking scheduled navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Found and attempted to reschedule Monthly Performance Report"""
        try:
            # Check intermediate steps for the scheduled emails list
            step_files = self.find_step_html("step_*")
            
            found_email = False
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    if 'Monthly Performance Report' in html_content:
                        found_email = True
                        self.details['monthly_report_found'] = f"Found in {step_file.name}"
                        break
            
            if not found_email:
                self.issues.append("Monthly Performance Report email not found in scheduled emails")
                return False
            
            # Check if agent attempted to reschedule it
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            attempted_reschedule = False
            for action in action_history:
                reasoning = action.get('action', {}).get('reasoning', '')
                if 'Monthly Performance Report' in reasoning and 'reschedule' in reasoning.lower():
                    attempted_reschedule = True
                    break
            
            if attempted_reschedule:
                self.details['reschedule_attempted'] = "Agent attempted to reschedule Monthly Performance Report"
                return True
            else:
                self.issues.append("Agent did not attempt to reschedule Monthly Performance Report")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking for Monthly Performance Report: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Rescheduled to January 16, 2028 at 8:55 AM"""
        try:
            # Check if the agent tried to modify the date and time fields
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            date_interactions = []
            time_interactions = []
            
            for action in action_history:
                action_type = action.get('action', {}).get('action_type', '')
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                reasoning = action.get('action', {}).get('reasoning', '')
                
                # Check for date field interactions
                if 'input' in selector.lower() and ('date' in selector.lower() or '2026' in selector or '2028' in selector):
                    date_interactions.append(action)
                    
                # Check for time field interactions
                if 'input' in selector.lower() and ('time' in selector.lower() or '8:55' in reasoning or '08:55' in reasoning):
                    time_interactions.append(action)
                    
                # Check reasoning mentions the target date/time
                if '01/16/2028' in reasoning or 'January 16, 2028' in reasoning or '8:55' in reasoning:
                    if action_type in ['TYPE', 'FILL', 'INPUT']:
                        self.details['datetime_modification_attempted'] = "Agent attempted to set date/time"
            
            # Check final HTML for the rescheduled email
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                    # Look for the date in various formats
                    if '01/16/2028' in html_content or 'January 16, 2028' in html_content or '2028-01-16' in html_content:
                        if '8:55' in html_content or '08:55' in html_content:
                            self.details['correct_datetime'] = "Found correct date and time in HTML"
                            return True
            
            # Since agent hit max steps and couldn't complete, this checkpoint fails
            self.issues.append("Agent did not successfully reschedule to January 16, 2028 at 8:55 AM")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking reschedule datetime: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Email was sent after rescheduling"""
        try:
            # Check if agent clicked Send button after rescheduling
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for send action after reschedule actions
            reschedule_found = False
            send_found = False
            
            for action in action_history:
                reasoning = action.get('action', {}).get('reasoning', '')
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                
                if 'reschedule' in reasoning.lower() and 'Monthly Performance Report' in reasoning:
                    reschedule_found = True
                    
                if reschedule_found:
                    if 'send' in selector.lower() or 'send' in reasoning.lower():
                        if action.get('result', {}).get('success'):
                            send_found = True
                            self.details['email_sent'] = "Email sent successfully"
                            break
            
            if not send_found:
                self.issues.append("Email was not sent after rescheduling")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking email send: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_scheduled'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_monthly_performance_report'] = self.check_checkpoint_2()
            self.checkpoints['cp3_rescheduled_to_correct_datetime'] = self.check_checkpoint_3()
            self.checkpoints['cp4_email_sent'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            # Add specific issue about the agent getting stuck
            if not overall_success:
                self.issues.append("Agent exceeded max steps (100) - got stuck in a loop trying to close wrong reschedule dialog")

            return {
                'query_id': 15,
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
                'query_id': 15,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query15.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
