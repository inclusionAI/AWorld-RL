#!/usr/bin/env python3
# Evaluator for email Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EmailHTMLParser(HTMLParser):
    """Parse HTML to extract email information from Sent folder."""

    def __init__(self):
        super().__init__()
        self.in_sent_main = False
        self.in_message_item = False
        self.in_recipient = False
        self.in_subject = False
        self.in_preview = False
        self.current_email = {}
        self.emails = []
        self.in_nav_link = False
        self.current_nav_text = ""
        self.nav_link_active = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in Sent main section
        if tag == "main" and attrs_dict.get("class") == "sent-main":
            self.in_sent_main = True
        
        # Check for message items in Sent folder
        if self.in_sent_main and tag == "div" and "message-item" in attrs_dict.get("class", ""):
            self.in_message_item = True
            self.current_email = {}
        
        # Check for recipient span
        if self.in_message_item and tag == "span" and attrs_dict.get("class") == "recipient":
            self.in_recipient = True
        
        # Check for subject div
        if self.in_message_item and tag == "div" and attrs_dict.get("class") == "subject":
            self.in_subject = True
        
        # Check for preview div
        if self.in_message_item and tag == "div" and attrs_dict.get("class") == "preview":
            self.in_preview = True
        
        # Check for active Sent nav link
        if tag == "a" and "nav-link" in attrs_dict.get("class", ""):
            self.in_nav_link = True
            self.current_nav_text = ""
            self.nav_link_active = "active" in attrs_dict.get("class", "")

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_recipient:
            self.current_email["recipient"] = data
        elif self.in_subject:
            self.current_email["subject"] = data
        elif self.in_preview:
            self.current_email["preview"] = data
        elif self.in_nav_link:
            self.current_nav_text += data

    def handle_endtag(self, tag):
        if tag == "main" and self.in_sent_main:
            self.in_sent_main = False
        
        if tag == "div" and self.in_message_item:
            if self.current_email:
                self.emails.append(self.current_email)
                self.current_email = {}
            self.in_message_item = False
        
        if tag == "span" and self.in_recipient:
            self.in_recipient = False
        
        if tag == "div" and self.in_subject:
            self.in_subject = False
        
        if tag == "div" and self.in_preview:
            self.in_preview = False
        
        if tag == "a" and self.in_nav_link:
            self.in_nav_link = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for Query 1: Compose and send email
        self.checkpoints = {
            "cp1_compose_opened": False,
            "cp2_recipient_filled": False,
            "cp3_subject_filled": False,
            "cp4_body_filled": False,
            "cp5_email_sent": False,
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
            actions = result_data.get("action_history", [])
            
            # Check if first action was clicking Compose
            if actions and len(actions) > 0:
                first_action = actions[0].get("action", {})
                if first_action.get("action_type") == "CLICK":
                    selector = first_action.get("parameters", {}).get("selector", "")
                    if "Compose" in selector or "compose" in selector.lower():
                        self.details["compose_clicked"] = True
                        return True
            
            self.issues.append("Compose button was not clicked")
            return False
        except Exception as e:
            self.issues.append(f"Error checking compose action: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Recipient email was filled correctly"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            # Look for FILL action with recipient email
            for action_item in actions:
                action = action_item.get("action", {})
                if action.get("action_type") == "FILL":
                    params = action.get("parameters", {})
                    text = params.get("text", "")
                    selector = params.get("selector", "")
                    
                    if "john.doe@example.com" in text and "Recipients" in selector:
                        self.details["recipient_email"] = text
                        return True
            
            self.issues.append("Recipient email 'john.doe@example.com' was not filled correctly")
            return False
        except Exception as e:
            self.issues.append(f"Error checking recipient: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Subject was filled correctly"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            # Look for FILL action with subject
            for action_item in actions:
                action = action_item.get("action", {})
                if action.get("action_type") == "FILL":
                    params = action.get("parameters", {})
                    text = params.get("text", "")
                    selector = params.get("selector", "")
                    
                    if "Project Update" == text and "subject" in selector.lower():
                        self.details["subject"] = text
                        return True
            
            self.issues.append("Subject 'Project Update' was not filled correctly")
            return False
        except Exception as e:
            self.issues.append(f"Error checking subject: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Email body was filled correctly"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            # Look for TYPE action with body text
            for action_item in actions:
                action = action_item.get("action", {})
                if action.get("action_type") == "TYPE":
                    params = action.get("parameters", {})
                    text = params.get("text", "")
                    
                    if "The project is on track." == text:
                        self.details["body"] = text
                        return True
            
            self.issues.append("Email body 'The project is on track.' was not typed correctly")
            return False
        except Exception as e:
            self.issues.append(f"Error checking body: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Email was sent and appears in Sent folder"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = EmailHTMLParser()
            parser.feed(html_content)
            
            # Check if we're viewing the Sent folder (looking for active nav link)
            if "sent-main" not in html_content.lower() and 'class="nav-link active"' not in html_content:
                self.issues.append("Not viewing Sent folder in final state")
                return False
            
            # Look for the sent email with correct details
            found_email = False
            for email in parser.emails:
                recipient = email.get("recipient", "")
                subject = email.get("subject", "")
                preview = email.get("preview", "")
                
                if ("john.doe@example.com" in recipient and 
                    "Project Update" == subject and 
                    "The project is on track" in preview):
                    found_email = True
                    self.details["sent_email_found"] = True
                    self.details["sent_email_details"] = email
                    break
            
            if not found_email:
                self.issues.append("Email with correct recipient, subject, and body not found in Sent folder")
                return False
            
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking sent email: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_compose_opened'] = self.check_checkpoint_1()
            self.checkpoints['cp2_recipient_filled'] = self.check_checkpoint_2()
            self.checkpoints['cp3_subject_filled'] = self.check_checkpoint_3()
            self.checkpoints['cp4_body_filled'] = self.check_checkpoint_4()
            self.checkpoints['cp5_email_sent'] = self.check_checkpoint_5()

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
        print("Usage: python email_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
