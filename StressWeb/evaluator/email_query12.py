#!/usr/bin/env python3
# Evaluator for email Query 12

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ContactHTMLParser(HTMLParser):
    """Parse HTML to extract contact information from the contacts page."""

    def __init__(self):
        super().__init__()
        self.contacts = []
        self.current_contact = {}
        self.in_contact_card = False
        self.in_contact_name = False
        self.in_contact_email = False
        self.in_contact_detail = False
        self.current_detail_type = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict:
            if 'contact-card' in attrs_dict['class']:
                self.in_contact_card = True
                self.current_contact = {}
            elif 'contact-name' in attrs_dict['class']:
                self.in_contact_name = True
            elif 'contact-email' in attrs_dict['class']:
                self.in_contact_email = True
            elif 'contact-detail' in attrs_dict['class']:
                self.in_contact_detail = True
        
        if tag == 'i' and self.in_contact_detail and 'class' in attrs_dict:
            icon_class = attrs_dict['class']
            if 'fa-birthday-cake' in icon_class:
                self.current_detail_type = 'birthday'
            elif 'fa-phone' in icon_class:
                self.current_detail_type = 'phone'
            elif 'fa-building' in icon_class:
                self.current_detail_type = 'company'

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_contact_name:
                self.in_contact_name = False
            elif self.in_contact_email:
                self.in_contact_email = False
            elif self.in_contact_detail:
                self.in_contact_detail = False
                self.current_detail_type = None
            elif self.in_contact_card:
                self.in_contact_card = False
                if self.current_contact:
                    self.contacts.append(self.current_contact)
                self.current_contact = {}

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        if self.in_contact_name:
            self.current_contact['name'] = text
        elif self.in_contact_email:
            self.current_contact['email'] = text
        elif self.in_contact_detail and self.current_detail_type:
            self.current_contact[self.current_detail_type] = text


class SentEmailHTMLParser(HTMLParser):
    """Parse HTML to extract sent email information."""

    def __init__(self):
        super().__init__()
        self.sent_emails = []
        self.current_email = {}
        self.in_message_item = False
        self.in_recipient = False
        self.in_subject = False
        self.in_preview = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict:
            if 'message-item' in attrs_dict['class']:
                self.in_message_item = True
                self.current_email = {}
        
        if tag == 'span' and 'class' in attrs_dict:
            if 'recipient' in attrs_dict['class']:
                self.in_recipient = True
        
        if tag == 'div' and 'class' in attrs_dict:
            if 'subject' in attrs_dict['class'] and self.in_message_item:
                self.in_subject = True
            elif 'preview' in attrs_dict['class'] and self.in_message_item:
                self.in_preview = True

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_subject:
                self.in_subject = False
            elif self.in_preview:
                self.in_preview = False
            elif self.in_message_item:
                self.in_message_item = False
                if self.current_email:
                    self.sent_emails.append(self.current_email)
                self.current_email = {}
        elif tag == 'span' and self.in_recipient:
            self.in_recipient = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        if self.in_recipient:
            # Extract email from "To: email@example.com"
            if text.startswith('To:'):
                email = text.replace('To:', '').strip()
                self.current_email['recipient'] = email
        elif self.in_subject:
            self.current_email['subject'] = text
        elif self.in_preview:
            self.current_email['preview'] = text


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_contacts": False,
            "cp2_contact_added_with_correct_info": False,
            "cp3_email_composed_and_sent": False,
            "cp4_email_in_sent_folder": False,
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
        """Checkpoint 1: Navigated to contacts page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked on Contacts link
            contacts_clicked = False
            for entry in trajectory:
                action = entry.get('action', {})
                selector = action.get('parameters', {}).get('selector', '')
                
                if 'Contacts' in str(selector) or 'contact' in str(selector).lower():
                    contacts_clicked = True
                    break
            
            if not contacts_clicked:
                self.issues.append("Agent did not navigate to Contacts page")
                return False
            
            self.details['contacts_navigation'] = "Successfully navigated to Contacts"
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking contacts navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Contact added with correct information"""
        try:
            # Check step 7 where contact should be saved and visible in contacts list
            step_7_files = self.find_step_html("step_7_*")
            if not step_7_files:
                self.issues.append("Step 7 HTML not found - cannot verify contact was saved")
                return False
            
            with open(step_7_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = ContactHTMLParser()
            parser.feed(html_content)
            
            # Find Emma Wilson contact
            emma_contact = None
            for contact in parser.contacts:
                if 'Emma Wilson' in contact.get('name', ''):
                    emma_contact = contact
                    break
            
            if not emma_contact:
                self.issues.append("Emma Wilson contact not found in contacts list")
                return False
            
            # Verify email
            if emma_contact.get('email') != 'emma.w@consulting.com':
                self.issues.append(f"Emma Wilson email is incorrect: {emma_contact.get('email')}")
                return False
            
            # Verify birthday (format might be "Mar 15, 1990" in display)
            birthday = emma_contact.get('birthday', '')
            if not ('Mar 15' in birthday and '1990' in birthday):
                self.issues.append(f"Emma Wilson birthday is incorrect: {birthday}")
                return False
            
            self.details['emma_contact'] = emma_contact
            self.details['contact_verification'] = "Emma Wilson added with correct email and birthday"
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking contact information: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Email composed and sent"""
        try:
            trajectory = self.get_trajectory()
            
            # Check that compose button was clicked
            compose_clicked = False
            recipient_filled = False
            subject_filled = False
            body_typed = False
            send_clicked = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                action_type = action.get('action_type', '')
                params = action.get('parameters', {})
                
                if action_type == 'CLICK':
                    selector = params.get('selector', '')
                    if 'Compose' in selector:
                        compose_clicked = True
                    elif 'Send' in selector:
                        send_clicked = True
                
                elif action_type == 'FILL':
                    text = params.get('text', '')
                    selector = params.get('selector', '')
                    
                    if 'emma.w@consulting.com' in text:
                        recipient_filled = True
                    elif 'Happy Birthday' in text:
                        subject_filled = True
                
                elif action_type == 'TYPE':
                    text = params.get('text', '')
                    if 'Happy Birthday! Wishing you a wonderful day.' in text:
                        body_typed = True
            
            missing_steps = []
            if not compose_clicked:
                missing_steps.append("compose button not clicked")
            if not recipient_filled:
                missing_steps.append("recipient not filled")
            if not subject_filled:
                missing_steps.append("subject not filled")
            if not body_typed:
                missing_steps.append("body not typed")
            if not send_clicked:
                missing_steps.append("send button not clicked")
            
            if missing_steps:
                self.issues.append(f"Email composition incomplete: {', '.join(missing_steps)}")
                return False
            
            self.details['email_composition'] = "All email composition steps completed"
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking email composition: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Email appears in Sent folder with correct details"""
        try:
            final_html_file = self.find_final_html()
            if not final_html_file:
                self.issues.append("Final HTML not found")
                return False
            
            with open(final_html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check that we're on the Sent page
            if 'sent-main' not in html_content and 'Sent' not in html_content:
                self.issues.append("Final page is not the Sent folder")
                return False
            
            parser = SentEmailHTMLParser()
            parser.feed(html_content)
            
            # Find the email to Emma Wilson
            emma_email = None
            for email in parser.sent_emails:
                if email.get('recipient') == 'emma.w@consulting.com':
                    emma_email = email
                    break
            
            if not emma_email:
                self.issues.append("Email to emma.w@consulting.com not found in Sent folder")
                return False
            
            # Verify subject
            if emma_email.get('subject') != 'Happy Birthday':
                self.issues.append(f"Email subject is incorrect: {emma_email.get('subject')}")
                return False
            
            # Verify preview/body starts with expected text
            preview = emma_email.get('preview', '')
            if 'Happy Birthday! Wishing you a wonderful day' not in preview:
                self.issues.append(f"Email body preview is incorrect: {preview}")
                return False
            
            self.details['sent_email'] = emma_email
            self.details['sent_verification'] = "Email found in Sent folder with correct details"
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking sent email: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_contacts'] = self.check_checkpoint_1()
            self.checkpoints['cp2_contact_added_with_correct_info'] = self.check_checkpoint_2()
            self.checkpoints['cp3_email_composed_and_sent'] = self.check_checkpoint_3()
            self.checkpoints['cp4_email_in_sent_folder'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 12,
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
                'query_id': 12,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
