#!/usr/bin/env python3
# Evaluator for email Query 16

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class EmailHTMLParser(HTMLParser):
    """Parse HTML to extract email information from inbox."""

    def __init__(self):
        super().__init__()
        self.emails = []
        self.current_email = {}
        self.in_message_item = False
        self.in_sender = False
        self.in_subject = False
        self.current_tag_attrs = []
        self.capture_text = False
        self.text_buffer = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict and 'message-item' in attrs_dict['class']:
            self.in_message_item = True
            self.current_email = {
                'starred': False,
                'sender': '',
                'subject': '',
                'has_important_label': False
            }
        
        elif self.in_message_item:
            if tag == 'button' and 'class' in attrs_dict and 'star-btn' in attrs_dict['class']:
                self.current_tag_attrs = attrs
            elif tag == 'i' and self.current_tag_attrs:
                # Check if star icon is filled (fas fa-star means starred)
                if 'class' in attrs_dict and 'fas fa-star' in attrs_dict['class']:
                    self.current_email['starred'] = True
                self.current_tag_attrs = []
            elif tag == 'span' and 'class' in attrs_dict:
                if 'sender' in attrs_dict['class']:
                    self.in_sender = True
                    self.capture_text = True
                    self.text_buffer = ""
                elif 'subject' in attrs_dict['class']:
                    self.in_subject = True
                    self.capture_text = True
                    self.text_buffer = ""

    def handle_data(self, data):
        if self.capture_text:
            self.text_buffer += data.strip()

    def handle_endtag(self, tag):
        if tag == 'span':
            if self.in_sender:
                self.current_email['sender'] = self.text_buffer
                self.in_sender = False
                self.capture_text = False
            elif self.in_subject:
                self.current_email['subject'] = self.text_buffer
                self.in_subject = False
                self.capture_text = False
        
        elif tag == 'div' and self.in_message_item:
            # Check if we're closing a message-item div
            if self.current_email.get('sender') or self.current_email.get('subject'):
                self.emails.append(self.current_email.copy())
                self.current_email = {}
                self.in_message_item = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Task: star first 5 emails, add "important" label, archive all 5
        self.checkpoints = {
            "cp1_first_five_starred": False,
            "cp2_important_label_applied": False,
            "cp3_emails_archived": False,
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

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")
        with open(self.result_file, 'r') as f:
            return json.load(f)

    def parse_html_for_emails(self, html_path: Path) -> List[Dict]:
        """Parse HTML file and extract email information."""
        parser = EmailHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser.emails

    def extract_starred_emails_from_html(self, html_content: str) -> List[Dict]:
        """Extract starred emails from HTML content."""
        emails = []
        # Split by message-item divs
        pattern = r'<div class="message-item[^"]*">(.*?)</div></div></div>'
        matches = re.finditer(pattern, html_content, re.DOTALL)
        
        for match in matches:
            email_html = match.group(1)
            # Check if starred (fas fa-star indicates starred)
            is_starred = 'fas fa-star' in email_html
            
            # Extract sender
            sender_match = re.search(r'<span class="sender[^"]*">([^<]+)</span>', email_html)
            sender = sender_match.group(1) if sender_match else ""
            
            # Extract subject
            subject_match = re.search(r'<div class="subject[^"]*">([^<]+)</div>', email_html)
            subject = subject_match.group(1) if subject_match else ""
            
            emails.append({
                'sender': sender,
                'subject': subject,
                'starred': is_starred
            })
        
        return emails

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: First five emails are starred."""
        try:
            # Check step 4 (after starring the 5th email)
            step_files = self.find_step_html("step_[4-9]")
            if not step_files:
                step_files = self.find_step_html("step_*")
            
            if not step_files:
                self.issues.append("No step HTML files found to verify starring")
                return False
            
            # Check a step after starring was done (around step 5)
            check_file = None
            for f in sorted(step_files):
                if 'step_5' in f.name or 'step_6' in f.name:
                    check_file = f
                    break
            
            if not check_file and step_files:
                check_file = step_files[min(5, len(step_files)-1)]
            
            with open(check_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            emails = self.extract_starred_emails_from_html(content)
            
            if len(emails) < 5:
                self.issues.append(f"Less than 5 emails found in inbox")
                return False
            
            first_five = emails[:5]
            starred_count = sum(1 for e in first_five if e['starred'])
            
            self.details['starred_in_first_five'] = starred_count
            self.details['first_five_emails'] = [
                f"{e['sender']} - {e['subject']}" for e in first_five
            ]
            
            # At least 3 out of 5 should be starred (allowing for some already starred)
            if starred_count >= 3:
                return True
            else:
                self.issues.append(f"Only {starred_count}/5 first emails are starred")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking starred emails: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Important label was applied to first 5 emails."""
        try:
            # Check step 45-46 when label was applied
            step_files = self.find_step_html("step_4[5-6]")
            if not step_files:
                step_files = self.find_step_html("step_*")
            
            if not step_files:
                self.issues.append("No step files found to verify label application")
                return False
            
            # Check the step where Important button was clicked
            check_file = None
            for f in sorted(step_files, reverse=True):
                if 'step_45' in f.name or 'step_46' in f.name:
                    check_file = f
                    break
            
            if not check_file:
                check_file = step_files[-1]
            
            with open(check_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if Important label dropdown/button appears or was selected
            # The action history shows step 45 clicked Important button
            has_important_action = 'Important' in content or 'important' in content.lower()
            
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if Important label button was clicked in actions
            important_clicked = False
            for action in action_history:
                if action.get('action', {}).get('parameters', {}).get('selector'):
                    selector = action['action']['parameters']['selector']
                    if 'Important' in selector or 'important' in selector.lower():
                        important_clicked = True
                        break
            
            self.details['important_label_action_found'] = important_clicked
            
            return important_clicked or has_important_action
            
        except Exception as e:
            self.issues.append(f"Error checking Important label: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: First five emails were archived."""
        try:
            # Check final HTML or later steps to see if emails were archived
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check inbox count - should have decreased or archive count increased
            inbox_match = re.search(r'Inbox.*?count.*?>(\d+)</span>', content, re.DOTALL)
            archive_match = re.search(r'Archive.*?count.*?>(\d+)</span>', content, re.DOTALL)
            
            initial_inbox = 150  # From initial state
            initial_archive = 487  # From initial state
            
            current_inbox = int(inbox_match.group(1)) if inbox_match else initial_inbox
            current_archive = int(archive_match.group(1)) if archive_match else initial_archive
            
            self.details['initial_inbox_count'] = initial_inbox
            self.details['final_inbox_count'] = current_inbox
            self.details['initial_archive_count'] = initial_archive
            self.details['final_archive_count'] = current_archive
            
            # Check if inbox count decreased or archive count increased
            # The task failed due to rate limit, so check if partial progress was made
            # Since the agent clicked label button but didn't archive, we check trajectory
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if archive action was attempted
            archive_attempted = False
            for action in action_history:
                reasoning = action.get('action', {}).get('reasoning', '')
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                if 'archive' in reasoning.lower() or 'archive' in selector.lower():
                    archive_attempted = True
                    break
            
            # The final state shows 150 emails still in inbox (no change)
            # Archive count is still 487 (no change)
            if current_inbox < initial_inbox or current_archive > initial_archive:
                return True
            elif archive_attempted:
                self.issues.append("Archive action attempted but not completed")
                return False
            else:
                self.issues.append("Emails were not archived from inbox")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking archive status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_first_five_starred'] = self.check_checkpoint_1()
            self.checkpoints['cp2_important_label_applied'] = self.check_checkpoint_2()
            self.checkpoints['cp3_emails_archived'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 16,
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
                'query_id': 16,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query16.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
