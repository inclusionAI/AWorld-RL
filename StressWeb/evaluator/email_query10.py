#!/usr/bin/env python3
# Evaluator for email Query 10

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TemplateHTMLParser(HTMLParser):
    """Parse HTML to extract template and sent email information."""

    def __init__(self):
        super().__init__()
        self.in_templates = False
        self.in_sent = False
        self.templates = []
        self.sent_emails = []
        self.current_template = {}
        self.current_email = {}
        self.current_tag = None
        self.current_class = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_class = attrs_dict.get('class', '').split()

        # Check if we're in templates section
        if 'templates' in attrs_dict.get('class', '').lower():
            self.in_templates = True
        
        # Check if we're in sent section
        if 'sent' in attrs_dict.get('class', '').lower():
            self.in_sent = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return

        # Capture template names
        if self.in_templates and 'template' in ' '.join(self.current_class).lower():
            if 'feedback request' in data.lower():
                self.templates.append(data)

        # Capture sent email info
        if self.in_sent:
            if 'students@mail.com' in data.lower():
                self.sent_emails.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_templates": False,
            "cp2_found_feedback_request_template": False,
            "cp3_filled_recipient_students_mail": False,
            "cp4_email_sent_successfully": False,
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
        """Checkpoint 1: Agent navigated to Templates section"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked on Templates link
            for action in action_history:
                action_obj = action.get('action', {})
                if action_obj.get('action_type') == 'CLICK':
                    selector = action_obj.get('parameters', {}).get('selector', '')
                    if 'template' in selector.lower():
                        self.details['templates_navigation'] = 'Found Templates navigation'
                        return True
            
            self.issues.append("Agent did not navigate to Templates section")
            return False
        except Exception as e:
            self.issues.append(f"Error checking templates navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Found and interacted with Feedback Request template"""
        try:
            # Check trajectory for interaction with Feedback Request template
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            for action in action_history:
                action_obj = action.get('action', {})
                reasoning = action_obj.get('reasoning', '')
                selector = action_obj.get('parameters', {}).get('selector', '')
                
                if 'feedback request' in reasoning.lower() or 'feedback request' in selector.lower():
                    self.details['template_interaction'] = 'Found interaction with Feedback Request template'
                    return True
            
            # Also check intermediate steps for template selection
            all_step_files = self.find_step_html("step_*")
            for step_file in all_step_files:
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        if 'feedback request' in html_content.lower() and 'use' in html_content.lower():
                            # Check if template was clicked or used
                            return True
                except:
                    continue
            
            self.issues.append("Agent did not find or interact with 'Feedback Request' template")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Feedback Request template: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Filled recipient field with students@mail.com"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check for fill action with students@mail.com
            for action in action_history:
                action_obj = action.get('action', {})
                if action_obj.get('action_type') == 'FILL':
                    value = action_obj.get('parameters', {}).get('value', '')
                    if 'students@mail.com' in value.lower():
                        self.details['recipient_filled'] = 'students@mail.com entered'
                        return True
            
            # Also check intermediate steps for the recipient field
            all_step_files = self.find_step_html("step_*")
            for step_file in all_step_files[5:]:  # Check steps after template selection
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        if 'students@mail.com' in html_content.lower():
                            self.details['recipient_filled'] = 'students@mail.com found in form'
                            return True
                except:
                    continue
            
            self.issues.append("Recipient 'students@mail.com' not filled")
            return False
        except Exception as e:
            self.issues.append(f"Error checking recipient field: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Email sent successfully to students@mail.com"""
        try:
            # Check final HTML or intermediate steps for sent email
            final_html = self.find_final_html()
            
            # Check final state
            if final_html and final_html.exists():
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                    # Look for sent confirmation or sent folder with students@mail.com
                    if 'students@mail.com' in html_content.lower():
                        # Check if it's in sent context
                        if 'sent' in html_content.lower() or 'success' in html_content.lower():
                            self.details['email_sent'] = 'Email to students@mail.com found in sent items'
                            return True
            
            # Check for send action in trajectory
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for send button click after filling recipient
            found_recipient = False
            for action in action_history:
                action_obj = action.get('action', {})
                
                # Track if recipient was filled
                if action_obj.get('action_type') == 'FILL':
                    value = action_obj.get('parameters', {}).get('value', '')
                    if 'students@mail.com' in value.lower():
                        found_recipient = True
                
                # Check for send action after recipient
                if found_recipient and action_obj.get('action_type') == 'CLICK':
                    selector = action_obj.get('parameters', {}).get('selector', '')
                    reasoning = action_obj.get('reasoning', '')
                    if 'send' in selector.lower() or 'send' in reasoning.lower():
                        self.details['email_sent'] = 'Send button clicked after filling recipient'
                        return True
            
            # Check intermediate steps for success notification
            all_step_files = self.find_step_html("step_*")
            for step_file in all_step_files[10:]:  # Check later steps
                try:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        if 'students@mail.com' in html_content.lower() and ('sent' in html_content.lower() or 'success' in html_content.lower()):
                            self.details['email_sent'] = 'Success notification found'
                            return True
                except:
                    continue
            
            self.issues.append("Email not sent successfully to students@mail.com")
            return False
        except Exception as e:
            self.issues.append(f"Error checking email sent status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_templates'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_feedback_request_template'] = self.check_checkpoint_2()
            self.checkpoints['cp3_filled_recipient_students_mail'] = self.check_checkpoint_3()
            self.checkpoints['cp4_email_sent_successfully'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
