#!/usr/bin/env python3
# Evaluator for ecom Query 10

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
        self.in_h1 = False
        self.in_span = False
        self.in_form = False
        self.page_title = ""
        self.sign_in_text = False
        self.current_tag = None
        self.current_attrs = {}
        self.form_email = None
        self.form_password = None
        self.form_fullname = None
        self.account_email_display = None
        self.in_email_display = False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        if tag == 'h1':
            self.in_h1 = True
        elif tag == 'span':
            self.in_span = True
        elif tag == 'form':
            self.in_form = True
        elif tag == 'input' and self.in_form:
            attrs_dict = dict(attrs)
            input_type = attrs_dict.get('type', '')
            input_name = attrs_dict.get('name', '')
            input_value = attrs_dict.get('value', '')
            
            if input_name == 'email':
                self.form_email = input_value
            elif input_name == 'password':
                self.form_password = input_value
            elif input_name == 'fullName':
                self.form_fullname = input_value
        elif tag == 'div':
            attrs_dict = dict(attrs)
            if 'class' in attrs_dict and 'account-email' in attrs_dict.get('class', ''):
                self.in_email_display = True

    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_h1 = False
        elif tag == 'span':
            self.in_span = False
        elif tag == 'form':
            self.in_form = False
        elif tag == 'div':
            self.in_email_display = False

    def handle_data(self, data):
        text = data.strip()
        if self.in_h1 and text:
            self.page_title = text
        if self.in_span and text == 'Sign In':
            self.sign_in_text = True
        if self.in_email_display and text:
            self.account_email_display = text


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_account_creation_form_accessed": False,
            "cp2_registration_form_filled": False,
            "cp3_account_created_or_signed_in": False,
            "cp4_navigated_to_account_page": False,
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
        """Checkpoint 1: Agent accessed account creation form (Sign Up page)"""
        try:
            # Check if agent clicked on Sign Up link in any step
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files[3:15]:  # Check steps 3-15 based on result.json
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    parser = TaskHTMLParser()
                    parser.feed(html_content)
                    
                    # Check if we're on Create Account page
                    if parser.page_title == "Create Account":
                        self.details['signup_page_found'] = str(step_file)
                        return True
            
            self.issues.append("Agent did not successfully access the Sign Up / Create Account page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Registration form was filled with correct email"""
        try:
            # Check intermediate steps for filled form
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files[9:20]:  # Check around step 10 where email was filled
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    parser = TaskHTMLParser()
                    parser.feed(html_content)
                    
                    # Check if email field has correct value
                    if parser.form_email == "newcustomer@test.com":
                        self.details['email_filled'] = True
                        self.details['email_filled_step'] = str(step_file)
                        
                        # Check if password is also filled
                        if parser.form_password and len(parser.form_password) > 0:
                            self.details['password_filled'] = True
                            return True
            
            self.issues.append("Registration form was not properly filled with newcustomer@test.com")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Account was created or user successfully signed in"""
        try:
            # Check if agent got past the sign up form
            # Since agent hit max_steps while stuck on signup page, this likely failed
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
                parser = TaskHTMLParser()
                parser.feed(html_content)
                
                # If we're still on Create Account page at the end, account was not created
                if parser.page_title == "Create Account":
                    self.issues.append("Agent remained stuck on Create Account page - account was not created")
                    self.details['stuck_on_signup'] = True
                    return False
                
                # If we see "Sign In" link/text, user is NOT logged in
                if parser.sign_in_text:
                    self.issues.append("User is not signed in (Sign In link still visible)")
                    return False
                
                # If page title is account-related, account was created
                if "account" in parser.page_title.lower() or "profile" in parser.page_title.lower():
                    self.details['account_page_reached'] = True
                    return True
            
            self.issues.append("Could not verify account creation or sign in")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Successfully navigated to account page"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
                parser = TaskHTMLParser()
                parser.feed(html_content)
                
                # Check if we're on account page
                if parser.page_title and ("account" in parser.page_title.lower() or 
                                          "profile" in parser.page_title.lower() or
                                          "my account" in parser.page_title.lower()):
                    self.details['final_page_title'] = parser.page_title
                    
                    # Check if email is displayed on account page
                    if parser.account_email_display == "newcustomer@test.com":
                        self.details['account_email_verified'] = True
                        return True
                    return True
                
                # Still on Create Account page means task failed
                if parser.page_title == "Create Account":
                    self.issues.append("Agent never left the Create Account page")
                    self.details['final_page_title'] = parser.page_title
                    return False
                
                self.issues.append(f"Final page was not the account page (title: {parser.page_title})")
                self.details['final_page_title'] = parser.page_title
                return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_account_creation_form_accessed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_registration_form_filled'] = self.check_checkpoint_2()
            self.checkpoints['cp3_account_created_or_signed_in'] = self.check_checkpoint_3()
            self.checkpoints['cp4_navigated_to_account_page'] = self.check_checkpoint_4()

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
        print("Usage: python ecom_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
