#!/usr/bin/env python3
# Evaluator for ecom Query 1

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
        self.in_user_menu = False
        self.in_account_page = False
        self.in_profile_section = False
        self.in_info_value = False
        self.current_label = None
        
        self.user_menu_text = ""
        self.email_found = None
        self.on_account_page = False
        self.account_title = None
        self.profile_section_found = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for user menu (signed in indicator)
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'user-menu' in classes:
                self.in_user_menu = True
            if 'account-page' in classes:
                self.in_account_page = True
                self.on_account_page = True
            if 'profile-section' in classes:
                self.in_profile_section = True
                self.profile_section_found = True
        
        # Check for account title
        if tag == 'h1' and 'class' in attrs_dict and 'account-title' in attrs_dict['class']:
            self.in_account_page = True
        
        # Check for info label
        if tag == 'span' and 'class' in attrs_dict:
            if 'info-label' in attrs_dict['class']:
                self.current_label = 'label'
            elif 'info-value' in attrs_dict['class']:
                self.in_info_value = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_user_menu:
            self.user_menu_text += data + " "
        
        if self.in_account_page and self.account_title is None:
            if 'My Account' in data or 'Account' in data:
                self.account_title = data
        
        if self.in_info_value:
            # Check if this is an email
            if '@' in data and 'mail.com' in data:
                self.email_found = data
        
        if self.current_label == 'label':
            if 'Email' in data:
                self.current_label = 'email'
            else:
                self.current_label = None

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_user_menu:
                self.in_user_menu = False
            if self.in_profile_section:
                self.in_profile_section = False
        if tag == 'span':
            self.in_info_value = False
            if self.current_label == 'label':
                self.current_label = None


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_signed_in": False,
            "cp2_on_account_page": False,
            "cp3_profile_visible": False,
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
        """Checkpoint 1: User successfully signed in"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = TaskHTMLParser()
        parser.feed(html_content)

        # Check if user is signed in by looking for "Demo User" or user menu
        if 'Demo User' in html_content or 'user-menu' in html_content:
            # Verify it's not showing "Sign In" which would indicate not signed in
            if 'Sign In' in parser.user_menu_text and 'Sign Out' not in html_content:
                self.issues.append("User appears to still be on sign-in page, not signed in")
                return False
            
            self.details['signed_in'] = True
            self.details['user_menu_text'] = parser.user_menu_text.strip()
            return True
        
        self.issues.append("No sign-in indicator found in final page")
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: User navigated to account page"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = TaskHTMLParser()
        parser.feed(html_content)

        # Check for account page indicators
        if parser.on_account_page or 'account-page' in html_content or 'My Account' in html_content:
            self.details['on_account_page'] = True
            self.details['account_title'] = parser.account_title
            return True
        
        # Check URL from trajectory
        trajectory = self.get_trajectory()
        if trajectory:
            last_url = None
            for entry in reversed(trajectory):
                if 'url' in entry:
                    last_url = entry['url']
                    break
            
            if last_url and '/account' in last_url:
                self.details['on_account_page'] = True
                self.details['account_url'] = last_url
                return True
        
        self.issues.append("User did not reach account page")
        return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Profile information visible with correct email"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = TaskHTMLParser()
        parser.feed(html_content)

        # Check if profile section is visible
        if not parser.profile_section_found and 'profile-section' not in html_content:
            self.issues.append("Profile section not found on account page")
            return False

        # Check if the correct email is displayed (either newuser@test.com or 12345@mail.com)
        if parser.email_found:
            if parser.email_found in ['newuser@test.com', '12345@mail.com']:
                self.details['email_displayed'] = parser.email_found
                return True
            else:
                self.issues.append(f"Unexpected email found: {parser.email_found}")
                return False
        
        # Fallback: check raw HTML content
        if '12345@mail.com' in html_content or 'newuser@test.com' in html_content:
            if '12345@mail.com' in html_content:
                self.details['email_displayed'] = '12345@mail.com'
            else:
                self.details['email_displayed'] = 'newuser@test.com'
            return True
        
        self.issues.append("Email not found in profile information")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_signed_in'] = self.check_checkpoint_1()
            self.checkpoints['cp2_on_account_page'] = self.check_checkpoint_2()
            self.checkpoints['cp3_profile_visible'] = self.check_checkpoint_3()

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
        print("Usage: python ecom_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
