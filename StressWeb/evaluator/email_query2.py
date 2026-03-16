#!/usr/bin/env python3
# Evaluator for email Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract email information and star status."""

    def __init__(self):
        super().__init__()
        self.in_message_item = False
        self.in_star_btn = False
        self.in_subject = False
        self.in_sender = False
        self.in_search_header = False
        self.in_search_results_h2 = False
        
        self.current_message = {}
        self.messages = []
        self.search_query = None
        self.star_btn_classes = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for search header
        if tag == 'div' and attrs_dict.get('class') == 'search-header':
            self.in_search_header = True
        
        # Check for h2 in search header
        if tag == 'h2' and self.in_search_header:
            self.in_search_results_h2 = True
        
        # Check for message items
        if tag == 'div' and 'message-item' in attrs_dict.get('class', ''):
            self.in_message_item = True
            self.current_message = {
                'sender': None,
                'subject': None,
                'starred': False
            }
        
        # Check for star button
        if tag == 'button' and self.in_message_item:
            classes = attrs_dict.get('class', '')
            if 'star-btn' in classes:
                self.in_star_btn = True
                # Check if starred
                if 'starred' in classes:
                    self.current_message['starred'] = True
        
        # Check for subject
        if tag == 'div' and self.in_message_item and attrs_dict.get('class') == 'subject':
            self.in_subject = True
        
        # Check for sender
        if tag == 'span' and self.in_message_item and attrs_dict.get('class') == 'sender':
            self.in_sender = True
    
    def handle_endtag(self, tag):
        if tag == 'div' and self.in_search_header:
            self.in_search_header = False
        
        if tag == 'h2' and self.in_search_results_h2:
            self.in_search_results_h2 = False
        
        if tag == 'button' and self.in_star_btn:
            self.in_star_btn = False
        
        if tag == 'div' and self.in_subject:
            self.in_subject = False
        
        if tag == 'span' and self.in_sender:
            self.in_sender = False
        
        if tag == 'div' and self.in_message_item:
            # End of message item
            if self.current_message.get('subject'):
                self.messages.append(self.current_message)
            self.in_message_item = False
            self.current_message = {}
    
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_search_results_h2:
            self.search_query = data
        
        if self.in_subject:
            self.current_message['subject'] = data
        
        if self.in_sender:
            self.current_message['sender'] = data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define 3 checkpoints based on task
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_search_results_displayed": False,
            "cp3_first_email_starred": False,
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

    def parse_html(self, html_path: Path) -> TaskHTMLParser:
        """Parse HTML file and extract email information."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Search was executed for emails from hr@company.com"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if search query was typed
            search_typed = False
            enter_pressed = False
            
            for step in trajectory:
                action = step.get('action', {})
                action_type = action.get('action_type', '')
                
                # Check if "from:hr@company.com" was typed
                if action_type == 'TYPE':
                    text = action.get('parameters', {}).get('text', '')
                    if 'from:hr@company.com' in text.lower():
                        search_typed = True
                
                # Check if Enter was pressed to execute search
                if action_type == 'PRESS':
                    key = action.get('parameters', {}).get('key', '')
                    if key == 'Enter':
                        enter_pressed = True
            
            if search_typed and enter_pressed:
                self.details['search_executed'] = True
                return True
            else:
                if not search_typed:
                    self.issues.append("Search query 'from:hr@company.com' was not typed")
                if not enter_pressed:
                    self.issues.append("Enter key was not pressed to execute search")
                self.details['search_executed'] = False
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Search results page is displayed with emails from hr@company.com"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html(final_html)
            
            # Check if search results are displayed
            if parser.search_query and 'from:hr@company.com' in parser.search_query:
                self.details['search_results_query'] = parser.search_query
                
                # Check if there are emails from hr@company.com
                hr_emails = [msg for msg in parser.messages if msg.get('sender') == 'hr@company.com']
                
                if hr_emails:
                    self.details['hr_emails_count'] = len(hr_emails)
                    self.details['total_emails_in_results'] = len(parser.messages)
                    return True
                else:
                    self.issues.append("No emails from hr@company.com found in search results")
                    return False
            else:
                self.issues.append("Search results page not displayed or incorrect query")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking search results: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: First email from hr@company.com is starred"""
        try:
            # Check both intermediate steps and final HTML
            # The star might appear in step 4 or 5 after the click
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html(final_html)
            
            # Find first email from hr@company.com
            hr_emails = [msg for msg in parser.messages if msg.get('sender') == 'hr@company.com']
            
            if not hr_emails:
                self.issues.append("No emails from hr@company.com found in final state")
                return False
            
            first_hr_email = hr_emails[0]
            self.details['first_hr_email_subject'] = first_hr_email.get('subject')
            self.details['first_hr_email_starred'] = first_hr_email.get('starred', False)
            
            if first_hr_email.get('starred', False):
                return True
            else:
                self.issues.append(f"First email from hr@company.com ('{first_hr_email.get('subject')}') is not starred")
                
                # Check if any intermediate step shows the star
                step_files = self.find_step_html("step_*")
                for step_file in step_files:
                    step_parser = self.parse_html(step_file)
                    step_hr_emails = [msg for msg in step_parser.messages if msg.get('sender') == 'hr@company.com']
                    if step_hr_emails and step_hr_emails[0].get('starred', False):
                        self.issues.append(f"Star was visible in {step_file.name} but not in final state")
                        break
                
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking star status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_search_results_displayed'] = self.check_checkpoint_2()
            self.checkpoints['cp3_first_email_starred'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 2,
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
                'query_id': 2,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
