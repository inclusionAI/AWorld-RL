#!/usr/bin/env python3
# Evaluator for email Query 11

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
        self.in_trash_link = False
        self.in_trash_count = False
        self.in_page_info = False
        self.in_empty_state = False
        self.trash_count = None
        self.page_info_text = None
        self.empty_state_text = None
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in a nav-link for Trash
        if tag == 'a' and 'class' in attrs_dict:
            if 'nav-link' in attrs_dict['class']:
                self.in_trash_link = True
                
        # Check for count span within trash link
        if tag == 'span' and 'class' in attrs_dict and self.in_trash_link:
            if 'count' in attrs_dict['class']:
                self.in_trash_count = True
                
        # Check for page info span
        if tag == 'span' and 'class' in attrs_dict:
            if 'page-info' in attrs_dict['class']:
                self.in_page_info = True
                
        # Check for empty state div
        if tag == 'div' and 'class' in attrs_dict:
            if 'empty-state' in attrs_dict['class']:
                self.in_empty_state = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture trash count
        if self.in_trash_count:
            try:
                self.trash_count = int(data)
            except ValueError:
                pass
            self.current_data.append(data)
            
        # Capture page info
        if self.in_page_info:
            self.page_info_text = data
            
        # Capture empty state text
        if self.in_empty_state:
            if self.empty_state_text is None:
                self.empty_state_text = data
            else:
                self.empty_state_text += " " + data

    def handle_endtag(self, tag):
        if tag == 'a':
            # Check if this was the trash link
            if self.in_trash_link and 'Trash' in ' '.join(self.current_data):
                self.in_trash_link = False
                self.current_data = []
        if tag == 'span':
            self.in_trash_count = False
            self.in_page_info = False
        if tag == 'div':
            self.in_empty_state = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_trash": False,
            "cp2_emails_existed_in_trash": False,
            "cp3_selected_all_emails": False,
            "cp4_clicked_restore_button": False,
            "cp5_trash_is_empty": False,
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
        """Parse HTML file and return parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Trash folder"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent navigated to trash page
            for entry in trajectory:
                if 'page_info' in entry and 'url' in entry['page_info']:
                    if '/trash' in entry['page_info']['url']:
                        self.details['navigated_to_trash'] = True
                        return True
                        
            self.issues.append("Agent did not navigate to Trash folder")
            return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Emails existed in Trash before restoration"""
        try:
            # Check step 1 (after navigating to trash) for existing emails
            step_files = self.find_step_html("step_1_*")
            if not step_files:
                self.issues.append("No step 1 HTML found to verify emails in trash")
                return False
                
            parser = self.parse_html(step_files[0])
            
            # Check page info shows items count
            if parser.page_info_text and '3 items' in parser.page_info_text:
                self.details['initial_trash_count'] = 3
                return True
                
            # Also check trash count in sidebar
            if parser.trash_count == 3:
                self.details['initial_trash_count'] = 3
                return True
                
            self.issues.append(f"Expected 3 emails in trash, found: {parser.page_info_text}")
            return False
        except Exception as e:
            self.issues.append(f"Error checking initial trash state: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Selected all emails in trash"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for checkbox selection action
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '')
                        if 'checkbox' in selector.lower():
                            self.details['selected_emails'] = True
                            return True
                            
            self.issues.append("Agent did not select emails using checkbox")
            return False
        except Exception as e:
            self.issues.append(f"Error checking email selection: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Clicked Restore button"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for restore button click
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '')
                        if 'restore' in selector.lower():
                            self.details['clicked_restore'] = True
                            return True
                            
            self.issues.append("Agent did not click Restore button")
            return False
        except Exception as e:
            self.issues.append(f"Error checking restore button click: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Trash is empty after restoration"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
                
            parser = self.parse_html(final_html)
            
            # Check for "0 items" in page info
            if parser.page_info_text and '0 items' in parser.page_info_text:
                self.details['final_trash_count'] = 0
                self.details['trash_empty'] = True
                
                # Also verify empty state message
                if parser.empty_state_text and 'empty' in parser.empty_state_text.lower():
                    return True
                    
            # Also check trash count in sidebar should be updated
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for trash count showing 0 or 3 (it might still show 3 in sidebar)
                if '0 items' in content and 'Trash is empty' in content:
                    self.details['final_trash_count'] = 0
                    self.details['trash_empty'] = True
                    return True
                    
            self.issues.append(f"Trash not empty after restoration. Page info: {parser.page_info_text}")
            return False
        except Exception as e:
            self.issues.append(f"Error checking final trash state: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_trash'] = self.check_checkpoint_1()
            self.checkpoints['cp2_emails_existed_in_trash'] = self.check_checkpoint_2()
            self.checkpoints['cp3_selected_all_emails'] = self.check_checkpoint_3()
            self.checkpoints['cp4_clicked_restore_button'] = self.check_checkpoint_4()
            self.checkpoints['cp5_trash_is_empty'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 11,
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
                'query_id': 11,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
