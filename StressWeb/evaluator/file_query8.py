#!/usr/bin/env python3
# Evaluator for file Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ShareDialogParser(HTMLParser):
    """Parse HTML to extract share dialog information."""

    def __init__(self):
        super().__init__()
        self.in_modal = False
        self.in_email_input = False
        self.in_permission_option = False
        self.in_permission_content = False
        self.in_notification = False
        self.in_share_button = False
        
        self.current_permission = None
        self.email_value = None
        self.selected_permission = None
        self.share_button_disabled = False
        self.notification_message = None
        self.notification_type = None
        self.current_permission_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for modal overlay
        if tag == 'div' and 'modal-overlay' in attrs_dict.get('class', ''):
            self.in_modal = True
        
        # Check for email input
        if tag == 'input' and 'user-search-input' in attrs_dict.get('class', ''):
            self.in_email_input = True
            self.email_value = attrs_dict.get('value', '')
        
        # Check for permission options
        if tag == 'div' and 'permission-option' in attrs_dict.get('class', ''):
            self.in_permission_option = True
            classes = attrs_dict.get('class', '')
            if 'selected' in classes:
                self.current_permission = 'selected'
            else:
                self.current_permission = 'not_selected'
        
        # Check for permission content
        if self.in_permission_option and tag == 'strong':
            self.in_permission_content = True
            self.current_permission_text = []
        
        # Check for share button
        if tag == 'button' and 'btn-primary' in attrs_dict.get('class', ''):
            self.in_share_button = True
            if 'disabled' in attrs_dict or attrs_dict.get('disabled') == '':
                self.share_button_disabled = True
        
        # Check for notification
        if tag == 'div' and 'notification' in attrs_dict.get('class', ''):
            self.in_notification = True
            classes = attrs_dict.get('class', '')
            if 'success' in classes:
                self.notification_type = 'success'
            elif 'error' in classes:
                self.notification_type = 'error'

    def handle_data(self, data):
        if self.in_permission_content:
            self.current_permission_text.append(data.strip())
        
        if self.in_notification and data.strip():
            self.notification_message = data.strip()

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_modal:
            self.in_modal = False
        
        if tag == 'input' and self.in_email_input:
            self.in_email_input = False
        
        if tag == 'div' and self.in_permission_option:
            if self.current_permission == 'selected' and self.current_permission_text:
                self.selected_permission = ''.join(self.current_permission_text)
            self.in_permission_option = False
            self.current_permission = None
            self.current_permission_text = []
        
        if tag == 'strong' and self.in_permission_content:
            self.in_permission_content = False
        
        if tag == 'button' and self.in_share_button:
            self.in_share_button = False
        
        if tag == 'div' and self.in_notification:
            self.in_notification = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for sharing the Finance folder
        self.checkpoints = {
            "cp1_navigate_to_finance": False,
            "cp2_open_share_dialog": False,
            "cp3_enter_email": False,
            "cp4_select_view_copy_permission": False,
            "cp5_share_completed": False,
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

    def parse_html_for_share_state(self, html_path: Path) -> Dict:
        """Parse HTML file to extract share dialog state."""
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = ShareDialogParser()
        parser.feed(content)
        
        return {
            'email': parser.email_value,
            'selected_permission': parser.selected_permission,
            'share_button_disabled': parser.share_button_disabled,
            'notification_message': parser.notification_message,
            'notification_type': parser.notification_type,
        }

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigate to Finance folder"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent navigated to Finance folder
            for action in action_history:
                action_detail = action.get('action', {})
                selector = action_detail.get('parameters', {}).get('selector', '')
                
                # Check for clicking on Finance folder
                if 'Finance' in selector or 'finance' in selector.lower():
                    self.details['finance_clicked'] = True
                    return True
            
            # Also check final HTML for Finance breadcrumb
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Finance</a></nav>' in content or 'Finance</h1>' in content:
                        self.details['finance_page_reached'] = True
                        return True
            
            self.issues.append("Did not navigate to Finance folder")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Open share dialog"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked Share button
            for action in action_history:
                action_detail = action.get('action', {})
                selector = action_detail.get('parameters', {}).get('selector', '')
                
                if 'Share' in selector or 'share' in selector.lower():
                    self.details['share_button_clicked'] = True
            
            # Check any HTML file for share modal presence
            all_html = self.find_step_html("step_*")
            for html_file in all_html[-10:]:  # Check last 10 steps
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Share "Finance"' in content or 'share-modal' in content:
                        self.details['share_dialog_opened'] = True
                        return True
            
            self.issues.append("Did not open share dialog")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking share dialog: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Enter email address"""
        try:
            # Check intermediate steps for email entry
            all_html = self.find_step_html("step_*")
            
            for html_file in all_html:
                state = self.parse_html_for_share_state(html_file)
                if state['email'] == 'sara.johnson@example.com':
                    self.details['email_entered'] = True
                    self.details['email_step'] = html_file.name
                    return True
            
            self.issues.append("Email address 'sara.johnson@example.com' was not entered")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking email: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Select View & Copy permission"""
        try:
            # Check intermediate steps for permission selection
            all_html = self.find_step_html("step_*")
            
            for html_file in all_html:
                state = self.parse_html_for_share_state(html_file)
                if state['selected_permission'] == 'View & Copy':
                    self.details['permission_selected'] = 'View & Copy'
                    self.details['permission_step'] = html_file.name
                    return True
            
            self.issues.append("'View & Copy' permission was not selected")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking permission: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Share completed successfully"""
        try:
            # Check for success notification in any step
            all_html = self.find_step_html("step_*")
            
            # Look for share success indicators
            for html_file in all_html:
                state = self.parse_html_for_share_state(html_file)
                
                # Check for success notification
                if state['notification_type'] == 'success':
                    if state['notification_message']:
                        if 'share' in state['notification_message'].lower():
                            self.details['share_success_notification'] = state['notification_message']
                            self.details['success_step'] = html_file.name
                            return True
            
            # Check if form was properly filled and button was enabled
            properly_filled_steps = []
            for html_file in all_html:
                state = self.parse_html_for_share_state(html_file)
                if (state['email'] == 'sara.johnson@example.com' and 
                    state['selected_permission'] == 'View & Copy' and 
                    not state['share_button_disabled']):
                    properly_filled_steps.append(html_file.name)
            
            if properly_filled_steps:
                self.details['form_properly_filled_steps'] = properly_filled_steps
                # Check if Share with User button was clicked after form was properly filled
                result_data = self.load_result_json()
                action_history = result_data.get('action_history', [])
                
                # Find step numbers where form was properly filled
                properly_filled_step_nums = [int(re.search(r'step_(\d+)', s).group(1)) for s in properly_filled_steps]
                
                # Check if "Share with User" was clicked after any of these steps
                for action in action_history:
                    step_num = action.get('step')
                    action_detail = action.get('action', {})
                    selector = action_detail.get('parameters', {}).get('selector', '')
                    
                    if step_num in properly_filled_step_nums:
                        # Check next action
                        next_action_idx = action_history.index(action) + 1
                        if next_action_idx < len(action_history):
                            next_action = action_history[next_action_idx]
                            next_selector = next_action.get('action', {}).get('parameters', {}).get('selector', '')
                            if 'Share with User' in next_selector:
                                self.details['share_button_clicked_when_ready'] = True
                                # Without backend verification, we consider this successful
                                return True
            
            self.issues.append("Share was not completed successfully - no success notification found")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking share completion: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigate_to_finance'] = self.check_checkpoint_1()
            self.checkpoints['cp2_open_share_dialog'] = self.check_checkpoint_2()
            self.checkpoints['cp3_enter_email'] = self.check_checkpoint_3()
            self.checkpoints['cp4_select_view_copy_permission'] = self.check_checkpoint_4()
            self.checkpoints['cp5_share_completed'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 8,
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
                'query_id': 8,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
