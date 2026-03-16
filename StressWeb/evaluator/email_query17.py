#!/usr/bin/env python3
# Evaluator for email Query 17

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EmailHTMLParser(HTMLParser):
    """Parse HTML to extract email information from the email client."""

    def __init__(self):
        super().__init__()
        self.in_email_item = False
        self.in_starred = False
        self.current_email = {}
        self.emails = []
        self.starred_emails = []
        self.labels = []
        self.current_label = None
        self.in_toolbar = False
        self.page_info = ""
        self.trash_empty = False
        self.in_empty_state = False
        self.capture_text = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for email items
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            
            # Check for starred icon
            if 'fa-star' in classes and 'fas' in classes:
                self.in_starred = True
            
            # Check for page info in toolbar
            if 'page-info' in classes:
                self.capture_text = True
                
            # Check for empty state (trash is empty)
            if 'empty-state' in classes:
                self.in_empty_state = True
                
        # Check for span with label class
        if tag == 'span' and 'class' in attrs_dict:
            if 'label' in attrs_dict['class']:
                self.capture_text = True

    def handle_data(self, data):
        data = data.strip()
        if self.capture_text and data:
            if self.in_empty_state and 'empty' in data.lower():
                self.trash_empty = True
            self.page_info = data
            self.capture_text = False
            
    def handle_endtag(self, tag):
        if tag == 'div' and self.in_empty_state:
            self.in_empty_state = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_search_and_star_hr": False,
            "cp2_delete_last_7_days": False,
            "cp3_apply_work_label": False,
            "cp4_star_five_emails": False,
            "cp5_add_important_label": False,
            "cp6_archive_emails": False,
            "cp7_restore_from_trash": False,
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

    def check_checkpoint_1(self, trajectory: List[Dict]) -> bool:
        """Checkpoint 1: Search for emails from hr@company.com and star the first result"""
        try:
            # Look for search action
            search_found = False
            star_action_found = False
            
            for action in trajectory:
                if 'action' in action:
                    act = action['action']
                    
                    # Check for search
                    if act.get('action_type') == 'TYPE':
                        text = act.get('parameters', {}).get('text', '')
                        if 'hr@company.com' in text:
                            search_found = True
                    
                    # Check for star action
                    if act.get('action_type') == 'CLICK':
                        selector = act.get('parameters', {}).get('selector', '')
                        reasoning = act.get('reasoning', '')
                        if 'star' in selector.lower() or 'star' in reasoning.lower():
                            if 'first' in reasoning.lower() and 'hr@company.com' in reasoning.lower():
                                star_action_found = True
            
            if search_found and star_action_found:
                self.details['search_and_star'] = 'Search executed and first result starred'
                return True
            else:
                missing = []
                if not search_found:
                    missing.append('search')
                if not star_action_found:
                    missing.append('star action')
                self.issues.append(f"Missing actions for hr@company.com: {', '.join(missing)}")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking search and star: {str(e)}")
            return False

    def check_checkpoint_2(self, trajectory: List[Dict]) -> bool:
        """Checkpoint 2: Delete all emails from the last 7 days"""
        try:
            # Look for delete action after filtering last 7 days
            delete_found = False
            
            for action in trajectory:
                if 'action' in action:
                    act = action['action']
                    reasoning = act.get('reasoning', '')
                    
                    # Check for delete action with 7 days context
                    if 'delete' in reasoning.lower() and '7 days' in reasoning.lower():
                        delete_found = True
                        break
            
            if delete_found:
                self.details['delete_7_days'] = 'Deleted emails from last 7 days'
                return True
            else:
                self.issues.append("No delete action found for emails from last 7 days")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking delete action: {str(e)}")
            return False

    def check_checkpoint_3(self, trajectory: List[Dict]) -> bool:
        """Checkpoint 3: Apply 'Work' label to 'Training session invitation' email"""
        try:
            # Look for work label application
            work_label_found = False
            
            for action in trajectory:
                if 'action' in action:
                    act = action['action']
                    reasoning = act.get('reasoning', '')
                    selector = act.get('parameters', {}).get('selector', '')
                    
                    # Check for Work label action on Training session invitation
                    if 'work' in reasoning.lower() or 'work' in selector.lower():
                        if 'training' in reasoning.lower():
                            work_label_found = True
                            break
            
            if work_label_found:
                self.details['work_label'] = 'Applied Work label to Training session invitation'
                return True
            else:
                self.issues.append("Work label not applied to Training session invitation")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Work label: {str(e)}")
            return False

    def check_checkpoint_4(self, trajectory: List[Dict]) -> bool:
        """Checkpoint 4: Star the first five emails in the inbox"""
        try:
            # Count star actions in inbox context
            star_count = 0
            
            for action in trajectory:
                if 'action' in action:
                    act = action['action']
                    reasoning = act.get('reasoning', '')
                    selector = act.get('parameters', {}).get('selector', '')
                    
                    # Check for starring actions
                    if act.get('action_type') == 'CLICK':
                        if 'star' in selector.lower() or 'star' in reasoning.lower():
                            if 'five' in reasoning.lower() or 'first 5' in reasoning.lower():
                                # Likely selected multiple and starred them together
                                star_count = 5
                                break
                            elif 'inbox' in reasoning.lower():
                                star_count += 1
            
            if star_count >= 5:
                self.details['star_five'] = f'Starred {star_count} emails'
                return True
            else:
                self.issues.append(f"Only {star_count} emails starred, expected 5")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking star action: {str(e)}")
            return False

    def check_checkpoint_5(self, trajectory: List[Dict]) -> bool:
        """Checkpoint 5: Add 'important' label to first five emails"""
        try:
            # Look for important label action
            important_label_found = False
            
            for action in trajectory:
                if 'action' in action:
                    act = action['action']
                    reasoning = act.get('reasoning', '')
                    selector = act.get('parameters', {}).get('selector', '')
                    
                    # Check for important label
                    if 'important' in reasoning.lower() or 'important' in selector.lower():
                        if 'five' in reasoning.lower() or 'label' in reasoning.lower():
                            important_label_found = True
                            break
            
            if important_label_found:
                self.details['important_label'] = 'Added Important label to emails'
                return True
            else:
                self.issues.append("Important label not added to first five emails")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Important label: {str(e)}")
            return False

    def check_checkpoint_6(self, trajectory: List[Dict]) -> bool:
        """Checkpoint 6: Archive all selected emails"""
        try:
            # Look for archive action
            archive_found = False
            
            for action in trajectory:
                if 'action' in action:
                    act = action['action']
                    reasoning = act.get('reasoning', '')
                    selector = act.get('parameters', {}).get('selector', '')
                    
                    # Check for archive action
                    if 'archive' in reasoning.lower() or 'archive' in selector.lower():
                        if 'five' in reasoning.lower() or 'all' in reasoning.lower():
                            archive_found = True
                            break
            
            if archive_found:
                self.details['archive'] = 'Archived selected emails'
                return True
            else:
                self.issues.append("Archive action not found")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking archive action: {str(e)}")
            return False

    def check_checkpoint_7(self) -> bool:
        """Checkpoint 7: Restore all emails from Trash (check final state)"""
        try:
            # Check final HTML to see if trash is empty
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if trash is empty (indicating restore was successful)
            if 'Trash is empty' in html_content or 'empty-state' in html_content:
                # Verify we're on the Trash page
                if 'trash' in html_content.lower() and 'active' in html_content:
                    self.details['restore_trash'] = 'All emails restored from Trash (Trash is empty)'
                    return True
            
            self.issues.append("Trash not empty - restore may not have completed")
            return False
                
        except Exception as e:
            self.issues.append(f"Error checking trash restore: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()
            trajectory = result_data.get('action_history', [])

            # Run checkpoint validations
            self.checkpoints['cp1_search_and_star_hr'] = self.check_checkpoint_1(trajectory)
            self.checkpoints['cp2_delete_last_7_days'] = self.check_checkpoint_2(trajectory)
            self.checkpoints['cp3_apply_work_label'] = self.check_checkpoint_3(trajectory)
            self.checkpoints['cp4_star_five_emails'] = self.check_checkpoint_4(trajectory)
            self.checkpoints['cp5_add_important_label'] = self.check_checkpoint_5(trajectory)
            self.checkpoints['cp6_archive_emails'] = self.check_checkpoint_6(trajectory)
            self.checkpoints['cp7_restore_from_trash'] = self.check_checkpoint_7()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 17,
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
                'query_id': 17,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query17.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
