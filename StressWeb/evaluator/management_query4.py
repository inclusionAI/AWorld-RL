#!/usr/bin/env python3
# Evaluator for management Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskDetailHTMLParser(HTMLParser):
    """Parse HTML to extract task detail information."""

    def __init__(self):
        super().__init__()
        self.in_title = False
        self.in_description = False
        self.in_edit_button = False
        self.in_save_button = False
        self.in_textarea = False
        self.in_meta_value = False
        self.current_meta_label = None
        
        self.task_title = None
        self.task_description = None
        self.has_edit_button = False
        self.has_save_button = False
        self.textarea_content = None
        self.is_edit_mode = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get('class', '')
        
        # Check for edit mode indicators
        if 'taskdetail-edit' in class_name or 'edit-form' in class_name:
            self.is_edit_mode = True
        
        # Task title
        if tag == 'h1' and 'taskdetail-title' in class_name:
            self.in_title = True
        elif tag == 'input' and 'title-input' in class_name:
            self.in_title = True
        
        # Task description
        if tag == 'p' and 'taskdetail-description' in class_name:
            self.in_description = True
        elif tag == 'textarea':
            self.in_textarea = True
            self.textarea_content = attrs_dict.get('value', '')
        
        # Edit button
        if tag == 'button' and 'edit-button' in class_name:
            self.has_edit_button = True
        
        # Save button
        if tag == 'button' and ('save-button' in class_name or 'submit-button' in class_name):
            self.has_save_button = True
        
        # Meta labels
        if tag == 'span' and 'meta-label' in class_name:
            self.current_meta_label = True

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        if self.in_title:
            self.task_title = text
        elif self.in_description:
            self.task_description = text
        elif self.in_textarea:
            if self.textarea_content:
                self.textarea_content += text
            else:
                self.textarea_content = text

    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_title = False
        elif tag == 'p':
            self.in_description = False
        elif tag == 'textarea':
            self.in_textarea = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on the task requirements
        self.checkpoints = {
            "cp1_found_task": False,
            "cp2_opened_detail_page": False,
            "cp3_clicked_edit_button": False,
            "cp4_changed_description": False,
            "cp5_saved_changes": False,
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

    def parse_html_file(self, html_path: Path) -> TaskDetailHTMLParser:
        """Parse an HTML file and return the parser with extracted data."""
        parser = TaskDetailHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Found the task 'Schedule team standup'"""
        try:
            # Check step 1 or 2 - should show the task in the list or detail page
            step_files = self.find_step_html("step_[12]*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Schedule team standup' in content:
                        self.details['found_task_at_step'] = step_file.name
                        return True
            
            self.issues.append("Could not find task 'Schedule team standup' in task list")
            return False
        except Exception as e:
            self.issues.append(f"Error checking cp1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Opened the task detail page"""
        try:
            # Check step 2-3 for task detail page
            step_files = self.find_step_html("step_[23]*")
            
            for step_file in step_files:
                parser = self.parse_html_file(step_file)
                if parser.task_title == 'Schedule team standup':
                    self.details['opened_detail_page_at'] = step_file.name
                    return True
            
            self.issues.append("Did not open the task detail page for 'Schedule team standup'")
            return False
        except Exception as e:
            self.issues.append(f"Error checking cp2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Clicked Edit button and entered edit mode"""
        try:
            # Check steps 3-5 for edit mode
            step_files = self.find_step_html("step_[3-9]*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for edit mode indicators
                    if ('textarea' in content and 'Schedule team standup' in content):
                        self.details['edit_mode_at'] = step_file.name
                        return True
            
            self.issues.append("Did not enter edit mode (no textarea found after clicking Edit)")
            return False
        except Exception as e:
            self.issues.append(f"Error checking cp3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Changed description to 'Daily standup at 9:30 AM'"""
        try:
            # Check all step files for the new description
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Daily standup at 9:30 AM' in content:
                        self.details['description_changed_at'] = step_file.name
                        return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Daily standup at 9:30 AM' in content:
                        self.details['description_changed_at'] = final_html.name
                        return True
            
            self.issues.append("Description was not changed to 'Daily standup at 9:30 AM'")
            return False
        except Exception as e:
            self.issues.append(f"Error checking cp4: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Saved the changes"""
        try:
            # Check if the description persists after leaving edit mode
            # Look for the new description in task detail view (not edit mode)
            step_files = self.find_step_html("step_*")
            
            found_in_edit = False
            found_saved = False
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Daily standup at 9:30 AM' in content:
                        # Check if in edit mode (has textarea) or detail mode
                        if 'textarea' in content or 'edit-form' in content:
                            found_in_edit = True
                        else:
                            # Found in detail view, meaning it was saved
                            found_saved = True
                            self.details['changes_saved_at'] = step_file.name
                            break
            
            # Also check final HTML
            if not found_saved:
                final_html = self.find_final_html()
                if final_html:
                    with open(final_html, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'Daily standup at 9:30 AM' in content:
                            # Check if in detail view (not edit mode)
                            if 'textarea' not in content or 'taskdetail-description' in content:
                                found_saved = True
                                self.details['changes_saved_at'] = final_html.name
            
            if found_saved:
                return True
            elif found_in_edit:
                self.issues.append("Description was typed but not saved (still in edit mode)")
                return False
            else:
                self.issues.append("Changes were not saved")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking cp5: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_found_task'] = self.check_checkpoint_1()
            self.checkpoints['cp2_opened_detail_page'] = self.check_checkpoint_2()
            self.checkpoints['cp3_clicked_edit_button'] = self.check_checkpoint_3()
            self.checkpoints['cp4_changed_description'] = self.check_checkpoint_4()
            self.checkpoints['cp5_saved_changes'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 4,
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
                'query_id': 4,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query4.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
