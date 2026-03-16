#!/usr/bin/env python3
# Evaluator for management Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.in_subtitle = False
        self.in_filter_button = False
        self.in_no_tasks = False
        self.current_tag_attrs = []
        self.subtitle_text = ""
        self.filter_button_text = ""
        self.no_tasks_text = ""
        self.low_filter_active = False
        self.task_count = 0
        self.tasks_showing = None
        self.tasks_total = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag_attrs = attrs_dict
        
        # Check for subtitle paragraph
        if tag == 'p' and attrs_dict.get('class') == 'tasklist-subtitle':
            self.in_subtitle = True
            
        # Check for filter button
        if tag == 'button' and 'filter-button' in attrs_dict.get('class', ''):
            self.in_filter_button = True
            
        # Check for no tasks message
        if tag == 'p' and self.in_no_tasks:
            pass
        
        if tag == 'div' and attrs_dict.get('class') == 'no-tasks-message':
            self.in_no_tasks = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture subtitle text
        if self.in_subtitle:
            self.subtitle_text += data
            # Parse "Showing X of Y tasks"
            match = re.search(r'Showing (\d+) of (\d+) tasks', data)
            if match:
                self.tasks_showing = int(match.group(1))
                self.tasks_total = int(match.group(2))
        
        # Check for active Low priority filter
        if self.in_filter_button:
            self.filter_button_text = data
            if '🟢 Low' in data and 'active' in self.current_tag_attrs.get('class', ''):
                self.low_filter_active = True
                
        # Check for "No tasks found" message
        if self.in_no_tasks and data == 'No tasks found':
            self.no_tasks_text = data

    def handle_endtag(self, tag):
        if tag == 'p' and self.in_subtitle:
            self.in_subtitle = False
        if tag == 'button' and self.in_filter_button:
            self.in_filter_button = False
        if tag == 'div' and self.in_no_tasks:
            self.in_no_tasks = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_tasks": False,
            "cp2_filtered_by_low_priority": False,
            "cp3_selected_all_tasks": False,
            "cp4_bulk_deleted_tasks": False,
            "cp5_confirmed_deletion": False,
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
        """Checkpoint 1: Navigated to Tasks page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if there's a click action on Tasks link
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Tasks' in selector or 'tasks' in selector.lower():
                        self.details['navigated_to_tasks'] = True
                        return True
            
            self.issues.append("Did not navigate to Tasks page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Filtered tasks by Low priority"""
        try:
            # Check intermediate step where Low filter was applied
            step_2_files = self.find_step_html("step_2_*")
            if not step_2_files:
                self.issues.append("Step 2 HTML not found - cannot verify Low priority filter")
                return False
            
            with open(step_2_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            if parser.low_filter_active:
                self.details['low_filter_applied'] = True
                self.details['tasks_after_filter'] = parser.tasks_showing
                return True
            else:
                self.issues.append("Low priority filter was not applied")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Low priority filter: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Selected all displayed Low priority tasks"""
        try:
            # Check step 3 where all tasks were selected
            step_3_files = self.find_step_html("step_3_*")
            if not step_3_files:
                self.issues.append("Step 3 HTML not found - cannot verify task selection")
                return False
            
            with open(step_3_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for selected tasks - look for class="task-card selected"
            selected_count = html_content.count('class="task-card selected"')
            
            # Check for "7 tasks selected" text
            if '7 tasks selected' in html_content:
                self.details['tasks_selected'] = 7
                self.details['selection_verified'] = True
                return True
            elif selected_count >= 7:
                self.details['tasks_selected'] = selected_count
                return True
            else:
                self.issues.append(f"Not all Low priority tasks were selected (found {selected_count} selected)")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking task selection: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Initiated bulk delete action"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for clicks on Batch Actions and Delete Tasks
            batch_actions_clicked = False
            delete_tasks_clicked = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Batch Actions' in selector:
                        batch_actions_clicked = True
                    if 'Delete Tasks' in selector or 'Delete' in selector:
                        delete_tasks_clicked = True
            
            if batch_actions_clicked and delete_tasks_clicked:
                self.details['bulk_delete_initiated'] = True
                return True
            else:
                if not batch_actions_clicked:
                    self.issues.append("Batch Actions was not clicked")
                if not delete_tasks_clicked:
                    self.issues.append("Delete Tasks option was not selected")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking bulk delete: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Confirmed deletion and verified tasks were deleted"""
        try:
            # Check trajectory for confirmation click
            trajectory = self.get_trajectory()
            confirmed = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Delete 7 Tasks' in selector or ('Delete' in selector and 'Tasks' in selector):
                        confirmed = True
                        break
            
            if not confirmed:
                self.issues.append("Deletion was not confirmed")
                return False
            
            # Check final HTML to verify tasks were deleted
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML not found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            # Verify Low filter is still active
            if not parser.low_filter_active:
                self.issues.append("Low priority filter not active in final state")
                return False
            
            # Verify no tasks are showing (0 of X tasks)
            if parser.tasks_showing == 0:
                self.details['tasks_after_deletion'] = 0
                self.details['deletion_verified'] = True
                
                # Check for "No tasks found" message
                if parser.no_tasks_text == 'No tasks found':
                    self.details['no_tasks_message_shown'] = True
                    return True
                else:
                    # Still successful if showing 0 tasks
                    return True
            else:
                self.issues.append(f"Tasks still present after deletion: showing {parser.tasks_showing} tasks")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking deletion confirmation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_tasks'] = self.check_checkpoint_1()
            self.checkpoints['cp2_filtered_by_low_priority'] = self.check_checkpoint_2()
            self.checkpoints['cp3_selected_all_tasks'] = self.check_checkpoint_3()
            self.checkpoints['cp4_bulk_deleted_tasks'] = self.check_checkpoint_4()
            self.checkpoints['cp5_confirmed_deletion'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 7,
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
                'query_id': 7,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
