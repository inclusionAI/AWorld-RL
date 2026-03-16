#!/usr/bin/env python3
# Evaluator for management Query 18

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task information from the task list."""

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.current_task = None
        self.in_task_card = False
        self.in_task_title = False
        self.in_task_status = False
        self.in_sort_button = False
        self.in_user_name = False
        self.current_sort = None
        self.current_user = None
        self.task_card_classes = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for task card
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'task-card' in classes:
                self.in_task_card = True
                self.task_card_classes = classes.split()
                self.current_task = {
                    'title': '',
                    'status': '',
                    'priority': '',
                    'selected': 'selected' in classes,
                    'checkbox_checked': False
                }
        
        # Check for task title link
        if self.in_task_card and tag == 'a' and 'class' in attrs_dict:
            if 'task-title' in attrs_dict['class']:
                self.in_task_title = True
        
        # Check for status span
        if self.in_task_card and tag == 'span' and 'class' in attrs_dict:
            if 'task-status' in attrs_dict['class']:
                self.in_task_status = True
            elif 'task-priority' in attrs_dict['class']:
                self.in_task_priority = True
        
        # Check for checkbox
        if self.in_task_card and tag == 'input' and 'class' in attrs_dict:
            if 'task-checkbox' in attrs_dict['class']:
                if 'checked' in attrs_dict or ('checked', '') in attrs:
                    self.current_task['checkbox_checked'] = True
        
        # Check for sort button
        if tag == 'button' and 'class' in attrs_dict:
            if 'sort-button' in attrs_dict['class']:
                self.in_sort_button = True
        
        # Check for user name
        if tag == 'span' and 'class' in attrs_dict:
            if 'user-name' in attrs_dict['class']:
                self.in_user_name = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_task_title:
            self.current_task['title'] = data
            self.in_task_title = False
        elif self.in_task_status:
            self.current_task['status'] = data
            self.in_task_status = False
        elif hasattr(self, 'in_task_priority') and self.in_task_priority:
            self.current_task['priority'] = data
            self.in_task_priority = False
        elif self.in_sort_button:
            self.current_sort = data
        elif self.in_user_name:
            self.current_user = data

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_task_card:
            if self.current_task and self.current_task.get('title'):
                self.tasks.append(self.current_task)
            self.in_task_card = False
            self.current_task = None
            self.task_card_classes = []
        elif tag == 'button' and self.in_sort_button:
            self.in_sort_button = False
        elif tag == 'span' and self.in_user_name:
            self.in_user_name = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_correct_user": False,
            "cp2_sorted_by_priority": False,
            "cp3_first_task_marked_completed": False,
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

    def parse_html_file(self, html_file: Path) -> TaskHTMLParser:
        """Parse an HTML file and return the parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Verify user is sarah.johnson"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html_file(html_file)
            
            if parser.current_user:
                self.details['current_user'] = parser.current_user
                if parser.current_user == 'sarah.johnson':
                    return True
                else:
                    self.issues.append(f"Wrong user: {parser.current_user} (expected sarah.johnson)")
                    return False
            else:
                self.issues.append("Could not detect current user")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking user: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Verify tasks are sorted by Priority (High to Low)"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html_file(html_file)
            
            # Check sort button text
            if parser.current_sort:
                self.details['current_sort'] = parser.current_sort
                if 'Priority (High to Low)' in parser.current_sort or 'Priority' in parser.current_sort:
                    # Verify actual task order follows priority
                    if parser.tasks:
                        priorities = [task.get('priority', '') for task in parser.tasks[:10]]
                        self.details['first_10_priorities'] = priorities
                        
                        # Check if urgent/high priority tasks come first
                        priority_order = {'🔴 Urgent': 4, '🟠 High': 3, '🟡 Medium': 2, '🟢 Low': 1}
                        
                        # Verify first few tasks are high priority
                        first_task_priority = parser.tasks[0].get('priority', '')
                        if '🔴' in first_task_priority or 'Urgent' in first_task_priority:
                            return True
                        elif '🟠' in first_task_priority or 'High' in first_task_priority:
                            return True
                        else:
                            self.issues.append(f"First task has priority '{first_task_priority}', not Urgent or High")
                            return False
                    else:
                        self.issues.append("No tasks found in HTML")
                        return False
                else:
                    self.issues.append(f"Not sorted by Priority (High to Low): {parser.current_sort}")
                    return False
            else:
                self.issues.append("Could not detect sort order")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking sort: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Verify first task is marked as completed"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No final HTML file found")
                return False
            
            parser = self.parse_html_file(html_file)
            
            if not parser.tasks:
                self.issues.append("No tasks found")
                return False
            
            first_task = parser.tasks[0]
            self.details['first_task_title'] = first_task.get('title', '')
            self.details['first_task_status'] = first_task.get('status', '')
            self.details['first_task_checkbox_checked'] = first_task.get('checkbox_checked', False)
            
            # Check if task is marked as completed (either by status or checkbox)
            status = first_task.get('status', '').lower()
            checkbox_checked = first_task.get('checkbox_checked', False)
            
            if 'completed' in status or checkbox_checked:
                return True
            else:
                self.issues.append(
                    f"First task '{first_task.get('title', '')}' is not marked as completed. "
                    f"Status: {status}, Checkbox: {checkbox_checked}"
                )
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking first task completion: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_correct_user'] = self.check_checkpoint_1()
            self.checkpoints['cp2_sorted_by_priority'] = self.check_checkpoint_2()
            self.checkpoints['cp3_first_task_marked_completed'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 18,
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
                'query_id': 18,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query18.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
