#!/usr/bin/env python3
# Evaluator for management Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class DueTodayHTMLParser(HTMLParser):
    """Parse HTML to extract Due Today section tasks and their checkbox states."""

    def __init__(self):
        super().__init__()
        self.in_due_today_section = False
        self.in_today_task = False
        self.in_task_content = False
        self.in_task_title = False
        self.in_task_meta = False
        self.current_task = {}
        self.tasks = []
        self.depth = 0
        self.section_title = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for section title
        if tag == 'h3' and 'class' in attrs_dict and 'section-title' in attrs_dict['class']:
            self.section_title = ""
            
        # Check if we're entering the Due Today section
        if tag == 'div' and 'class' in attrs_dict and 'dashboard-section' in attrs_dict['class']:
            self.depth += 1
            
        # Check for today-task div
        if self.in_due_today_section and tag == 'div' and 'class' in attrs_dict and 'today-task' in attrs_dict['class']:
            self.in_today_task = True
            self.current_task = {'checked': False, 'title': '', 'meta': ''}
            
        # Check for checkbox within today-task
        if self.in_today_task and tag == 'input' and attrs_dict.get('type') == 'checkbox':
            if 'checked' in attrs_dict or ('checked' in str(attrs) and attrs_dict.get('checked') == ''):
                self.current_task['checked'] = True
            else:
                self.current_task['checked'] = False
                
        # Check for task content
        if self.in_today_task and tag == 'div' and 'class' in attrs_dict and 'today-task-content' in attrs_dict['class']:
            self.in_task_content = True
            
        # Check for task title link
        if self.in_task_content and tag == 'a' and 'class' in attrs_dict and 'today-task-title' in attrs_dict['class']:
            self.in_task_title = True
            
        # Check for task meta
        if self.in_task_content and tag == 'p' and 'class' in attrs_dict and 'today-task-meta' in attrs_dict['class']:
            self.in_task_meta = True

    def handle_data(self, data):
        if self.section_title == "" and not self.in_due_today_section:
            text = data.strip()
            if text == "Due Today":
                self.in_due_today_section = True
            self.section_title = text
            
        if self.in_task_title:
            self.current_task['title'] = data.strip()
            
        if self.in_task_meta:
            self.current_task['meta'] = data.strip()

    def handle_endtag(self, tag):
        if tag == 'h3':
            self.section_title = ""
            
        if tag == 'a' and self.in_task_title:
            self.in_task_title = False
            
        if tag == 'p' and self.in_task_meta:
            self.in_task_meta = False
            
        if tag == 'div' and self.in_task_content:
            self.in_task_content = False
            
        if tag == 'div' and self.in_today_task:
            self.in_today_task = False
            if self.current_task:
                self.tasks.append(self.current_task)
                self.current_task = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_due_today_section_found": False,
            "cp2_tasks_present_in_due_today": False,
            "cp3_checkbox_clicked": False,
            "cp4_task_marked_completed": False,
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

    def parse_due_today_section(self, html_path: Path) -> List[Dict]:
        """Parse HTML and extract Due Today section tasks."""
        parser = DueTodayHTMLParser()
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            parser.feed(html_content)
            
        return parser.tasks

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Due Today section exists in final HTML."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        if 'Due Today' in html_content:
            self.details['due_today_section_found'] = True
            return True
        else:
            self.issues.append("Due Today section not found in final HTML")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Tasks present in Due Today section (initial state)."""
        initial_files = list(self.result_dir.glob("initial_*_raw.html"))
        if not initial_files:
            self.issues.append("No initial HTML file found")
            return False
            
        initial_tasks = self.parse_due_today_section(initial_files[0])
        
        if len(initial_tasks) > 0:
            self.details['initial_due_today_tasks'] = initial_tasks
            self.details['initial_task_count'] = len(initial_tasks)
            return True
        else:
            self.issues.append("No tasks found in Due Today section initially")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Checkbox was clicked (from trajectory)."""
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        checkbox_clicked = False
        for action in action_history:
            action_data = action.get('action', {})
            if action_data.get('action_type') == 'CLICK':
                selector = action_data.get('parameters', {}).get('selector', '')
                if 'checkbox' in selector.lower() or "input[type='checkbox']" in selector:
                    checkbox_clicked = True
                    self.details['checkbox_click_action'] = action_data
                    break
                    
        if checkbox_clicked:
            return True
        else:
            self.issues.append("No checkbox click action found in trajectory")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Task is marked as completed (checkbox checked in final state)."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
            
        final_tasks = self.parse_due_today_section(final_html)
        self.details['final_due_today_tasks'] = final_tasks
        self.details['final_task_count'] = len(final_tasks)
        
        # Check if any task has checked checkbox
        completed_tasks = [task for task in final_tasks if task.get('checked', False)]
        
        if len(completed_tasks) > 0:
            self.details['completed_tasks'] = completed_tasks
            self.details['completed_task_count'] = len(completed_tasks)
            return True
        else:
            self.issues.append("No tasks marked as completed in Due Today section")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_due_today_section_found'] = self.check_checkpoint_1()
            self.checkpoints['cp2_tasks_present_in_due_today'] = self.check_checkpoint_2()
            self.checkpoints['cp3_checkbox_clicked'] = self.check_checkpoint_3()
            self.checkpoints['cp4_task_marked_completed'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 3,
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
                'query_id': 3,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
