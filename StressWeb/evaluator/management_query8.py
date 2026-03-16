#!/usr/bin/env python3
# Evaluator for management Query 8

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
        self.in_task_card = False
        self.in_task_title = False
        self.in_task_status = False
        self.in_task_priority = False
        self.current_task = {}
        self.tasks = []
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict and 'task-card' in attrs_dict['class']:
            self.in_task_card = True
            self.current_task = {}
        elif self.in_task_card:
            if tag == 'a' and 'class' in attrs_dict and 'task-title' in attrs_dict['class']:
                self.in_task_title = True
                self.current_data = []
            elif tag == 'span' and 'class' in attrs_dict:
                if 'task-status' in attrs_dict['class']:
                    self.in_task_status = True
                    self.current_data = []
                elif 'task-priority' in attrs_dict['class']:
                    self.in_task_priority = True
                    self.current_data = []

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_task_card:
            if self.current_task:
                self.tasks.append(self.current_task)
                self.current_task = {}
            self.in_task_card = False
        elif tag == 'a' and self.in_task_title:
            self.current_task['title'] = ''.join(self.current_data).strip()
            self.in_task_title = False
        elif tag == 'span':
            if self.in_task_status:
                self.current_task['status'] = ''.join(self.current_data).strip()
                self.in_task_status = False
            elif self.in_task_priority:
                self.current_task['priority'] = ''.join(self.current_data).strip()
                self.in_task_priority = False

    def handle_data(self, data):
        if self.in_task_title or self.in_task_status or self.in_task_priority:
            self.current_data.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_bulk_create_opened": False,
            "cp2_five_tasks_created": False,
            "cp3_correct_titles": False,
            "cp4_urgent_priority": False,
            "cp5_todo_status": False,
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
        """Checkpoint 1: Bulk Create dialog was opened"""
        trajectory = self.get_trajectory()
        
        for step in trajectory:
            action = step.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if 'Bulk Create' in selector or 'bulk-create' in selector.lower():
                    self.details['bulk_create_opened'] = True
                    return True
        
        self.issues.append("Bulk Create button was not clicked")
        self.details['bulk_create_opened'] = False
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Five tasks were created (Annual Review 2-6)"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Find all "Annual Review" tasks with numbers
        annual_review_pattern = r'Annual Review (\d+)'
        matches = re.findall(annual_review_pattern, html_content)
        
        found_numbers = sorted([int(m) for m in matches])
        expected_numbers = [2, 3, 4, 5, 6]
        
        self.details['annual_review_tasks_found'] = found_numbers
        self.details['expected_tasks'] = expected_numbers
        
        if found_numbers == expected_numbers:
            return True
        else:
            self.issues.append(f"Expected tasks Annual Review 2-6, but found: Annual Review {found_numbers}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Tasks have correct titles (Annual Review 2 through Annual Review 6)"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = TaskHTMLParser()
        parser.feed(html_content)
        
        annual_review_tasks = [t for t in parser.tasks if 'Annual Review' in t.get('title', '')]
        
        expected_titles = [f"Annual Review {i}" for i in range(2, 7)]
        actual_titles = [t['title'] for t in annual_review_tasks]
        
        self.details['actual_titles'] = actual_titles
        self.details['expected_titles'] = expected_titles
        
        if sorted(actual_titles) == sorted(expected_titles):
            return True
        else:
            missing = set(expected_titles) - set(actual_titles)
            if missing:
                self.issues.append(f"Missing tasks: {missing}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: All Annual Review tasks have Urgent priority"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = TaskHTMLParser()
        parser.feed(html_content)
        
        annual_review_tasks = [t for t in parser.tasks if 'Annual Review' in t.get('title', '')]
        
        if not annual_review_tasks:
            self.issues.append("No Annual Review tasks found to check priority")
            return False
        
        wrong_priority = []
        for task in annual_review_tasks:
            priority = task.get('priority', '')
            if 'Urgent' not in priority:
                wrong_priority.append(f"{task['title']}: {priority}")
        
        self.details['annual_review_priorities'] = {t['title']: t.get('priority', 'unknown') for t in annual_review_tasks}
        
        if wrong_priority:
            self.issues.append(f"Tasks with wrong priority (should be Urgent): {wrong_priority}")
            return False
        
        return True

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: All Annual Review tasks have To Do status"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = TaskHTMLParser()
        parser.feed(html_content)
        
        annual_review_tasks = [t for t in parser.tasks if 'Annual Review' in t.get('title', '')]
        
        if not annual_review_tasks:
            self.issues.append("No Annual Review tasks found to check status")
            return False
        
        wrong_status = []
        for task in annual_review_tasks:
            status = task.get('status', '')
            if status != 'todo':
                wrong_status.append(f"{task['title']}: {status}")
        
        self.details['annual_review_statuses'] = {t['title']: t.get('status', 'unknown') for t in annual_review_tasks}
        
        if wrong_status:
            self.issues.append(f"Tasks with wrong status (should be 'todo'): {wrong_status}")
            return False
        
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_bulk_create_opened'] = self.check_checkpoint_1()
            self.checkpoints['cp2_five_tasks_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_titles'] = self.check_checkpoint_3()
            self.checkpoints['cp4_urgent_priority'] = self.check_checkpoint_4()
            self.checkpoints['cp5_todo_status'] = self.check_checkpoint_5()

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
        print("Usage: python management_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
