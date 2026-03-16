#!/usr/bin/env python3
# Evaluator for management Query 11

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
        self.current_task = {}
        self.in_task_card = False
        self.in_task_title = False
        self.in_task_priority = False
        self.in_task_description = False
        self.in_task_due_date = False
        self.current_data = []
        self.current_class = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get('class', '')
        
        # Detect task card
        if tag == 'div' and 'task-card' in class_name:
            self.in_task_card = True
            self.current_task = {}
        
        # Detect task title
        if self.in_task_card and tag == 'h3' and 'task-title' in class_name:
            self.in_task_title = True
            self.current_data = []
        
        # Detect priority badge
        if self.in_task_card and tag == 'span' and 'priority-badge' in class_name:
            self.in_task_priority = True
            self.current_data = []
            self.current_class = class_name
        
        # Detect description
        if self.in_task_card and tag == 'p' and 'task-description' in class_name:
            self.in_task_description = True
            self.current_data = []
        
        # Detect due date
        if self.in_task_card and tag == 'span' and 'due-date' in class_name:
            self.in_task_due_date = True
            self.current_data = []

    def handle_data(self, data):
        data = data.strip()
        if data:
            if self.in_task_title or self.in_task_priority or self.in_task_description or self.in_task_due_date:
                self.current_data.append(data)

    def handle_endtag(self, tag):
        if self.in_task_title and tag == 'h3':
            self.current_task['title'] = ' '.join(self.current_data).strip()
            self.in_task_title = False
            self.current_data = []
        
        if self.in_task_priority and tag == 'span':
            priority_text = ' '.join(self.current_data).strip()
            self.current_task['priority'] = priority_text
            self.current_task['priority_class'] = self.current_class
            self.in_task_priority = False
            self.current_data = []
        
        if self.in_task_description and tag == 'p':
            self.current_task['description'] = ' '.join(self.current_data).strip()
            self.in_task_description = False
            self.current_data = []
        
        if self.in_task_due_date and tag == 'span':
            self.current_task['due_date'] = ' '.join(self.current_data).strip()
            self.in_task_due_date = False
            self.current_data = []
        
        if self.in_task_card and tag == 'div':
            if self.current_task:
                self.tasks.append(self.current_task.copy())
            self.in_task_card = False
            self.current_task = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_task_created": False,
            "cp2_title_updated": False,
            "cp3_priority_updated": False,
            "cp4_description_added": False,
            "cp5_due_date_set": False,
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

    def parse_tasks_from_html(self, html_path: Path) -> List[Dict]:
        """Parse tasks from HTML file."""
        if not html_path.exists():
            return []
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = TaskHTMLParser()
        parser.feed(html_content)
        return parser.tasks

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Task was created (with any title or priority)"""
        # Since the agent failed to create the task (result.json shows failure),
        # we need to check if ANY task exists that could match our criteria
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        
        # Look for a task with either the original or updated title
        target_titles = [
            "Complete project proposal",
            "Complete and submit project proposal"
        ]
        
        matching_tasks = [t for t in tasks if t.get('title') in target_titles]
        
        if matching_tasks:
            self.details['task_found'] = matching_tasks[0]
            return True
        else:
            self.issues.append("Task 'Complete project proposal' or 'Complete and submit project proposal' not found")
            self.details['found_tasks'] = [t.get('title', '') for t in tasks[:5]]
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Title was changed to 'Complete and submit project proposal'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        
        matching_task = None
        for task in tasks:
            if task.get('title') == "Complete and submit project proposal":
                matching_task = task
                break
        
        if matching_task:
            self.details['final_title'] = matching_task['title']
            return True
        else:
            self.issues.append("Task title not updated to 'Complete and submit project proposal'")
            # Check if original title exists
            for task in tasks:
                if task.get('title') == "Complete project proposal":
                    self.details['original_title_found'] = True
                    self.issues.append("Found original title but not updated title")
                    break
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Priority updated to Urgent"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        
        target_task = None
        for task in tasks:
            if task.get('title') == "Complete and submit project proposal":
                target_task = task
                break
        
        if not target_task:
            self.issues.append("Cannot check priority - task not found")
            return False
        
        priority = target_task.get('priority', '').lower()
        if 'urgent' in priority:
            self.details['priority'] = target_task.get('priority')
            return True
        else:
            self.issues.append(f"Priority not set to Urgent (found: {target_task.get('priority', 'none')})")
            self.details['priority'] = target_task.get('priority')
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Description added: 'Must be submitted to the client by Friday'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        
        target_task = None
        for task in tasks:
            if task.get('title') == "Complete and submit project proposal":
                target_task = task
                break
        
        if not target_task:
            self.issues.append("Cannot check description - task not found")
            return False
        
        description = target_task.get('description', '').strip()
        expected_desc = "Must be submitted to the client by Friday"
        
        if description == expected_desc:
            self.details['description'] = description
            return True
        else:
            self.issues.append(f"Description not correct (found: '{description}')")
            self.details['description'] = description
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Due date set to 2026-02-07"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        
        target_task = None
        for task in tasks:
            if task.get('title') == "Complete and submit project proposal":
                target_task = task
                break
        
        if not target_task:
            self.issues.append("Cannot check due date - task not found")
            return False
        
        due_date = target_task.get('due_date', '').strip()
        
        # Expected formats: "2026-02-07" or "Feb 7, 2026" or similar
        if '2026-02-07' in due_date or '02/07/2026' in due_date or '2026/02/07' in due_date:
            self.details['due_date'] = due_date
            return True
        elif 'Feb' in due_date and '7' in due_date and '2026' in due_date:
            self.details['due_date'] = due_date
            return True
        else:
            self.issues.append(f"Due date not set to 2026-02-07 (found: '{due_date}')")
            self.details['due_date'] = due_date
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_task_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_title_updated'] = self.check_checkpoint_2()
            self.checkpoints['cp3_priority_updated'] = self.check_checkpoint_3()
            self.checkpoints['cp4_description_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_due_date_set'] = self.check_checkpoint_5()

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
        print("Usage: python management_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
