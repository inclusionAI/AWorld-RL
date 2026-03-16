#!/usr/bin/env python3
# Evaluator for management Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task information."""

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.current_task = {}
        self.in_task_card = False
        self.in_task_title = False
        self.in_task_priority = False
        self.in_task_status = False
        self.capture_data = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict and 'task-card' in attrs_dict['class']:
            self.in_task_card = True
            self.current_task = {}
        
        if self.in_task_card:
            if tag == 'a' and 'class' in attrs_dict and 'task-title' in attrs_dict['class']:
                self.in_task_title = True
                self.capture_data = True
            elif tag == 'h1' and 'class' in attrs_dict and 'taskdetail-title' in attrs_dict['class']:
                self.in_task_title = True
                self.capture_data = True
            elif tag == 'span' and 'class' in attrs_dict:
                if 'task-priority' in attrs_dict['class']:
                    self.in_task_priority = True
                    self.capture_data = True
                elif 'task-status' in attrs_dict['class']:
                    self.in_task_status = True
                    self.capture_data = True
                elif 'meta-value' in attrs_dict['class']:
                    # Check if this is in task detail page priority section
                    if 'style' in attrs_dict:
                        self.in_task_priority = True
                        self.capture_data = True

    def handle_data(self, data):
        if self.capture_data:
            text = data.strip()
            if text:
                if self.in_task_title:
                    self.current_task['title'] = text
                elif self.in_task_priority:
                    # Extract priority level from text like "🟠 High" or just "High"
                    priority_match = re.search(r'(Urgent|High|Medium|Low)', text, re.IGNORECASE)
                    if priority_match:
                        self.current_task['priority'] = priority_match.group(1).capitalize()
                elif self.in_task_status:
                    self.current_task['status'] = text.lower()

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_task_card:
            if self.current_task:
                self.tasks.append(self.current_task.copy())
            self.in_task_card = False
            self.current_task = {}
        
        if tag in ['a', 'h1', 'span']:
            self.in_task_title = False
            self.in_task_priority = False
            self.in_task_status = False
            self.capture_data = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_all_three_tasks_exist": False,
            "cp2_send_client_proposal_high": False,
            "cp3_update_project_timeline_high": False,
            "cp4_schedule_team_standup_high": False,
        }

        self.issues = []
        self.details = {}
        
        # Target tasks to check
        self.target_tasks = [
            "Send client proposal",
            "Update project timeline", 
            "Schedule team standup"
        ]

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

    def parse_tasks_from_html(self, html_file: Path) -> List[Dict]:
        """Parse tasks from HTML file."""
        if not html_file.exists():
            return []
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = TaskHTMLParser()
        parser.feed(content)
        return parser.tasks

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: All three tasks exist in the system."""
        # Check in task list pages (intermediate steps around step 20-22)
        task_list_files = self.find_step_html("step_2*")
        
        tasks_found = set()
        for html_file in task_list_files:
            tasks = self.parse_tasks_from_html(html_file)
            for task in tasks:
                title = task.get('title', '').lower()
                for target in self.target_tasks:
                    if target.lower() in title:
                        tasks_found.add(target)
        
        # Also check final state
        final_html = self.find_final_html()
        if final_html:
            tasks = self.parse_tasks_from_html(final_html)
            for task in tasks:
                title = task.get('title', '').lower()
                for target in self.target_tasks:
                    if target.lower() in title:
                        tasks_found.add(target)
        
        self.details['tasks_found'] = list(tasks_found)
        
        if len(tasks_found) == 3:
            return True
        else:
            missing = set(self.target_tasks) - tasks_found
            self.issues.append(f"Missing tasks: {missing}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: 'Send client proposal' has High priority."""
        return self._check_task_priority("Send client proposal", "High")

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: 'Update project timeline' has High priority."""
        return self._check_task_priority("Update project timeline", "High")

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: 'Schedule team standup' has High priority."""
        return self._check_task_priority("Schedule team standup", "High")

    def _check_task_priority(self, task_title: str, expected_priority: str) -> bool:
        """Check if a specific task has the expected priority."""
        # Check final state first
        final_html = self.find_final_html()
        if final_html:
            tasks = self.parse_tasks_from_html(final_html)
            for task in tasks:
                if task.get('title', '').lower() == task_title.lower():
                    priority = task.get('priority', '').capitalize()
                    self.details[f'{task_title}_priority_final'] = priority
                    if priority == expected_priority:
                        return True
                    else:
                        self.issues.append(f"Task '{task_title}' has priority '{priority}', expected '{expected_priority}'")
                        return False
        
        # Check in intermediate steps (task list pages)
        task_list_files = self.find_step_html("step_2*")
        for html_file in reversed(task_list_files):  # Check from most recent
            tasks = self.parse_tasks_from_html(html_file)
            for task in tasks:
                if task.get('title', '').lower() == task_title.lower():
                    priority = task.get('priority', '').capitalize()
                    if priority == expected_priority:
                        return True
        
        self.issues.append(f"Task '{task_title}' not found or doesn't have priority '{expected_priority}'")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_all_three_tasks_exist'] = self.check_checkpoint_1()
            self.checkpoints['cp2_send_client_proposal_high'] = self.check_checkpoint_2()
            self.checkpoints['cp3_update_project_timeline_high'] = self.check_checkpoint_3()
            self.checkpoints['cp4_schedule_team_standup_high'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 6,
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
                'query_id': 6,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
