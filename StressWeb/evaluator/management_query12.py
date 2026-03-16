#!/usr/bin/env python3
# Evaluator for management Query 12

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
        self.current_user = None
        self.in_user_name = False
        self.in_tasklist_subtitle = False
        self.tasklist_subtitle = None
        self.due_today_tasks = []
        self.in_due_today_section = False
        self.in_task_title = False
        self.current_task_title = None
        self.task_checked = False
        self.in_section_title = False
        self.current_section_title = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for user name span
        if tag == 'span' and attrs_dict.get('class') == 'user-name':
            self.in_user_name = True
        
        # Check for tasklist subtitle
        if tag == 'p' and 'tasklist-subtitle' in attrs_dict.get('class', ''):
            self.in_tasklist_subtitle = True
        
        # Check for section title
        if tag == 'h3' and 'section-title' in attrs_dict.get('class', ''):
            self.in_section_title = True
        
        # Check for task title link
        if tag == 'a' and 'today-task-title' in attrs_dict.get('class', ''):
            self.in_task_title = True
        
        # Check for task checkbox
        if tag == 'input' and attrs_dict.get('type') == 'checkbox':
            self.task_checked = 'checked' in attrs_dict

    def handle_data(self, data):
        if self.in_user_name:
            self.current_user = data.strip()
            self.in_user_name = False
        
        if self.in_tasklist_subtitle:
            self.tasklist_subtitle = data.strip()
            self.in_tasklist_subtitle = False
        
        if self.in_section_title:
            self.current_section_title = data.strip()
            if self.current_section_title == "Due Today":
                self.in_due_today_section = True
            self.in_section_title = False
        
        if self.in_task_title:
            self.current_task_title = data.strip()
            self.in_task_title = False
            if self.in_due_today_section:
                self.due_today_tasks.append({
                    'title': self.current_task_title,
                    'checked': self.task_checked
                })
            self.task_checked = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_switched_to_john_doe": False,
            "cp2_found_due_today_tasks": False,
            "cp3_deleted_all_due_today_tasks": False,
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
        """Find intermediate step HTML files.

        Args:
            step_pattern: Pattern like "step_*", "step_10_*", "step_[12]*"
        Returns:
            List of matching HTML files sorted by step number
        """
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
        """Checkpoint 1: User switched to john_doe"""
        try:
            # Check intermediate steps where user switch happened (around step 17)
            step_17_files = self.find_step_html("step_17_*")
            if step_17_files:
                parser = self.parse_html_file(step_17_files[0])
                if parser.current_user == "john_doe":
                    self.details['switched_user'] = 'john_doe'
                    return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html:
                parser = self.parse_html_file(final_html)
                if parser.current_user == "john_doe":
                    self.details['final_user'] = 'john_doe'
                    return True
            
            self.issues.append("Failed to switch to user john_doe")
            return False
        except Exception as e:
            self.issues.append(f"Error checking user switch: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Found tasks due today for john_doe"""
        try:
            # Check step 17 where john_doe's dashboard shows Due Today tasks
            step_17_files = self.find_step_html("step_17_*")
            if step_17_files:
                parser = self.parse_html_file(step_17_files[0])
                if parser.due_today_tasks:
                    self.details['due_today_tasks_found'] = len(parser.due_today_tasks)
                    self.details['task_titles'] = [task['title'] for task in parser.due_today_tasks]
                    return True
            
            self.issues.append("Could not find Due Today tasks for john_doe")
            return False
        except Exception as e:
            self.issues.append(f"Error checking due today tasks: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: All Due Today tasks deleted (showing 0 tasks)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML file not found")
                return False
            
            parser = self.parse_html_file(final_html)
            
            # Check if on tasks page and filtered by Today
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for "Showing 0 of" pattern indicating no tasks
            showing_match = re.search(r'Showing (\d+) of (\d+) tasks', content)
            if showing_match:
                shown_tasks = int(showing_match.group(1))
                if shown_tasks == 0:
                    self.details['final_due_today_count'] = 0
                    self.details['deletion_confirmed'] = True
                    return True
                else:
                    self.issues.append(f"Still showing {shown_tasks} tasks due today, expected 0")
                    self.details['final_due_today_count'] = shown_tasks
                    return False
            
            # Check for "No tasks found" message as alternative
            if 'No tasks found' in content and 'tasklist' in content:
                self.details['final_due_today_count'] = 0
                self.details['deletion_confirmed'] = True
                return True
            
            self.issues.append("Could not verify deletion of Due Today tasks")
            return False
        except Exception as e:
            self.issues.append(f"Error checking task deletion: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_switched_to_john_doe'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_due_today_tasks'] = self.check_checkpoint_2()
            self.checkpoints['cp3_deleted_all_due_today_tasks'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 12,
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
                'query_id': 12,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
