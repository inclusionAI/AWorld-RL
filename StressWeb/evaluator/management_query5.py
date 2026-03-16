#!/usr/bin/env python3
# Evaluator for management Query 5

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
        self.in_task_card = False
        self.in_task_title = False
        self.in_task_description = False
        self.in_task_meta = False
        self.in_priority = False
        self.in_due_date = False
        self.current_task = {}
        self.tasks = []
        self.current_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "div" and attrs_dict.get("class") == "task-card":
            self.in_task_card = True
            self.current_task = {}
        elif self.in_task_card:
            if tag == "a" and "task-title" in attrs_dict.get("class", ""):
                self.in_task_title = True
                self.current_text = ""
            elif tag == "p" and "task-description" in attrs_dict.get("class", ""):
                self.in_task_description = True
                self.current_text = ""
            elif tag == "div" and "task-meta" in attrs_dict.get("class", ""):
                self.in_task_meta = True
            elif self.in_task_meta and tag == "span":
                if "task-priority" in attrs_dict.get("class", ""):
                    self.in_priority = True
                    self.current_text = ""
                elif "task-due-date" in attrs_dict.get("class", ""):
                    self.in_due_date = True
                    self.current_text = ""

    def handle_data(self, data):
        if self.in_task_title:
            self.current_text += data.strip()
        elif self.in_task_description:
            self.current_text += data.strip()
        elif self.in_priority:
            self.current_text += data.strip()
        elif self.in_due_date:
            self.current_text += data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self.in_task_title:
            self.current_task["title"] = self.current_text
            self.in_task_title = False
        elif tag == "p" and self.in_task_description:
            self.current_task["description"] = self.current_text
            self.in_task_description = False
        elif tag == "span" and self.in_priority:
            self.current_task["priority"] = self.current_text
            self.in_priority = False
        elif tag == "span" and self.in_due_date:
            self.current_task["due_date"] = self.current_text
            self.in_due_date = False
        elif tag == "div" and self.in_task_meta:
            self.in_task_meta = False
        elif tag == "div" and self.in_task_card:
            self.tasks.append(self.current_task)
            self.in_task_card = False


class FormHTMLParser(HTMLParser):
    """Parse HTML to extract form field values."""

    def __init__(self):
        super().__init__()
        self.form_data = {}
        self.in_form = False
        self.current_text = ""
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "form" and "newtask-form" in attrs_dict.get("class", ""):
            self.in_form = True
        elif self.in_form:
            if tag == "input" and attrs_dict.get("id") == "title":
                self.form_data["title"] = attrs_dict.get("value", "")
            elif tag == "input" and attrs_dict.get("id") == "due_date":
                self.form_data["due_date"] = attrs_dict.get("value", "")
            elif tag == "select" and attrs_dict.get("id") == "priority":
                self.form_data["priority_field_exists"] = True
            elif tag == "textarea" and attrs_dict.get("id") == "description":
                self.current_text = ""
                
    def handle_data(self, data):
        if self.in_form and self.current_text is not None:
            self.current_text += data.strip()
    
    def handle_endtag(self, tag):
        if tag == "textarea" and self.current_text is not None:
            self.form_data["description"] = self.current_text
            self.current_text = None
        elif tag == "form" and self.in_form:
            self.in_form = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_task_created": False,
            "cp2_correct_title": False,
            "cp3_priority_medium": False,
            "cp4_correct_description": False,
            "cp5_correct_due_date": False,
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

    def parse_tasks_from_html(self, html_file: Path) -> List[Dict]:
        """Parse tasks from HTML file."""
        if not html_file.exists():
            return []
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = TaskHTMLParser()
        parser.feed(html_content)
        return parser.tasks

    def parse_form_from_html(self, html_file: Path) -> Dict:
        """Parse form data from HTML file."""
        if not html_file.exists():
            return {}
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = FormHTMLParser()
        parser.feed(html_content)
        return parser.form_data

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Task was created in the system"""
        # Check final HTML for the task
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        
        # Look for the task by title
        target_task = None
        for task in tasks:
            if task.get("title") == "Client presentation preparation":
                target_task = task
                break
        
        if target_task:
            self.details["task_found"] = True
            self.details["task_data"] = target_task
            return True
        else:
            self.issues.append("Task 'Client presentation preparation' not found in final task list")
            self.details["task_found"] = False
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Task has correct title"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        for task in tasks:
            if task.get("title") == "Client presentation preparation":
                self.details["title_correct"] = True
                return True
        
        self.issues.append("Task title is not exactly 'Client presentation preparation'")
        self.details["title_correct"] = False
        return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Priority is set to Medium"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        for task in tasks:
            if task.get("title") == "Client presentation preparation":
                priority = task.get("priority", "")
                self.details["priority"] = priority
                
                # Check if priority contains "Medium"
                if "Medium" in priority or "🟡" in priority:
                    self.details["priority_correct"] = True
                    return True
                else:
                    self.issues.append(f"Task priority is '{priority}', expected Medium")
                    self.details["priority_correct"] = False
                    return False
        
        # If we're here, task wasn't found
        self.issues.append("Cannot verify priority - task not found")
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Description is set correctly"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        expected_description = "Prepare presentation slides and demo data"
        
        for task in tasks:
            if task.get("title") == "Client presentation preparation":
                description = task.get("description", "")
                self.details["description"] = description
                
                if description == expected_description:
                    self.details["description_correct"] = True
                    return True
                else:
                    self.issues.append(f"Task description is '{description}', expected '{expected_description}'")
                    self.details["description_correct"] = False
                    return False
        
        self.issues.append("Cannot verify description - task not found")
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Due date is set to 2026-02-10"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        tasks = self.parse_tasks_from_html(final_html)
        
        for task in tasks:
            if task.get("title") == "Client presentation preparation":
                due_date = task.get("due_date", "")
                self.details["due_date"] = due_date
                
                # Check for various date formats
                if "Feb 10, 2026" in due_date or "2026-02-10" in due_date or "10" in due_date:
                    self.details["due_date_correct"] = True
                    return True
                else:
                    self.issues.append(f"Task due date is '{due_date}', expected Feb 10, 2026")
                    self.details["due_date_correct"] = False
                    return False
        
        self.issues.append("Cannot verify due date - task not found")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_task_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_correct_title'] = self.check_checkpoint_2()
            self.checkpoints['cp3_priority_medium'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_description'] = self.check_checkpoint_4()
            self.checkpoints['cp5_correct_due_date'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 5,
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
                'query_id': 5,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
