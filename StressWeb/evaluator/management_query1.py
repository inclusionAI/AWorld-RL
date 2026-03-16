#!/usr/bin/env python3
# Evaluator for management Query 1

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
        self.tasks = []
        self.in_task = False
        self.current_task = {}
        self.in_task_title = False
        self.in_priority_item = False
        self.current_priority = None
        self.priority_counts = {}
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for priority items on dashboard
        if tag == 'a' and 'class' in attrs_dict and 'priority-item' in attrs_dict['class']:
            self.in_priority_item = True
            href = attrs_dict.get('href', '')
            if 'priority=' in href:
                self.current_priority = href.split('priority=')[1].split('&')[0]
        
        # Check for priority count
        if tag == 'span' and 'class' in attrs_dict and 'priority-count' in attrs_dict['class']:
            self.in_priority_count = True
            
        # Check for task title links
        if tag == 'a' and 'class' in attrs_dict and 'today-task-title' in attrs_dict['class']:
            self.in_task_title = True
            self.current_task = {'title': '', 'href': attrs_dict.get('href', '')}

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture priority counts
        if hasattr(self, 'in_priority_count') and self.in_priority_count and self.current_priority:
            try:
                count = int(data)
                self.priority_counts[self.current_priority] = count
            except ValueError:
                pass
            self.in_priority_count = False
            
        # Capture task titles
        if self.in_task_title:
            self.current_task['title'] = data
            
        # Look for task title anywhere in the page (for verification)
        if 'Review Q1 Budget Report' in data:
            self.tasks.append({'title': data, 'found': True})

    def handle_endtag(self, tag):
        if tag == 'a' and self.in_priority_item:
            self.in_priority_item = False
            self.current_priority = None
            
        if tag == 'a' and self.in_task_title:
            if self.current_task.get('title'):
                self.tasks.append(self.current_task)
            self.in_task_title = False
            self.current_task = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_tasks": False,
            "cp2_opened_new_task_form": False,
            "cp3_filled_task_title": False,
            "cp4_set_priority_to_high": False,
            "cp5_task_created_successfully": False,
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
        """Checkpoint 1: Navigated to tasks page"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if there was a click on Tasks link
            for action_entry in action_history:
                action = action_entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Tasks' in selector or 'tasks' in selector.lower():
                        self.details['navigated_to_tasks'] = True
                        return True
                        
            # Also check trajectory for navigation to /tasks
            trajectory = self.get_trajectory()
            for entry in trajectory:
                page_info = entry.get('page_info', {})
                url = page_info.get('url', '')
                if '/tasks' in url.lower():
                    self.details['navigated_to_tasks'] = True
                    return True
                    
            self.issues.append("Did not navigate to tasks page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Opened new task creation form"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if "New Task" button was clicked
            for action_entry in action_history:
                action = action_entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    reasoning = action.get('reasoning', '')
                    result = action_entry.get('result', {})
                    
                    if ('New Task' in selector or 'New Task' in reasoning) and result.get('success'):
                        self.details['opened_new_task_form'] = True
                        return True
                        
            self.issues.append("Did not successfully open new task form")
            return False
        except Exception as e:
            self.issues.append(f"Error checking form opening: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Filled task title correctly"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if task title was filled
            for action_entry in action_history:
                action = action_entry.get('action', {})
                if action.get('action_type') == 'FILL':
                    text = action.get('parameters', {}).get('text', '')
                    if text == 'Review Q1 Budget Report':
                        result = action_entry.get('result', {})
                        if result.get('success'):
                            self.details['task_title_filled'] = text
                            return True
                            
            self.issues.append("Task title 'Review Q1 Budget Report' was not filled correctly")
            return False
        except Exception as e:
            self.issues.append(f"Error checking task title: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Set priority to High"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if priority was set to High
            for action_entry in action_history:
                action = action_entry.get('action', {})
                if action.get('action_type') == 'SELECT':
                    label = action.get('parameters', {}).get('label', '')
                    selector = action.get('parameters', {}).get('selector', '')
                    
                    if label == 'High' and 'priority' in selector.lower():
                        result = action_entry.get('result', {})
                        if result.get('success'):
                            self.details['priority_set_to'] = 'High'
                            return True
                            
            self.issues.append("Priority was not set to High")
            return False
        except Exception as e:
            self.issues.append(f"Error checking priority: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Task created successfully and appears in system"""
        try:
            # First check if Create Task button was clicked
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            create_clicked = False
            for action_entry in action_history:
                action = action_entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Create Task' in selector:
                        result = action_entry.get('result', {})
                        if result.get('success'):
                            create_clicked = True
                            break
                            
            if not create_clicked:
                self.issues.append("Create Task button was not clicked successfully")
                return False
                
            # Check final HTML for the task
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
                
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            # Parse HTML to check for task and High priority count
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            # Check if task title appears in HTML
            if 'Review Q1 Budget Report' in html_content:
                self.details['task_found_in_html'] = True
                
                # Check if High priority count increased
                high_count = parser.priority_counts.get('high', 0)
                if high_count > 0:
                    self.details['high_priority_count'] = high_count
                    self.details['task_created_successfully'] = True
                    return True
                else:
                    # Still consider success if task title is present
                    self.details['task_created_successfully'] = True
                    return True
            else:
                self.issues.append("Task 'Review Q1 Budget Report' not found in final HTML")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking task creation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_tasks'] = self.check_checkpoint_1()
            self.checkpoints['cp2_opened_new_task_form'] = self.check_checkpoint_2()
            self.checkpoints['cp3_filled_task_title'] = self.check_checkpoint_3()
            self.checkpoints['cp4_set_priority_to_high'] = self.check_checkpoint_4()
            self.checkpoints['cp5_task_created_successfully'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 1,
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
                'query_id': 1,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
