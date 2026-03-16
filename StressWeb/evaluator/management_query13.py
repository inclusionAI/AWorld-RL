#!/usr/bin/env python3
# Evaluator for management Query 13

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task information from the Dashboard."""
    def __init__(self):
        super().__init__()
        self.in_dashboard = False
        self.in_due_today_section = False
        self.in_today_task = False
        self.in_task_title = False
        self.in_task_checkbox = False
        self.current_task = {}
        self.due_today_tasks = []
        self.current_tag = None
        self.current_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_attrs = attrs_dict
        
        # Check for dashboard section
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class'].split()
            if 'dashboard' in classes:
                self.in_dashboard = True
            if 'dashboard-section' in classes:
                # Check next elements for "Due Today" title
                pass
            if 'today-task' in classes:
                self.in_today_task = True
                self.current_task = {'checked': False, 'title': '', 'meta': ''}
                
        # Check for task checkbox
        if self.in_today_task and tag == 'input' and attrs_dict.get('type') == 'checkbox':
            self.in_task_checkbox = True
            # Check if checkbox is checked
            if 'checked' in attrs_dict or attrs_dict.get('checked') == 'checked' or attrs_dict.get('checked') == '':
                self.current_task['checked'] = True
                
        # Check for task title link
        if self.in_today_task and tag == 'a' and 'class' in attrs_dict:
            if 'today-task-title' in attrs_dict['class']:
                self.in_task_title = True
                
        # Check for section titles
        if tag == 'h3' and 'class' in attrs_dict and 'section-title' in attrs_dict['class']:
            self.next_is_section_title = True
    
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Check for "Due Today" section
        if hasattr(self, 'next_is_section_title') and self.next_is_section_title:
            if data == 'Due Today':
                self.in_due_today_section = True
            self.next_is_section_title = False
            
        # Capture task title
        if self.in_task_title and data:
            self.current_task['title'] = data
            self.in_task_title = False
            
        # Capture task meta information
        if self.in_today_task and self.current_tag == 'p' and 'Due today' in data:
            self.current_task['meta'] = data
    
    def handle_endtag(self, tag):
        if tag == 'div' and self.in_today_task:
            if self.current_task.get('title'):
                self.due_today_tasks.append(self.current_task.copy())
            self.in_today_task = False
            self.current_task = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_identified_users": False,
            "cp2_switched_users": False,
            "cp3_marked_all_completed": False,
        }
        
        self.issues = []
        self.details = {}
    
    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        return step_files[-1] if step_files else None
    
    def find_initial_html(self) -> Optional[Path]:
        """Find initial HTML state file."""
        initial_files = list(self.result_dir.glob("initial_*_raw.html"))
        return initial_files[0] if initial_files else None
    
    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data."""
        if not self.traj_file.exists():
            return []
        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def parse_html_for_tasks(self, html_path: Path) -> Dict:
        """Parse HTML to extract task information."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract user information from header
        user_match = re.search(r'<span class="user-name">([^<]+)</span>', html_content)
        current_user = user_match.group(1) if user_match else 'unknown'
        
        # Extract Due Today tasks using regex
        due_today_tasks = []
        
        # Find the Due Today section
        due_today_section_match = re.search(
            r'<h3 class="section-title">Due Today</h3>.*?<div class="today-tasks">(.*?)</div>.*?<a class="view-all-button"',
            html_content, re.DOTALL
        )
        
        if due_today_section_match:
            today_tasks_html = due_today_section_match.group(1)
            
            # Find all individual task items
            task_pattern = r'<div class="today-task">(.*?)</div>\s*</div>'
            tasks = re.finditer(task_pattern, today_tasks_html, re.DOTALL)
            
            for task_match in tasks:
                task_html = task_match.group(1)
                
                # Check if checkbox is checked
                checkbox_match = re.search(r'<input type="checkbox" class="task-checkbox"[^>]*>', task_html)
                is_checked = False
                if checkbox_match:
                    checkbox_tag = checkbox_match.group(0)
                    is_checked = 'checked' in checkbox_tag
                
                # Extract task title
                title_match = re.search(r'<a class="today-task-title"[^>]*>([^<]+)</a>', task_html)
                title = title_match.group(1) if title_match else ''
                
                # Extract task meta (priority and due date info)
                meta_match = re.search(r'<p class="today-task-meta">([^<]+)</p>', task_html)
                meta = meta_match.group(1) if meta_match else ''
                
                if title:
                    due_today_tasks.append({
                        'title': title,
                        'checked': is_checked,
                        'meta': meta
                    })
        
        return {
            'user': current_user,
            'due_today_tasks': due_today_tasks
        }
    
    def check_checkpoint_1(self) -> bool:
        """Check if agent identified that there are Due Today tasks for multiple users."""
        initial_html = self.find_initial_html()
        if not initial_html:
            self.issues.append("No initial HTML found")
            return False
        
        initial_data = self.parse_html_for_tasks(initial_html)
        initial_due_today = initial_data['due_today_tasks']
        
        self.details['initial_user'] = initial_data['user']
        self.details['initial_due_today_count'] = len(initial_due_today)
        self.details['initial_due_today_tasks'] = [t['title'] for t in initial_due_today]
        
        # For Query 13: "Mark all Due Today tasks for two users as completed"
        # Need to check that the agent understood there are tasks for multiple users
        # Initial state typically shows one user's tasks
        
        if len(initial_due_today) > 0:
            return True
        else:
            self.issues.append("No Due Today tasks found in initial state")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if agent switched users to access tasks for different users."""
        trajectory = self.get_trajectory()
        
        # Look for user switch actions in trajectory
        user_switches = []
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            # Check for user switch button clicks
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if 'user-switch' in selector.lower() or 'switch user' in reasoning:
                    user_switches.append(step['step'])
        
        self.details['user_switch_steps'] = user_switches
        self.details['user_switch_count'] = len(user_switches)
        
        # Need at least one user switch to access second user's tasks
        if len(user_switches) >= 1:
            return True
        else:
            self.issues.append("Agent did not switch users to access tasks for multiple users")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if all Due Today tasks for two users were marked as completed."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        final_data = self.parse_html_for_tasks(final_html)
        final_due_today = final_data['due_today_tasks']
        
        self.details['final_user'] = final_data['user']
        self.details['final_due_today_count'] = len(final_due_today)
        self.details['final_due_today_tasks'] = [
            {'title': t['title'], 'checked': t['checked']} for t in final_due_today
        ]
        
        # Count trajectory data to verify both users' tasks were handled
        trajectory = self.get_trajectory()
        checkbox_clicks = []
        
        for step in trajectory:
            action = step.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                reasoning = action.get('reasoning', '').lower()
                
                # Look for checkbox clicks in Due Today section
                if 'checkbox' in selector.lower() or 'task-checkbox' in selector.lower():
                    if 'due today' in reasoning or 'mark' in reasoning or 'complete' in reasoning:
                        checkbox_clicks.append({
                            'step': step['step'],
                            'reasoning': action.get('reasoning', '')
                        })
        
        self.details['checkbox_clicks'] = checkbox_clicks
        self.details['checkbox_click_count'] = len(checkbox_clicks)
        
        # Query requires marking tasks for TWO users
        # Initial state shows tasks for user 1 (e.g., sarah.johnson)
        # Need to verify agent clicked checkboxes for at least 2 different users
        
        # Count expected: Initial showed tasks for first user, need to switch to second user
        # Minimum expectation: At least 2 checkbox clicks (one per user minimum)
        
        if len(checkbox_clicks) >= 2:
            return True
        else:
            self.issues.append(f"Expected at least 2 checkbox clicks for two users, found {len(checkbox_clicks)}")
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_identified_users'] = self.check_checkpoint_1()
            self.checkpoints['cp2_switched_users'] = self.check_checkpoint_2()
            self.checkpoints['cp3_marked_all_completed'] = self.check_checkpoint_3()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 13,
                'query': 'Mark all Due Today tasks for two users as completed',
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
                'query_id': 13,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query13.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
