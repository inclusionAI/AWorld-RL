#!/usr/bin/env python3
# Evaluator for management Query 17

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
        self.in_task_card = False
        self.in_status_badge = False
        self.in_priority_badge = False
        self.in_task_title = False
        self.current_task = {}
        self.tasks = []
        self.current_tag = None
        self.current_attrs = []
        self.filter_buttons = []
        self.in_filter_button = False
        self.current_filter_button = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_attrs = attrs_dict
        
        # Check for current user
        if 'class' in attrs_dict and 'user-name' in attrs_dict['class']:
            self.in_user_name = True
        
        # Check for task cards
        if 'class' in attrs_dict and 'task-card' in attrs_dict['class']:
            self.in_task_card = True
            self.current_task = {'title': '', 'status': '', 'priority': '', 'due_date': ''}
        
        # Check for task title
        if self.in_task_card and 'class' in attrs_dict and 'task-title' in attrs_dict['class']:
            self.in_task_title = True
        
        # Check for status badge
        if self.in_task_card and 'class' in attrs_dict and 'task-status' in attrs_dict['class']:
            self.in_status_badge = True
        
        # Check for priority badge
        if self.in_task_card and 'class' in attrs_dict and 'priority-badge' in attrs_dict['class']:
            self.in_priority_badge = True
        
        # Check for filter buttons
        if 'class' in attrs_dict and 'filter-button' in attrs_dict['class']:
            self.in_filter_button = True
            self.current_filter_button = {
                'text': '',
                'active': 'active' in attrs_dict['class']
            }

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        if self.in_user_name:
            self.current_user = text
        
        if self.in_task_title:
            self.current_task['title'] = text
        
        if self.in_status_badge:
            self.current_task['status'] = text
        
        if self.in_priority_badge:
            self.current_task['priority'] = text
        
        if self.in_filter_button:
            self.current_filter_button['text'] = text

    def handle_endtag(self, tag):
        if tag == 'span' and self.in_user_name:
            self.in_user_name = False
        
        if tag == 'h3' and self.in_task_title:
            self.in_task_title = False
        
        if tag == 'span' and self.in_status_badge:
            self.in_status_badge = False
        
        if tag == 'span' and self.in_priority_badge:
            self.in_priority_badge = False
        
        if tag == 'div' and self.in_task_card:
            if self.current_task.get('title'):
                self.tasks.append(self.current_task)
            self.in_task_card = False
            self.current_task = {}
        
        if tag == 'button' and self.in_filter_button:
            self.filter_buttons.append(self.current_filter_button)
            self.in_filter_button = False
            self.current_filter_button = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_switched_to_john_doe": False,
            "cp2_filtered_in_progress_status": False,
            "cp3_filtered_medium_priority": False,
            "cp4_filtered_this_week_due_date": False,
            "cp5_changed_tasks_to_blocked": False,
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
        """Parse an HTML file and return the parser."""
        parser = TaskHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            parser.feed(html_content)
        return parser

    def check_user_in_html(self, html_file: Path) -> Optional[str]:
        """Extract current user from HTML file."""
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for user-name span
            match = re.search(r'<span class="user-name">([^<]+)</span>', content)
            if match:
                return match.group(1)
        return None

    def check_active_filters_in_html(self, html_file: Path) -> Dict[str, List[str]]:
        """Extract active filter buttons from HTML."""
        filters = {
            'status': [],
            'priority': [],
            'due_date': []
        }
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Find all filter buttons with 'active' class
            # Status filters
            status_match = re.findall(r'<button class="filter-button[^"]*active[^"]*">([^<]+)</button>', content)
            
            # More specific approach - find filter sections
            # Extract Status filter section
            status_section = re.search(r'<label class="filter-label">Status:</label>(.*?)</div>', content, re.DOTALL)
            if status_section:
                active_status = re.findall(r'<button class="filter-button[^"]*active[^"]*">([^<]+)</button>', status_section.group(1))
                filters['status'].extend(active_status)
            
            # Extract Priority filter section
            priority_section = re.search(r'<label class="filter-label">Priority:</label>(.*?)</div>', content, re.DOTALL)
            if priority_section:
                active_priority = re.findall(r'<button class="filter-button[^"]*active[^"]*">([^<]+)</button>', priority_section.group(1))
                filters['priority'].extend(active_priority)
            
            # Extract Due Date filter section
            due_date_section = re.search(r'<label class="filter-label">Due Date:</label>(.*?)</div>', content, re.DOTALL)
            if due_date_section:
                active_due = re.findall(r'<button class="filter-button[^"]*active[^"]*">([^<]+)</button>', due_date_section.group(1))
                filters['due_date'].extend(active_due)
        
        return filters

    def check_tasks_status_in_html(self, html_file: Path) -> List[Dict[str, str]]:
        """Extract tasks and their statuses from HTML."""
        tasks = []
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Find all task cards
            task_cards = re.findall(r'<div class="task-card[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', content, re.DOTALL)
            
            for card in task_cards:
                task = {'title': '', 'status': '', 'priority': ''}
                
                # Extract title
                title_match = re.search(r'<h3 class="task-title">([^<]+)</h3>', card)
                if title_match:
                    task['title'] = title_match.group(1).strip()
                
                # Extract status
                status_match = re.search(r'<span class="task-status[^"]*">([^<]+)</span>', card)
                if status_match:
                    task['status'] = status_match.group(1).strip()
                
                # Extract priority
                priority_match = re.search(r'<span class="priority-badge[^"]*">([^<]+)</span>', card)
                if priority_match:
                    task['priority'] = priority_match.group(1).strip()
                
                if task['title']:
                    tasks.append(task)
        
        return tasks

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Switched to john_doe user"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any step involved switching to john_doe
            for step in trajectory:
                action = step.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                
                # Check if there was a user switch action
                if 'switch' in reasoning and 'john' in reasoning:
                    # Check subsequent steps for john_doe user
                    step_num = step.get('step_num', 0)
                    step_files = self.find_step_html(f"step_{step_num + 1}_*")
                    
                    if step_files:
                        user = self.check_user_in_html(step_files[0])
                        if user and 'john' in user.lower():
                            self.details['switched_to_user'] = user
                            return True
            
            # Check final HTML for current user
            final_html = self.find_final_html()
            if final_html:
                user = self.check_user_in_html(final_html)
                self.details['final_user'] = user if user else 'unknown'
                if user and 'john' in user.lower():
                    return True
            
            self.issues.append("Failed to switch to john_doe user")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking user switch: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Filtered tasks by 'In Progress' status"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for filter actions in trajectory
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                reasoning = action.get('reasoning', '').lower()
                
                # Check if In Progress filter was clicked
                if 'selector' in params:
                    selector = params['selector'].lower()
                    if 'in progress' in selector or ('in progress' in reasoning and 'filter' in reasoning):
                        # Verify in subsequent steps
                        step_num = step.get('step_num', 0)
                        step_files = self.find_step_html(f"step_{step_num + 1}_*")
                        
                        if step_files:
                            filters = self.check_active_filters_in_html(step_files[0])
                            if 'In Progress' in filters.get('status', []):
                                self.details['in_progress_filtered'] = True
                                return True
            
            # Check final HTML
            final_html = self.find_final_html()
            if final_html:
                filters = self.check_active_filters_in_html(final_html)
                if 'In Progress' in filters.get('status', []):
                    self.details['in_progress_filtered'] = True
                    return True
            
            self.issues.append("Did not filter by 'In Progress' status")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking In Progress filter: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Filtered tasks by 'Medium' priority"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for Medium priority filter
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                reasoning = action.get('reasoning', '').lower()
                
                if 'selector' in params:
                    selector = params['selector'].lower()
                    if 'medium' in selector or ('medium' in reasoning and 'priority' in reasoning):
                        step_num = step.get('step_num', 0)
                        step_files = self.find_step_html(f"step_{step_num + 1}_*")
                        
                        if step_files:
                            filters = self.check_active_filters_in_html(step_files[0])
                            # Check for Medium or 🟡 Medium
                            for priority in filters.get('priority', []):
                                if 'medium' in priority.lower():
                                    self.details['medium_priority_filtered'] = True
                                    return True
            
            # Check final HTML
            final_html = self.find_final_html()
            if final_html:
                filters = self.check_active_filters_in_html(final_html)
                for priority in filters.get('priority', []):
                    if 'medium' in priority.lower():
                        self.details['medium_priority_filtered'] = True
                        return True
            
            self.issues.append("Did not filter by 'Medium' priority")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Medium priority filter: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Filtered tasks by 'This Week' due date"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for This Week filter
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                reasoning = action.get('reasoning', '').lower()
                
                if 'selector' in params:
                    selector = params['selector'].lower()
                    if 'this week' in selector or ('this week' in reasoning and ('due' in reasoning or 'date' in reasoning)):
                        step_num = step.get('step_num', 0)
                        step_files = self.find_step_html(f"step_{step_num + 1}_*")
                        
                        if step_files:
                            filters = self.check_active_filters_in_html(step_files[0])
                            # Check for This Week or 📆 This Week
                            for due_date in filters.get('due_date', []):
                                if 'this week' in due_date.lower():
                                    self.details['this_week_filtered'] = True
                                    return True
            
            # Check final HTML
            final_html = self.find_final_html()
            if final_html:
                filters = self.check_active_filters_in_html(final_html)
                for due_date in filters.get('due_date', []):
                    if 'this week' in due_date.lower():
                        self.details['this_week_filtered'] = True
                        return True
            
            self.issues.append("Did not filter by 'This Week' due date")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking This Week filter: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Changed status of filtered tasks to 'Blocked'"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for status change actions
            # This could be bulk status change or individual changes
            status_change_found = False
            
            for step in trajectory:
                action = step.get('action', {})
                params = action.get('parameters', {})
                reasoning = action.get('reasoning', '').lower()
                
                # Check for bulk status change or individual status changes to Blocked
                if 'selector' in params:
                    selector = params['selector'].lower()
                    if 'blocked' in selector or ('blocked' in reasoning and 'status' in reasoning):
                        status_change_found = True
                        break
            
            if status_change_found:
                self.details['status_change_attempted'] = True
                
                # Verify in final state or later steps
                # Since the task was to change status to Blocked, 
                # we need to check if any tasks have Blocked status
                final_html = self.find_final_html()
                if final_html:
                    tasks = self.check_tasks_status_in_html(final_html)
                    blocked_tasks = [t for t in tasks if 'blocked' in t.get('status', '').lower()]
                    
                    if blocked_tasks:
                        self.details['blocked_tasks_count'] = len(blocked_tasks)
                        return True
                
                # Also check intermediate steps for evidence of status change
                all_step_files = sorted(self.result_dir.glob("step_*_raw.html"))
                for step_file in reversed(all_step_files[-10:]):  # Check last 10 steps
                    tasks = self.check_tasks_status_in_html(step_file)
                    blocked_tasks = [t for t in tasks if 'blocked' in t.get('status', '').lower()]
                    if blocked_tasks:
                        self.details['blocked_tasks_count'] = len(blocked_tasks)
                        self.details['blocked_tasks_found_in'] = step_file.name
                        return True
            
            self.issues.append("Did not change task status to 'Blocked'")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking status change to Blocked: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_switched_to_john_doe'] = self.check_checkpoint_1()
            self.checkpoints['cp2_filtered_in_progress_status'] = self.check_checkpoint_2()
            self.checkpoints['cp3_filtered_medium_priority'] = self.check_checkpoint_3()
            self.checkpoints['cp4_filtered_this_week_due_date'] = self.check_checkpoint_4()
            self.checkpoints['cp5_changed_tasks_to_blocked'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 17,
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
                'query_id': 17,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query17.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
