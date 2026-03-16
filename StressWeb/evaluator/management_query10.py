#!/usr/bin/env python3
# Evaluator for management Query 10

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
        self.in_task_card = False
        self.in_task_title = False
        self.in_task_priority = False
        self.in_selection_count = False
        self.tasks = []
        self.current_task = {}
        self.selection_count = 0
        self.current_data = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict and 'task-card' in attrs_dict['class']:
            self.in_task_card = True
            self.current_task = {'selected': 'selected' in attrs_dict['class']}
            
        if self.in_task_card:
            if tag == 'a' and 'class' in attrs_dict and 'task-title' in attrs_dict['class']:
                self.in_task_title = True
                self.current_data = []
            elif tag == 'span' and 'class' in attrs_dict and 'task-priority' in attrs_dict['class']:
                self.in_task_priority = True
                self.current_data = []
        
        if tag == 'span' and 'class' in attrs_dict and 'selection-count' in attrs_dict['class']:
            self.in_selection_count = True
            self.current_data = []

    def handle_data(self, data):
        if self.in_task_title or self.in_task_priority or self.in_selection_count:
            self.current_data.append(data.strip())

    def handle_endtag(self, tag):
        if self.in_task_title and tag == 'a':
            self.current_task['title'] = ''.join(self.current_data).strip()
            self.in_task_title = False
            self.current_data = []
            
        if self.in_task_priority and tag == 'span':
            priority_text = ''.join(self.current_data).strip()
            self.current_task['priority'] = priority_text
            self.in_task_priority = False
            self.current_data = []
            
        if self.in_task_card and tag == 'div':
            if self.current_task and 'title' in self.current_task:
                self.tasks.append(self.current_task)
                self.current_task = {}
            self.in_task_card = False
            
        if self.in_selection_count and tag == 'span':
            count_text = ''.join(self.current_data).strip()
            match = re.search(r'(\d+)\s+tasks?\s+selected', count_text)
            if match:
                self.selection_count = int(match.group(1))
            self.in_selection_count = False
            self.current_data = []


class DashboardHTMLParser(HTMLParser):
    """Parse HTML to check if we're on the Dashboard page."""

    def __init__(self):
        super().__init__()
        self.on_dashboard = False
        self.in_title = False
        self.current_data = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'h1' and 'class' in attrs_dict and 'dashboard-title' in attrs_dict.get('class', ''):
            self.in_title = True
            self.current_data = []
        # Check for active navigation link
        if tag == 'a' and 'class' in attrs_dict:
            classes = attrs_dict.get('class', '')
            href = attrs_dict.get('href', '')
            if 'active' in classes and '/dashboard' in href:
                self.on_dashboard = True

    def handle_data(self, data):
        if self.in_title:
            self.current_data.append(data.strip())

    def handle_endtag(self, tag):
        if self.in_title and tag == 'h1':
            title_text = ''.join(self.current_data).strip()
            if 'Dashboard' in title_text or 'Overview' in title_text:
                self.on_dashboard = True
            self.in_title = False
            self.current_data = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_clicked_high_priority": False,
            "cp2_selected_first_three_tasks": False,
            "cp3_changed_priority_to_medium": False,
            "cp4_returned_to_dashboard": False,
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
        """Checkpoint 1: Agent clicked on 'High' priority from Dashboard"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Check if first action clicked on High priority item
            if len(actions) > 0:
                first_action = actions[0].get('action', {})
                selector = first_action.get('parameters', {}).get('selector', '')
                
                if 'High' in selector or 'high' in selector.lower():
                    if first_action.get('result', {}).get('success'):
                        self.details['cp1_action'] = 'Clicked High priority item'
                        return True
                    
            self.issues.append("Did not click on 'High' priority item from Dashboard as first step")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking cp1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Selected the first three tasks"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Look for steps around 16-18 where agent selected 3 tasks using coordinates
            selected_count = 0
            for i, action_entry in enumerate(actions):
                action = action_entry.get('action', {})
                reasoning = action.get('reasoning', '')
                
                # Check for coordinate clicks on task checkboxes
                if action.get('action_type') == 'CLICK':
                    params = action.get('parameters', {})
                    if 'x' in params and 'y' in params:
                        y_coord = params.get('y', 0)
                        # Check if y-coordinate suggests clicking task checkboxes (around 397, 508, 620)
                        if 350 < y_coord < 650 and i >= 15 and i <= 20:
                            if 'first' in reasoning.lower() or 'second' in reasoning.lower() or 'third' in reasoning.lower():
                                selected_count += 1
            
            # Also check intermediate HTML for "3 tasks selected"
            step_files = self.find_step_html("step_1[89]_*")
            step_files.extend(self.find_step_html("step_2[0-5]_*"))
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '3 tasks selected' in content:
                        self.details['cp2_selection'] = 'Found 3 tasks selected in HTML'
                        return True
            
            if selected_count >= 3:
                self.details['cp2_selection'] = f'Agent attempted to select 3 tasks via coordinates'
                return True
                
            self.issues.append("Did not successfully select exactly 3 tasks")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking cp2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Changed priority to Medium"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Look for actions related to changing priority to Medium
            found_medium_click = False
            found_apply_click = False
            
            for action_entry in actions:
                action = action_entry.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                selector = action.get('parameters', {}).get('selector', '').lower()
                
                # Check for clicking Medium option
                if 'medium' in reasoning or 'medium' in selector:
                    if action.get('action_type') == 'CLICK':
                        found_medium_click = True
                        
                # Check for applying the change
                if 'apply' in reasoning or 'apply' in selector:
                    found_apply_click = True
            
            # Check final HTML to see if tasks have Medium priority
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parser = TaskHTMLParser()
                    parser.feed(content)
                    
                    # Count tasks with Medium priority
                    medium_count = sum(1 for task in parser.tasks[:3] if 'Medium' in task.get('priority', ''))
                    
                    if medium_count >= 3:
                        self.details['cp3_priority_change'] = f'{medium_count} tasks have Medium priority'
                        return True
                    elif medium_count > 0:
                        self.details['cp3_partial'] = f'Only {medium_count}/3 tasks have Medium priority'
            
            # If we found attempts but no success in final state
            if found_medium_click:
                self.issues.append("Attempted to change priority to Medium but did not complete successfully")
            else:
                self.issues.append("Did not attempt to change priority to Medium")
                
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking cp3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Returned to Dashboard page"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Check if agent navigated to Dashboard
            for action_entry in actions:
                action = action_entry.get('action', {})
                if action.get('action_type') == 'NAVIGATE':
                    url = action.get('parameters', {}).get('url', '')
                    if '/dashboard' in url.lower():
                        self.details['cp4_navigation'] = 'Navigated to Dashboard'
                        return True
                        
                # Also check for clicking Dashboard link
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    reasoning = action.get('reasoning', '')
                    if 'dashboard' in selector.lower() or 'dashboard' in reasoning.lower():
                        self.details['cp4_click'] = 'Clicked Dashboard link'
                        return True
            
            # Check final HTML to see if on Dashboard
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parser = DashboardHTMLParser()
                    parser.feed(content)
                    
                    if parser.on_dashboard:
                        self.details['cp4_final_state'] = 'Final page is Dashboard'
                        return True
                    else:
                        self.issues.append("Final page is not Dashboard - still on Tasks page")
            
            self.issues.append("Did not return to Dashboard page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking cp4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_clicked_high_priority'] = self.check_checkpoint_1()
            self.checkpoints['cp2_selected_first_three_tasks'] = self.check_checkpoint_2()
            self.checkpoints['cp3_changed_priority_to_medium'] = self.check_checkpoint_3()
            self.checkpoints['cp4_returned_to_dashboard'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
