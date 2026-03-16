#!/usr/bin/env python3
# Evaluator for management Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task information."""

    def __init__(self):
        super().__init__()
        self.in_task_card = False
        self.in_task_priority = False
        self.urgent_task_count = 0
        self.current_priority = None
        self.filter_button_text = []
        self.in_filter_button = False
        self.active_filter = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for task cards
        if tag == 'div' and 'class' in attrs_dict and 'task-card' in attrs_dict['class']:
            self.in_task_card = True
            self.current_priority = None
        
        # Check for task priority span
        if tag == 'span' and 'class' in attrs_dict and 'task-priority' in attrs_dict['class']:
            self.in_task_priority = True
        
        # Check for filter buttons
        if tag == 'button' and 'class' in attrs_dict and 'filter-button' in attrs_dict['class']:
            self.in_filter_button = True
            if 'active' in attrs_dict['class']:
                # Mark this as potential active filter
                self.in_filter_button = 'active'

    def handle_data(self, data):
        text = data.strip()
        
        # Capture priority data
        if self.in_task_priority:
            if '🔴' in text or 'Urgent' in text:
                self.current_priority = 'urgent'
        
        # Capture filter button text
        if self.in_filter_button:
            if text:
                if self.in_filter_button == 'active' and 'Urgent' in text:
                    self.active_filter = 'urgent'
                self.filter_button_text.append(text)

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_task_card:
            # End of task card - count if urgent
            if self.current_priority == 'urgent':
                self.urgent_task_count += 1
            self.in_task_card = False
            self.current_priority = None
        
        if tag == 'span' and self.in_task_priority:
            self.in_task_priority = False
        
        if tag == 'button' and self.in_filter_button:
            self.in_filter_button = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_started_on_dashboard": False,
            "cp2_clicked_urgent_card": False,
            "cp3_navigated_to_urgent_tasks": False,
            "cp4_correct_url_parameter": False,
            "cp5_displays_urgent_tasks": False,
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
        """Checkpoint 1: Started on Dashboard page"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                self.issues.append("No trajectory data found")
                return False
            
            # Check first step URL
            first_step = trajectory[0]
            if 'page_info' in first_step and 'url' in first_step['page_info']:
                url = first_step['page_info']['url']
                if 'dashboard' in url.lower():
                    self.details['starting_page'] = 'Dashboard'
                    return True
                else:
                    self.issues.append(f"Did not start on dashboard page. Started on: {url}")
                    return False
            
            self.issues.append("Could not determine starting page from trajectory")
            return False
        except Exception as e:
            self.issues.append(f"Error checking starting page: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Clicked on Urgent priority card"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                return False
            
            # Look for click action with Urgent reference
            clicked_urgent = False
            for step in trajectory:
                if 'action' in step and step['action'].get('action_type') == 'CLICK':
                    # Check reasoning or parameters for "Urgent"
                    reasoning = step['action'].get('reasoning', '')
                    params = step['action'].get('parameters', {})
                    selector = params.get('selector', '')
                    
                    if 'urgent' in reasoning.lower() or 'urgent' in selector.lower():
                        clicked_urgent = True
                        self.details['urgent_click_step'] = step.get('step_num', 'unknown')
                        break
            
            if clicked_urgent:
                return True
            else:
                self.issues.append("No click action found targeting Urgent priority card")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking urgent click: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Successfully navigated to tasks page"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                return False
            
            # Check last step URL contains /tasks
            last_step = trajectory[-1]
            if 'page_info' in last_step and 'url' in last_step['page_info']:
                url = last_step['page_info']['url']
                if '/tasks' in url:
                    self.details['final_url'] = url
                    return True
                else:
                    self.issues.append(f"Did not navigate to tasks page. Final URL: {url}")
                    return False
            
            self.issues.append("Could not determine final page URL from trajectory")
            return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: URL contains correct priority parameter"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                return False
            
            # Check last step URL for priority=urgent parameter
            last_step = trajectory[-1]
            if 'page_info' in last_step and 'url' in last_step['page_info']:
                url = last_step['page_info']['url']
                if 'priority=urgent' in url.lower():
                    self.details['url_parameter'] = 'priority=urgent'
                    return True
                else:
                    self.issues.append(f"URL does not contain priority=urgent parameter. URL: {url}")
                    return False
            
            self.issues.append("Could not determine URL parameters from trajectory")
            return False
        except Exception as e:
            self.issues.append(f"Error checking URL parameter: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Final page displays urgent priority tasks"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            # Check if the urgent filter is active
            if parser.active_filter == 'urgent':
                self.details['active_filter'] = 'Urgent'
                self.details['urgent_tasks_displayed'] = parser.urgent_task_count
                
                # Verify at least some urgent tasks are displayed
                if parser.urgent_task_count > 0:
                    return True
                else:
                    self.issues.append("Urgent filter active but no urgent tasks displayed")
                    return False
            else:
                # Check if page at least shows urgent tasks
                if parser.urgent_task_count > 0:
                    self.details['urgent_tasks_displayed'] = parser.urgent_task_count
                    # Still pass if urgent tasks are shown
                    return True
                else:
                    self.issues.append("Final page does not display urgent priority tasks")
                    return False
                    
        except Exception as e:
            self.issues.append(f"Error parsing final HTML: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_started_on_dashboard'] = self.check_checkpoint_1()
            self.checkpoints['cp2_clicked_urgent_card'] = self.check_checkpoint_2()
            self.checkpoints['cp3_navigated_to_urgent_tasks'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_url_parameter'] = self.check_checkpoint_4()
            self.checkpoints['cp5_displays_urgent_tasks'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 2,
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
                'query_id': 2,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
