#!/usr/bin/env python3
# Evaluator for management Query 14

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
        self.in_task_item = False
        self.in_task_title = False
        self.in_task_desc = False
        self.in_task_priority = False
        self.in_task_due_date = False
        self.current_data = []
        self.in_detail_view = False
        self.detail_task = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get('class', '')
        
        # Check if we're in a task list item
        if 'task-item' in classes or 'task-row' in classes:
            self.in_task_item = True
            self.current_task = {}
        
        # Check for task title
        if self.in_task_item and ('task-title' in classes or 'task-name' in classes):
            self.in_task_title = True
            self.current_data = []
        
        # Check for task description
        if 'task-description' in classes or 'description' in classes:
            self.in_task_desc = True
            self.current_data = []
        
        # Check for priority badge
        if 'priority-badge' in classes or 'priority' in classes:
            self.in_task_priority = True
            self.current_data = []
        
        # Check for due date
        if 'due-date' in classes or 'task-due' in classes:
            self.in_task_due_date = True
            self.current_data = []
        
        # Check if we're in detail view
        if 'task-detail' in classes or 'detail-view' in classes:
            self.in_detail_view = True

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        if self.in_task_title:
            self.current_data.append(text)
        elif self.in_task_desc:
            self.current_data.append(text)
        elif self.in_task_priority:
            self.current_data.append(text)
        elif self.in_task_due_date:
            self.current_data.append(text)

    def handle_endtag(self, tag):
        if self.in_task_title:
            self.in_task_title = False
            if self.current_data:
                self.current_task['title'] = ' '.join(self.current_data).strip()
        
        if self.in_task_desc:
            self.in_task_desc = False
            if self.current_data:
                self.current_task['description'] = ' '.join(self.current_data).strip()
        
        if self.in_task_priority:
            self.in_task_priority = False
            if self.current_data:
                self.current_task['priority'] = ' '.join(self.current_data).strip()
        
        if self.in_task_due_date:
            self.in_task_due_date = False
            if self.current_data:
                self.current_task['due_date'] = ' '.join(self.current_data).strip()
        
        if self.in_task_item and tag in ['div', 'tr', 'li']:
            if self.current_task.get('title'):
                self.tasks.append(self.current_task.copy())
            self.in_task_item = False
            self.current_task = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_three_tasks_created": False,
            "cp2_prepare_meeting_edited": False,
            "cp3_title_changed_correctly": False,
            "cp4_description_added": False,
            "cp5_priority_urgent_and_due_date_set": False,
            "cp6_review_contract_deleted": False,
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

    def search_task_in_html(self, html_content: str, task_title: str) -> Optional[Dict]:
        """Search for a specific task in HTML content."""
        # Simple search for task title
        if task_title in html_content:
            # Try to extract task details around the title
            lines = html_content.split('\n')
            for i, line in enumerate(lines):
                if task_title in line:
                    # Check surrounding lines for details
                    context = '\n'.join(lines[max(0, i-10):min(len(lines), i+10)])
                    
                    task_info = {'title': task_title, 'found': True}
                    
                    # Look for priority
                    if 'Urgent' in context or 'urgent' in context:
                        task_info['priority'] = 'Urgent'
                    elif 'High' in context or 'high' in context:
                        task_info['priority'] = 'High'
                    elif 'Medium' in context or 'medium' in context:
                        task_info['priority'] = 'Medium'
                    elif 'Low' in context or 'low' in context:
                        task_info['priority'] = 'Low'
                    
                    # Look for due date (2026-02-05)
                    date_match = re.search(r'2026-02-0[0-9]', context)
                    if date_match:
                        task_info['due_date'] = date_match.group(0)
                    
                    # Look for description
                    if 'Including financial reports, progress updates, and future planning' in context:
                        task_info['description'] = 'Including financial reports, progress updates, and future planning'
                    
                    return task_info
        
        return None

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Three tasks created (Prepare meeting materials, Review contract documents, Update client feedback)"""
        try:
            # Check trajectory for task creation actions
            trajectory = self.get_trajectory()
            
            # Count tasks created from action history in result.json
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for creation of the three tasks
            tasks_created = set()
            
            for action in action_history:
                reasoning = action.get('action', {}).get('reasoning', '')
                if 'Prepare meeting materials' in reasoning or 'Prepare meeting materials' in str(action):
                    tasks_created.add('Prepare meeting materials')
                if 'Review contract documents' in reasoning or 'Review contract documents' in str(action):
                    tasks_created.add('Review contract documents')
                if 'Update client feedback' in reasoning or 'Update client feedback' in str(action):
                    tasks_created.add('Update client feedback')
            
            # Also check intermediate HTML files
            html_files = self.find_step_html("step_*")
            for html_file in html_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Prepare meeting materials' in content:
                        tasks_created.add('Prepare meeting materials')
                    if 'Review contract documents' in content:
                        tasks_created.add('Review contract documents')
                    if 'Update client feedback' in content:
                        tasks_created.add('Update client feedback')
            
            if len(tasks_created) >= 3:
                self.details['tasks_created'] = list(tasks_created)
                return True
            else:
                self.issues.append(f"Only {len(tasks_created)} tasks created out of 3 required")
                self.details['tasks_created'] = list(tasks_created)
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Opened details of 'Prepare meeting materials' and clicked Edit"""
        try:
            trajectory = self.get_trajectory()
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for actions related to opening task details or clicking edit
            found_open_details = False
            found_edit_click = False
            
            for action in action_history:
                reasoning = action.get('action', {}).get('reasoning', '').lower()
                action_type = action.get('action', {}).get('action_type', '')
                
                if 'prepare meeting materials' in reasoning or 'prepare q1 meeting materials' in reasoning:
                    if 'detail' in reasoning or 'open' in reasoning or 'click' in reasoning:
                        found_open_details = True
                    if 'edit' in reasoning:
                        found_edit_click = True
            
            # Check intermediate steps for edit forms
            html_files = self.find_step_html("step_*")
            for html_file in html_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for edit form or modal with Prepare meeting materials
                    if ('Prepare meeting materials' in content or 'Prepare Q1 meeting materials' in content):
                        if 'edit' in content.lower() or 'save' in content.lower():
                            found_edit_click = True
                            break
            
            if found_edit_click:
                self.details['edit_opened'] = True
                return True
            else:
                self.issues.append("Did not find evidence of editing 'Prepare meeting materials'")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Title changed to 'Prepare Q1 meeting materials'"""
        try:
            # Check final HTML and intermediate steps
            final_html = self.find_final_html()
            
            # Check all HTML files for the updated title
            html_files = self.find_step_html("step_*")
            if final_html:
                html_files.append(final_html)
            
            found_updated_title = False
            for html_file in html_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Prepare Q1 meeting materials' in content:
                        found_updated_title = True
                        self.details['title_updated'] = 'Prepare Q1 meeting materials'
                        break
            
            if found_updated_title:
                return True
            else:
                self.issues.append("Title was not changed to 'Prepare Q1 meeting materials'")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Description added 'Including financial reports, progress updates, and future planning'"""
        try:
            # Check intermediate HTML files for the description
            html_files = self.find_step_html("step_*")
            final_html = self.find_final_html()
            if final_html:
                html_files.append(final_html)
            
            found_description = False
            for html_file in html_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Including financial reports, progress updates, and future planning' in content:
                        found_description = True
                        self.details['description_added'] = True
                        break
                    # Also check for partial matches
                    if 'financial reports' in content and 'progress updates' in content and 'future planning' in content:
                        found_description = True
                        self.details['description_added'] = True
                        break
            
            if found_description:
                return True
            else:
                self.issues.append("Description was not added correctly")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Priority set to Urgent and due date to 2026-02-05"""
        try:
            # Check HTML files for priority and due date
            html_files = self.find_step_html("step_*")
            final_html = self.find_final_html()
            if final_html:
                html_files.append(final_html)
            
            found_urgent = False
            found_due_date = False
            
            for html_file in html_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for Prepare Q1 meeting materials with Urgent priority and due date
                    if 'Prepare Q1 meeting materials' in content or 'Prepare meeting materials' in content:
                        # Check within context
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'Prepare Q1 meeting materials' in line or 'Prepare meeting materials' in line:
                                # Check surrounding lines (50 lines context)
                                context = '\n'.join(lines[max(0, i-50):min(len(lines), i+50)])
                                if 'Urgent' in context or 'urgent' in context:
                                    found_urgent = True
                                if '2026-02-05' in context:
                                    found_due_date = True
                                
                                if found_urgent and found_due_date:
                                    break
                
                if found_urgent and found_due_date:
                    break
            
            self.details['priority_urgent'] = found_urgent
            self.details['due_date_2026_02_05'] = found_due_date
            
            if found_urgent and found_due_date:
                return True
            else:
                if not found_urgent:
                    self.issues.append("Priority was not set to Urgent")
                if not found_due_date:
                    self.issues.append("Due date was not set to 2026-02-05")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 5: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: 'Review contract documents' task deleted"""
        try:
            # Check final HTML to ensure Review contract documents is NOT present
            final_html = self.find_final_html()
            
            if not final_html:
                self.issues.append("No final HTML found to verify deletion")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Task should NOT be present in final HTML
            if 'Review contract documents' not in content:
                self.details['review_contract_deleted'] = True
                return True
            else:
                self.issues.append("'Review contract documents' task was not deleted - still present in final state")
                self.details['review_contract_deleted'] = False
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 6: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_three_tasks_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_prepare_meeting_edited'] = self.check_checkpoint_2()
            self.checkpoints['cp3_title_changed_correctly'] = self.check_checkpoint_3()
            self.checkpoints['cp4_description_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_priority_urgent_and_due_date_set'] = self.check_checkpoint_5()
            self.checkpoints['cp6_review_contract_deleted'] = self.check_checkpoint_6()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 14,
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
                'query_id': 14,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query14.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
