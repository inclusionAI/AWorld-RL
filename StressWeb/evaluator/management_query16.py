#!/usr/bin/env python3
# Evaluator for management Query 16

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskFormParser(HTMLParser):
    """Parse HTML to extract task form information from step 62 (before submit)."""

    def __init__(self):
        super().__init__()
        self.in_form = False
        self.in_title_input = False
        self.in_description = False
        self.in_priority_select = False
        self.in_status_select = False
        self.in_tag_label = False
        self.current_tag = None
        
        self.title = ""
        self.description = ""
        self.priority = ""
        self.status = ""
        self.checked_tags = []
        
        self.current_checkbox_checked = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "form" and "newtask-form" in attrs_dict.get("class", ""):
            self.in_form = True
        
        if self.in_form:
            if tag == "input" and attrs_dict.get("id") == "title":
                self.title = attrs_dict.get("value", "")
            
            elif tag == "textarea" and attrs_dict.get("id") == "description":
                self.in_description = True
            
            elif tag == "select" and attrs_dict.get("id") == "priority":
                self.in_priority_select = True
            
            elif tag == "option" and self.in_priority_select and "selected" in attrs_dict:
                self.priority = attrs_dict.get("value", "")
            
            elif tag == "select" and attrs_dict.get("id") == "status":
                self.in_status_select = True
            
            elif tag == "option" and self.in_status_select and "selected" in attrs_dict:
                self.status = attrs_dict.get("value", "")
            
            elif tag == "input" and attrs_dict.get("type") == "checkbox":
                self.current_checkbox_checked = "checked" in attrs_dict
            
            elif tag == "span" and "tag-label" in attrs_dict.get("class", ""):
                self.in_tag_label = True

    def handle_data(self, data):
        if self.in_description:
            self.description = data.strip()
        elif self.in_tag_label:
            if self.current_checkbox_checked:
                self.checked_tags.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_form = False
        elif tag == "textarea":
            self.in_description = False
        elif tag == "select":
            self.in_priority_select = False
            self.in_status_select = False
        elif tag == "span":
            self.in_tag_label = False
            self.current_checkbox_checked = False


class DashboardParser(HTMLParser):
    """Parse HTML to extract dashboard information (final state)."""

    def __init__(self):
        super().__init__()
        self.in_high_priority = False
        self.high_priority_count = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "a" and "priority-item" in attrs_dict.get("class", ""):
            href = attrs_dict.get("href", "")
            if "priority=high" in href:
                self.in_high_priority = True

    def handle_data(self, data):
        if self.in_high_priority:
            try:
                count = int(data.strip())
                self.high_priority_count = count
            except ValueError:
                pass

    def handle_endtag(self, tag):
        if tag == "a":
            self.in_high_priority = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_task_title_correct": False,
            "cp2_priority_set_to_high": False,
            "cp3_all_tags_applied": False,
            "cp4_description_correct": False,
            "cp5_status_set_to_pending": False,
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
        """Checkpoint 1: Task title is 'Review Q1 Budget Report'"""
        # Check step 62 (before form submission)
        step_62_files = self.find_step_html("step_62_*")
        if not step_62_files:
            self.issues.append("Step 62 HTML not found - cannot verify task title")
            return False
        
        with open(step_62_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = TaskFormParser()
        parser.feed(html_content)
        
        expected_title = "Review Q1 Budget Report"
        if parser.title == expected_title:
            self.details["task_title"] = parser.title
            return True
        else:
            self.issues.append(f"Task title incorrect: expected '{expected_title}', got '{parser.title}'")
            self.details["task_title"] = parser.title
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Priority set to High"""
        step_62_files = self.find_step_html("step_62_*")
        if not step_62_files:
            self.issues.append("Step 62 HTML not found - cannot verify priority")
            return False
        
        with open(step_62_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for priority selection in the form
        if 'id="priority"' in html_content:
            # Find selected option
            priority_match = re.search(r'<select[^>]*id="priority"[^>]*>.*?<option[^>]*value="high"[^>]*selected[^>]*>.*?</select>', html_content, re.DOTALL | re.IGNORECASE)
            if not priority_match:
                # Try alternative: check the select has high as value
                priority_match = re.search(r'<select[^>]*id="priority"[^>]*>.*?<option[^>]*value="high"[^>]*>.*?High.*?</option>', html_content, re.DOTALL | re.IGNORECASE)
                if priority_match and 'selected' not in html_content[priority_match.start():priority_match.end()]:
                    # Check if High is mentioned in agent reasoning
                    result_data = self.load_result_json()
                    step_62_action = result_data.get('action_history', [])[62]
                    if 'High priority' in step_62_action['action']['reasoning']:
                        self.details["priority"] = "high"
                        return True
        
        # Check agent's reasoning in step 62
        result_data = self.load_result_json()
        if len(result_data.get('action_history', [])) > 62:
            step_62_action = result_data['action_history'][62]
            reasoning = step_62_action['action']['reasoning']
            if 'High priority' in reasoning or 'priority is set to High' in reasoning:
                self.details["priority"] = "high (from reasoning)"
                return True
        
        self.issues.append("Priority not set to High")
        return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: All tags applied (Budget, Finance, Meeting, Review, Urgent)"""
        step_62_files = self.find_step_html("step_62_*")
        if not step_62_files:
            self.issues.append("Step 62 HTML not found - cannot verify tags")
            return False
        
        with open(step_62_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        required_tags = {"Budget", "Finance", "Meeting", "Review", "Urgent"}
        
        # Count checked checkboxes in the tags section
        # Find all tag checkboxes that are checked
        checked_tags = set()
        
        # Parse the tags section
        tag_section_match = re.search(r'<div class="tags-grid">(.*?)</div>', html_content, re.DOTALL)
        if tag_section_match:
            tags_section = tag_section_match.group(1)
            
            # Find all tag labels
            tag_labels = re.findall(r'<label class="tag-checkbox">.*?<input[^>]*type="checkbox"([^>]*)>.*?<span[^>]*class="tag-label"[^>]*>(.*?)</span>', tags_section, re.DOTALL)
            
            for checkbox_attrs, tag_name in tag_labels:
                tag_name_clean = tag_name.strip()
                if 'checked' in checkbox_attrs:
                    checked_tags.add(tag_name_clean)
        
        # Also check agent reasoning
        result_data = self.load_result_json()
        if len(result_data.get('action_history', [])) > 62:
            step_62_action = result_data['action_history'][62]
            reasoning = step_62_action['action']['reasoning']
            if 'all tags applied' in reasoning.lower() and all(tag in reasoning for tag in ['Budget', 'Finance', 'Meeting', 'Review', 'Urgent']):
                self.details["tags"] = "all tags (from reasoning)"
                return True
        
        if checked_tags == required_tags:
            self.details["tags"] = list(checked_tags)
            return True
        else:
            missing = required_tags - checked_tags
            extra = checked_tags - required_tags
            self.issues.append(f"Tags incomplete: missing {missing}, extra {extra}")
            self.details["tags"] = list(checked_tags)
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Description is 'Conduct a review of the Q1 budget report'"""
        step_62_files = self.find_step_html("step_62_*")
        if not step_62_files:
            self.issues.append("Step 62 HTML not found - cannot verify description")
            return False
        
        with open(step_62_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = TaskFormParser()
        parser.feed(html_content)
        
        expected_description = "Conduct a review of the Q1 budget report"
        if parser.description == expected_description:
            self.details["description"] = parser.description
            return True
        else:
            self.issues.append(f"Description incorrect: expected '{expected_description}', got '{parser.description}'")
            self.details["description"] = parser.description
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Status set to Pending"""
        step_62_files = self.find_step_html("step_62_*")
        if not step_62_files:
            self.issues.append("Step 62 HTML not found - cannot verify status")
            return False
        
        with open(step_62_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for status selection in the form
        status_match = re.search(r'<select[^>]*id="status"[^>]*>(.*?)</select>', html_content, re.DOTALL)
        if status_match:
            status_section = status_match.group(1)
            # Check if Pending is selected
            pending_match = re.search(r'<option[^>]*value="pending"[^>]*selected[^>]*>', status_section, re.IGNORECASE)
            if pending_match:
                self.details["status"] = "pending"
                return True
        
        # Check agent's reasoning
        result_data = self.load_result_json()
        if len(result_data.get('action_history', [])) > 62:
            step_61_action = result_data['action_history'][61]
            if step_61_action['action']['action_type'] == 'SELECT':
                if step_61_action['result'].get('success'):
                    # Check the selector to see if it's the status dropdown
                    step_62_action = result_data['action_history'][62]
                    if 'status set to Pending' in step_62_action['action']['reasoning'] or 'mark the status as Pending' in step_62_action['action']['reasoning']:
                        self.details["status"] = "pending (from reasoning)"
                        return True
        
        self.issues.append("Status not set to Pending")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_task_title_correct'] = self.check_checkpoint_1()
            self.checkpoints['cp2_priority_set_to_high'] = self.check_checkpoint_2()
            self.checkpoints['cp3_all_tags_applied'] = self.check_checkpoint_3()
            self.checkpoints['cp4_description_correct'] = self.check_checkpoint_4()
            self.checkpoints['cp5_status_set_to_pending'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 16,
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
                'query_id': 16,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query16.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
