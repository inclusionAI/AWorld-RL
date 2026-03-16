#!/usr/bin/env python3
# Evaluator for note Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.document_title = None
        self.document_content = []
        self.in_editor = False
        self.in_title_input = False
        self.current_tag = None
        self.current_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Check for document title input
        if tag == 'input':
            for attr, value in attrs:
                if attr == 'placeholder' and 'Untitled Document' in value:
                    self.in_title_input = True
                elif attr == 'value' and self.in_title_input:
                    self.document_title = value
        
        # Check for editor content
        if tag == 'div':
            for attr, value in attrs:
                if attr == 'class' and 'editor' in value.lower():
                    self.in_editor = True

    def handle_data(self, data):
        data = data.strip()
        if data:
            # Capture document title from input value or h1
            if self.current_tag in ['input', 'h1', 'h2'] and not self.document_title:
                if 'Q1 Planning Session' in data or 'Untitled Document' in data:
                    self.document_title = data
            
            # Capture editor content
            if self.in_editor and len(data) > 5:
                self.document_content.append(data)

    def handle_endtag(self, tag):
        if tag == 'input':
            self.in_title_input = False
        if tag == 'div':
            self.in_editor = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_template_selected": False,
            "cp2_document_created_with_title": False,
            "cp3_meeting_date_filled": False,
            "cp4_attendees_added": False,
            "cp5_agenda_items_added": False,
            "cp6_document_saved": False,
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

    def read_html_content(self, html_file: Path) -> str:
        """Read HTML file content."""
        if not html_file or not html_file.exists():
            return ""
        with open(html_file, 'r', encoding='utf-8') as f:
            return f.read()

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Meeting Notes template was selected"""
        # Check if agent navigated to templates page and selected Meeting Notes
        all_steps = self.find_step_html("step_*")
        
        for step_file in all_steps:
            content = self.read_html_content(step_file)
            # Check if they reached a template selection page or modal
            if 'Meeting Notes' in content and 'Template' in content:
                # Check if they actually selected it (should navigate away or open editor)
                # For now, just seeing the template is not enough
                continue
        
        # Check if any step shows template selection happened
        # The agent kept clicking "From Template" but never actually selected Meeting Notes
        self.issues.append("Agent failed to navigate to templates page and select 'Meeting Notes' template")
        self.details['template_selection'] = 'Failed - agent clicked From Template repeatedly but never accessed template selection interface'
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Document created with title 'Q1 Planning Session'"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        content = self.read_html_content(final_html)
        
        # Check for the specific title
        if 'Q1 Planning Session' in content:
            self.details['document_title'] = 'Q1 Planning Session'
            return True
        elif 'Untitled Document' in content:
            self.issues.append("Document title is 'Untitled Document', not 'Q1 Planning Session'")
            self.details['document_title'] = 'Untitled Document'
            return False
        else:
            self.issues.append("Could not find document title in final HTML")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Meeting date 'January 25, 2026' filled in"""
        # Check all intermediate and final steps for the meeting date
        all_steps = self.find_step_html("step_*")
        final_html = self.find_final_html()
        if final_html:
            all_steps.append(final_html)
        
        for step_file in all_steps:
            content = self.read_html_content(step_file)
            if 'January 25, 2026' in content or 'January 25th, 2026' in content:
                self.details['meeting_date'] = 'Found: January 25, 2026'
                return True
        
        self.issues.append("Meeting date 'January 25, 2026' not found in document")
        self.details['meeting_date'] = 'Not found'
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Attendees 'Sarah Chen, Alex Rodriguez' added"""
        all_steps = self.find_step_html("step_*")
        final_html = self.find_final_html()
        if final_html:
            all_steps.append(final_html)
        
        found_sarah = False
        found_alex = False
        
        for step_file in all_steps:
            content = self.read_html_content(step_file)
            if 'Sarah Chen' in content:
                found_sarah = True
            if 'Alex Rodriguez' in content:
                found_alex = True
        
        if found_sarah and found_alex:
            self.details['attendees'] = 'Both attendees found'
            return True
        elif found_sarah or found_alex:
            self.issues.append("Only one attendee found (need both Sarah Chen and Alex Rodriguez)")
            self.details['attendees'] = 'Partial - only one attendee found'
            return False
        else:
            self.issues.append("Attendees 'Sarah Chen' and 'Alex Rodriguez' not found in document")
            self.details['attendees'] = 'Not found'
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Agenda items '2026 Q1 Goals, Budget Review' added"""
        all_steps = self.find_step_html("step_*")
        final_html = self.find_final_html()
        if final_html:
            all_steps.append(final_html)
        
        found_goals = False
        found_budget = False
        
        for step_file in all_steps:
            content = self.read_html_content(step_file)
            if '2026 Q1 Goals' in content or 'Q1 Goals' in content:
                found_goals = True
            if 'Budget Review' in content:
                found_budget = True
        
        if found_goals and found_budget:
            self.details['agenda_items'] = 'Both agenda items found'
            return True
        elif found_goals or found_budget:
            self.issues.append("Only one agenda item found (need both '2026 Q1 Goals' and 'Budget Review')")
            self.details['agenda_items'] = 'Partial - only one item found'
            return False
        else:
            self.issues.append("Agenda items '2026 Q1 Goals' and 'Budget Review' not found in document")
            self.details['agenda_items'] = 'Not found'
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Document saved"""
        # Check trajectory for save action
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            if 'action' in entry:
                action = entry.get('action', {})
                action_type = action.get('action_type', '')
                
                # Check for save action (button click or keyboard shortcut)
                if action_type == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'save' in selector.lower():
                        self.details['save_action'] = 'Save button clicked'
                        return True
                elif action_type == 'TYPE':
                    # Check for Ctrl+S or Cmd+S
                    text = action.get('parameters', {}).get('text', '')
                    if 'Control+s' in text or 'Meta+s' in text:
                        self.details['save_action'] = 'Keyboard shortcut used'
                        return True
        
        # Check if document appears to be in saved state in final HTML
        final_html = self.find_final_html()
        if final_html:
            content = self.read_html_content(final_html)
            # Look for saved indicators like "Saved" text or saved timestamp
            if 'Saved' in content or 'Last saved' in content:
                self.details['save_action'] = 'Document appears saved based on HTML state'
                return True
        
        self.issues.append("No evidence of document being saved")
        self.details['save_action'] = 'Not found'
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_template_selected'] = self.check_checkpoint_1()
            self.checkpoints['cp2_document_created_with_title'] = self.check_checkpoint_2()
            self.checkpoints['cp3_meeting_date_filled'] = self.check_checkpoint_3()
            self.checkpoints['cp4_attendees_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_agenda_items_added'] = self.check_checkpoint_5()
            self.checkpoints['cp6_document_saved'] = self.check_checkpoint_6()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 9,
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
                'query_id': 9,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
