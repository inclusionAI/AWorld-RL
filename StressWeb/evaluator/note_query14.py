#!/usr/bin/env python3
# Evaluator for note Query 14

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
        self.in_title_input = False
        self.in_editor_content = False
        self.document_title = ""
        self.editor_content = ""
        self.current_tag = None
        self.current_attrs = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_attrs = attrs_dict
        
        # Check for title input field
        if tag == "input" and attrs_dict.get("class") == "title-input":
            self.in_title_input = True
            self.document_title = attrs_dict.get("value", "")
        
        # Check for editor content div
        if tag == "div" and "editor-content" in attrs_dict.get("class", ""):
            self.in_editor_content = True

    def handle_data(self, data):
        if self.in_editor_content:
            self.editor_content += data.strip()

    def handle_endtag(self, tag):
        if tag == "div" and self.in_editor_content:
            self.in_editor_content = False
        self.current_tag = None


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_document_created": False,
            "cp2_title_correct": False,
            "cp3_content_added": False,
            "cp4_content_has_bold": False,
            "cp5_document_saved": False,
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
        """Parse an HTML file and return the parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Document created (editor page is active)"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No HTML file found")
            return False

        parser = self.parse_html_file(html_file)
        
        # Check if we're on the editor page by looking for editor-specific elements
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class="editor-page"' in content or 'class="editor-content"' in content:
                self.details['document_created'] = True
                return True
        
        self.issues.append("Editor page not found - document may not have been created")
        self.details['document_created'] = False
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Document title is 'Project Bold Notes'"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        parser = self.parse_html_file(html_file)
        
        if parser.document_title == "Project Bold Notes":
            self.details['title'] = parser.document_title
            return True
        
        self.issues.append(f"Title is '{parser.document_title}' instead of 'Project Bold Notes'")
        self.details['title'] = parser.document_title
        return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Content added with required text"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        parser = self.parse_html_file(html_file)
        
        required_text = "Initial project meeting scheduled for January 20, 2026"
        
        if required_text in parser.editor_content or required_text.replace(" ", "") in parser.editor_content.replace(" ", ""):
            self.details['content'] = parser.editor_content
            return True
        
        self.issues.append(f"Required text not found. Content: '{parser.editor_content}'")
        self.details['content'] = parser.editor_content
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: First line has bold formatting"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        # Check HTML for bold tags in editor content
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for bold formatting in editor content
            # Could be <strong>, <b>, or style="font-weight: bold"
            editor_content_match = re.search(r'<div class="editor-content"[^>]*>(.*?)</div>', content, re.DOTALL)
            if editor_content_match:
                editor_html = editor_content_match.group(1)
                
                # Check for bold tags
                if '<strong>' in editor_html or '<b>' in editor_html or 'font-weight: bold' in editor_html or 'font-weight:bold' in editor_html:
                    self.details['has_bold_formatting'] = True
                    return True
        
        self.issues.append("No bold formatting detected in the content")
        self.details['has_bold_formatting'] = False
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Document saved (Save button was clicked)"""
        trajectory = self.get_trajectory()
        
        # Check if any action clicked the Save button
        for action in trajectory:
            if action.get('action', {}).get('action_type') == 'CLICK':
                params = action.get('action', {}).get('parameters', {})
                selector = params.get('selector', '')
                
                # Look for save button clicks
                if 'Save' in selector or 'save' in selector.lower():
                    self.details['save_clicked'] = True
                    return True
        
        # Also check HTML for save status indicators
        html_file = self.find_final_html()
        if html_file:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for save status messages
                if 'Draft saved' in content or 'saved' in content.lower():
                    self.details['save_status_shown'] = True
                    # This is less definitive but shows auto-save happened
                    return True
        
        self.issues.append("No evidence of document being saved")
        self.details['save_clicked'] = False
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_document_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_title_correct'] = self.check_checkpoint_2()
            self.checkpoints['cp3_content_added'] = self.check_checkpoint_3()
            self.checkpoints['cp4_content_has_bold'] = self.check_checkpoint_4()
            self.checkpoints['cp5_document_saved'] = self.check_checkpoint_5()

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
        print("Usage: python note_query14.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
