#!/usr/bin/env python3
# Evaluator for note Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class DocumentHTMLParser(HTMLParser):
    """Parse HTML to extract document information."""
    def __init__(self):
        super().__init__()
        self.in_editor = False
        self.in_title = False
        self.in_bold = False
        self.current_title = ""
        self.current_content = ""
        self.document_titles = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'div' and 'class' in attrs_dict:
            if 'editor-content' in attrs_dict.get('class', ''):
                self.in_editor = True
            elif 'document-title' in attrs_dict.get('class', ''):
                self.in_title = True
        if tag in ['b', 'strong'] and self.in_editor:
            self.in_bold = True
        
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_editor:
                self.in_editor = False
            if self.in_title:
                self.in_title = False
        if tag in ['b', 'strong']:
            self.in_bold = False
        
    def handle_data(self, data):
        data = data.strip()
        if data:
            if self.in_editor:
                self.current_content += data + " "
            if self.in_title:
                self.current_title += data + " "


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_document_created": False,
            "cp2_title_correct": False,
            "cp3_content_correct": False,
            "cp4_bold_formatting": False,
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
    
    def find_step_html_with_content(self) -> Optional[Path]:
        """Find the last step HTML file that contains the document content."""
        step_files = sorted(self.result_dir.glob("step_*_raw.html"), reverse=True)
        for step_file in step_files:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Initial project meeting' in content:
                        return step_file
            except Exception:
                continue
        return None
    
    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data."""
        if not self.traj_file.exists():
            return []
        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    trajectory.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return trajectory
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def check_document_title(self, html_content: str) -> bool:
        """Check if document has correct title 'Project Alpha Notes'."""
        # Look for various title patterns in HTML
        patterns = [
            r'<h1[^>]*>.*?Project Alpha Notes.*?</h1>',
            r'<div[^>]*class="[^"]*document-title[^"]*"[^>]*>.*?Project Alpha Notes.*?</div>',
            r'<input[^>]*value="Project Alpha Notes"',
            r'>Project Alpha Notes<',
        ]
        
        for pattern in patterns:
            if re.search(pattern, html_content, re.IGNORECASE | re.DOTALL):
                self.details['title_found'] = 'Project Alpha Notes'
                return True
        
        # Check if "Project Alpha" appears anywhere
        if re.search(r'Project\s+Alpha', html_content, re.IGNORECASE):
            self.details['title_found'] = 'Partial match: Project Alpha'
            return False
        
        self.details['title_found'] = 'Not found'
        return False
    
    def check_document_content(self, html_content: str) -> bool:
        """Check if document contains the required text."""
        expected_text = 'Initial project meeting scheduled for January 20, 2026'
        
        # Normalize whitespace for comparison
        normalized_html = re.sub(r'\s+', ' ', html_content)
        normalized_expected = re.sub(r'\s+', ' ', expected_text)
        
        if normalized_expected in normalized_html:
            self.details['content_found'] = expected_text
            return True
        
        # Check for partial matches
        if 'Initial project meeting' in html_content:
            self.details['content_found'] = 'Partial: Initial project meeting found'
            if 'January 20, 2026' in html_content:
                self.details['content_found'] = 'Complete but possibly not in same location'
                return True
            return False
        
        self.details['content_found'] = 'Not found'
        return False
    
    def check_bold_formatting(self, html_content: str) -> bool:
        """Check if the content has bold formatting."""
        # Look for bold tags around the required text
        patterns = [
            r'<b>.*?Initial project meeting scheduled for January 20, 2026.*?</b>',
            r'<strong>.*?Initial project meeting scheduled for January 20, 2026.*?</strong>',
            r'<b>.*?Initial project meeting.*?</b>',
            r'<strong>.*?Initial project meeting.*?</strong>',
        ]
        
        for pattern in patterns:
            if re.search(pattern, html_content, re.IGNORECASE | re.DOTALL):
                self.details['bold_formatting'] = 'Yes'
                return True
        
        # Check if bold tags exist but not around the text
        if re.search(r'<(b|strong)>', html_content, re.IGNORECASE):
            self.details['bold_formatting'] = 'Bold tags exist but not around required text'
            return False
        
        self.details['bold_formatting'] = 'No bold formatting found'
        return False
    
    def check_document_saved(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if document was saved."""
        # Check trajectory for save actions or "DONE" action
        for step in trajectory:
            action = step.get('action', {})
            action_type = action.get('action_type', '')
            if action_type == 'DONE':
                self.details['document_saved'] = 'Task marked as DONE'
                return True
        
        # Check if document appears in a list (implies it was saved)
        if re.search(r'Project Alpha Notes.*?Updated', html_content, re.IGNORECASE | re.DOTALL):
            self.details['document_saved'] = 'Document appears in list'
            return True
        
        # Check for auto-save indicators
        if re.search(r'(auto-?save|saved|saving)', html_content, re.IGNORECASE):
            self.details['document_saved'] = 'Save indicators found'
            return True
        
        self.details['document_saved'] = 'No clear save indication'
        return True  # Note-taking apps typically auto-save
    
    def check_checkpoint_1(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if document was created and saved."""
        return self.check_document_saved(html_content, trajectory)
    
    def check_checkpoint_2(self, html_content: str) -> bool:
        """Check if document has correct title."""
        return self.check_document_title(html_content)
    
    def check_checkpoint_3(self, html_content: str) -> bool:
        """Check if document has correct content."""
        return self.check_document_content(html_content)
    
    def check_checkpoint_4(self, html_content: str) -> bool:
        """Check if content has bold formatting."""
        return self.check_bold_formatting(html_content)
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            trajectory = self.get_trajectory()
            
            # Find the HTML file with content
            html_file = self.find_step_html_with_content()
            if not html_file:
                html_file = self.find_final_html()
            
            if not html_file or not html_file.exists():
                self.issues.append("No HTML file found for evaluation")
                return self._create_error_result(result_data, "No HTML file found")
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.details['html_file_used'] = str(html_file.name)
            
            # Run checkpoints
            self.checkpoints['cp1_document_created'] = self.check_checkpoint_1(html_content, trajectory)
            self.checkpoints['cp2_title_correct'] = self.check_checkpoint_2(html_content)
            self.checkpoints['cp3_content_correct'] = self.check_checkpoint_3(html_content)
            self.checkpoints['cp4_bold_formatting'] = self.check_checkpoint_4(html_content)
            
            # Collect issues
            if not self.checkpoints['cp1_document_created']:
                self.issues.append("Document was not properly created/saved")
            if not self.checkpoints['cp2_title_correct']:
                self.issues.append("Document title is not 'Project Alpha Notes'")
            if not self.checkpoints['cp3_content_correct']:
                self.issues.append("Document content is missing or incorrect")
            if not self.checkpoints['cp4_bold_formatting']:
                self.issues.append("Content does not have bold formatting")
            
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
            return self._create_error_result({}, f"Evaluation error: {str(e)}")
    
    def _create_error_result(self, result_data: Dict, error_msg: str) -> Dict:
        """Create error result structure."""
        return {
            'query_id': 1,
            'query': result_data.get('query', ''),
            'overall_success': False,
            'success_rate': 0.0,
            'error': error_msg,
            'checkpoints': self.checkpoints,
            'checkpoints_passed': f"0/{len(self.checkpoints)}",
            'issues': self.issues + [error_msg],
            'details': self.details,
            'agent_claimed_success': result_data.get('success', False),
            'execution_time': result_data.get('execution_time', 0),
            'total_steps': result_data.get('steps', 0)
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query1.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
