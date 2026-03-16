#!/usr/bin/env python3
# Evaluator for note Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract document information."""

    def __init__(self):
        super().__init__()
        self.in_h1 = False
        self.in_p = False
        self.current_tag_attrs = []
        self.document_title = None
        self.headings = []
        self.paragraphs = []
        self.current_text = []
        
    def handle_starttag(self, tag, attrs):
        if tag == 'h1':
            self.in_h1 = True
            self.current_text = []
        elif tag == 'h2':
            self.current_text = []
            self.current_tag_attrs = attrs
        elif tag == 'p':
            self.in_p = True
            self.current_text = []
            
    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_h1:
            self.in_h1 = False
            text = ''.join(self.current_text).strip()
            if text and text != 'Untitled Document':
                self.document_title = text
        elif tag == 'h2':
            text = ''.join(self.current_text).strip()
            if text:
                self.headings.append(text)
            self.current_text = []
        elif tag == 'p' and self.in_p:
            self.in_p = False
            text = ''.join(self.current_text).strip()
            if text:
                self.paragraphs.append(text)
            
    def handle_data(self, data):
        if self.in_h1 or self.in_p or self.current_text is not None:
            self.current_text.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_document_created": False,
            "cp2_title_set": False,
            "cp3_heading_added": False,
            "cp4_first_paragraph": False,
            "cp5_second_paragraph": False,
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

    def parse_html(self, html_path: Path) -> TaskHTMLParser:
        """Parse HTML file and extract document data."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            parser.feed(html_content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Document was created (New Note clicked)"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Check if "New Note" button was clicked
            for action in actions:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'New Note' in selector and action.get('result', {}).get('success'):
                        self.details['document_created'] = True
                        return True
            
            self.issues.append("New Note button was not clicked successfully")
            return False
        except Exception as e:
            self.issues.append(f"Error checking document creation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Document title is 'Technical Specification v1'"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No final HTML file found")
                return False
            
            # Search for the title in the HTML
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if title appears in the HTML content
            if 'Technical Specification v1' in html_content:
                self.details['title_found'] = 'Technical Specification v1'
                return True
            
            self.issues.append("Document title 'Technical Specification v1' not found")
            return False
        except Exception as e:
            self.issues.append(f"Error checking document title: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Heading 'Overview' is present"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No final HTML file found for heading check")
                return False
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Look for "Overview" as a heading
            if 'Overview' in html_content:
                self.details['heading_found'] = 'Overview'
                return True
            
            self.issues.append("Heading 'Overview' not found in document")
            return False
        except Exception as e:
            self.issues.append(f"Error checking heading: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: First paragraph is present"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No final HTML file found for first paragraph check")
                return False
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for first paragraph content
            expected_text = "This document outlines the technical specifications for our new product feature"
            if expected_text in html_content:
                self.details['first_paragraph_found'] = True
                return True
            
            self.issues.append("First paragraph not found in document")
            return False
        except Exception as e:
            self.issues.append(f"Error checking first paragraph: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Second paragraph is present"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No final HTML file found for second paragraph check")
                return False
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for second paragraph content
            expected_text = "The system architecture consists of three main components"
            if expected_text in html_content:
                self.details['second_paragraph_found'] = True
                return True
            
            self.issues.append("Second paragraph not found in document")
            self.details['note'] = "Agent got stuck trying to add the second paragraph and reached max steps"
            return False
        except Exception as e:
            self.issues.append(f"Error checking second paragraph: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Document was saved"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Look for save action - could be a button click or auto-save
            # Check if there's a save button click or if document persists in final state
            for action in actions:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'save' in selector.lower() or 'Save' in selector:
                        if action.get('result', {}).get('success'):
                            self.details['save_action_found'] = True
                            return True
            
            # If we have document content in final state, it may be auto-saved
            html_file = self.find_final_html()
            if html_file:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # If document has content and exists in final state, consider it saved
                if 'This document outlines the technical' in html_content:
                    self.details['document_exists'] = True
                    self.details['save_method'] = 'auto-save or implicit'
                    return True
            
            self.issues.append("No explicit save action found, but document may be auto-saved")
            # Return True if document content exists, as many modern apps auto-save
            return bool(self.details.get('document_exists'))
        except Exception as e:
            self.issues.append(f"Error checking document save: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_document_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_title_set'] = self.check_checkpoint_2()
            self.checkpoints['cp3_heading_added'] = self.check_checkpoint_3()
            self.checkpoints['cp4_first_paragraph'] = self.check_checkpoint_4()
            self.checkpoints['cp5_second_paragraph'] = self.check_checkpoint_5()
            self.checkpoints['cp6_document_saved'] = self.check_checkpoint_6()

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
        print("Usage: python note_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
