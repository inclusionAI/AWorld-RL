#!/usr/bin/env python3
# Evaluator for note Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class DocumentHTMLParser(HTMLParser):
    """Parse HTML to extract document information."""

    def __init__(self):
        super().__init__()
        self.document_title = None
        self.workspace = None
        self.tags = []
        self.in_title_input = False
        self.in_workspace_select = False
        self.in_workspace_option = False
        self.in_tag_container = False
        self.in_tag_input = False
        self.current_option_selected = False
        self.tag_input_value = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for document title input
        if tag == 'input' and attrs_dict.get('class') == 'title-input':
            self.document_title = attrs_dict.get('value', '')
        
        # Check for workspace select
        if tag == 'select' and 'sidebar-select' in attrs_dict.get('class', ''):
            self.in_workspace_select = True
        
        # Check for workspace options
        if tag == 'option' and self.in_workspace_select:
            self.in_workspace_option = True
            self.current_option_selected = 'selected' in attrs_dict
        
        # Check for tags container
        if tag == 'div' and 'tags-container' in attrs_dict.get('class', ''):
            self.in_tag_container = True
        
        # Check for tag badge (showing added tags)
        if tag == 'span' and self.in_tag_container and 'tag-badge' in attrs_dict.get('class', ''):
            # Tag content will be in handle_data
            pass
        
        # Check for tag input field with value
        if tag == 'input' and 'tag-input' in attrs_dict.get('class', ''):
            self.tag_input_value = attrs_dict.get('value', '')

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        # Capture workspace name from selected option
        if self.in_workspace_option and self.current_option_selected:
            if data and data != "No Workspace":
                self.workspace = data
        
        # Capture tag from tag container
        if self.in_tag_container:
            if data and data not in ['', '×']:  # Exclude empty and close button
                self.tags.append(data)

    def handle_endtag(self, tag):
        if tag == 'select' and self.in_workspace_select:
            self.in_workspace_select = False
        if tag == 'option':
            self.in_workspace_option = False
            self.current_option_selected = False
        if tag == 'div' and self.in_tag_container:
            self.in_tag_container = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_document_created": False,
            "cp2_title_set": False,
            "cp3_tag_added": False,
            "cp4_workspace_assigned": False,
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

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def parse_html_for_document_info(self, html_file: Path) -> tuple:
        """Parse HTML to extract document title, workspace, and tags."""
        parser = DocumentHTMLParser()
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        
        return parser.document_title, parser.workspace, parser.tags, parser.tag_input_value

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Document was created (not on dashboard)"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if we're in editor page
        if 'class="editor-page"' in content or 'class="editor-container"' in content:
            self.details['document_created'] = True
            return True
        else:
            self.issues.append("Document editor not opened")
            self.details['document_created'] = False
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Document title is 'Project Alpha Notes'"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found for title check")
            return False
        
        title, workspace, tags, tag_input = self.parse_html_for_document_info(final_html)
        
        expected_title = "Project Alpha Notes"
        self.details['document_title'] = title
        
        if title and title.strip() == expected_title:
            return True
        else:
            self.issues.append(f"Document title is '{title}', expected '{expected_title}'")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Tag 'Personal' is added"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found for tag check")
            return False
        
        title, workspace, tags, tag_input = self.parse_html_for_document_info(final_html)
        
        self.details['tags_found'] = tags
        self.details['tag_input_value'] = tag_input
        
        # Check if 'Personal' tag is either in the tags container or in the input field
        # (if it's in the input field, it means it's been typed but not yet added with the + button)
        tag_in_container = 'Personal' in tags
        tag_in_input = tag_input and 'Personal' in tag_input
        
        if tag_in_container:
            self.details['personal_tag_status'] = 'added to document'
            return True
        elif tag_in_input:
            # Tag is typed but not added yet
            self.details['personal_tag_status'] = 'typed in input field but not added'
            self.issues.append("Tag 'Personal' is typed in input field but not added to document")
            return False
        else:
            self.issues.append(f"Tag 'Personal' not found. Tags found: {tags}")
            self.details['personal_tag_status'] = 'not found'
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Document is assigned to 'Personal Workspace'"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found for workspace check")
            return False
        
        title, workspace, tags, tag_input = self.parse_html_for_document_info(final_html)
        
        expected_workspace = "Personal Workspace"
        self.details['workspace'] = workspace
        
        # Check if workspace dropdown has the expected workspace selected
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for selected workspace option
        workspace_match = re.search(r'<option[^>]*selected[^>]*>([^<]+)</option>', content)
        if workspace_match:
            selected_workspace = workspace_match.group(1).strip()
            self.details['workspace_from_html'] = selected_workspace
            
            if selected_workspace == expected_workspace:
                return True
            else:
                self.issues.append(f"Document workspace is '{selected_workspace}', expected '{expected_workspace}'")
                return False
        
        # Fallback: check if workspace text appears in the workspace select area
        if workspace and workspace == expected_workspace:
            return True
        
        self.issues.append(f"Workspace '{expected_workspace}' not selected. Found: {workspace}")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_document_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_title_set'] = self.check_checkpoint_2()
            self.checkpoints['cp3_tag_added'] = self.check_checkpoint_3()
            self.checkpoints['cp4_workspace_assigned'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 6,
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
                'query_id': 6,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
