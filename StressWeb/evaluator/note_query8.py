#!/usr/bin/env python3
# Evaluator for note Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class NoteHTMLParser(HTMLParser):
    """Parse HTML to extract note document information."""
    def __init__(self):
        super().__init__()
        self.title = None
        self.in_title_input = False
        self.shareable_links = []
        self.share_modal_open = False
        self.current_link_permission = None
        self.current_link_expiration = None
        self.save_button_present = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for title input
        if tag == 'input' and attrs_dict.get('class') == 'title-input':
            self.in_title_input = True
            self.title = attrs_dict.get('value', '')
        
        # Check for save button
        if tag == 'button':
            button_text = attrs_dict.get('class', '')
            if 'btn-primary' in button_text or 'Save' in str(attrs_dict):
                self.save_button_present = True
        
        # Check for share modal elements
        if 'share' in attrs_dict.get('class', '').lower():
            self.share_modal_open = True
        
        # Check for shareable link with permission
        if 'permission' in attrs_dict.get('class', '').lower():
            for attr_name, attr_value in attrs:
                if 'edit' in str(attr_value).lower():
                    self.current_link_permission = 'edit'
                elif 'view' in str(attr_value).lower():
                    self.current_link_permission = 'view'
        
        # Check for expiration info
        if 'expir' in attrs_dict.get('class', '').lower():
            for attr_name, attr_value in attrs:
                if '30' in str(attr_value):
                    self.current_link_expiration = '30-day'
    
    def handle_data(self, data):
        data_lower = data.lower().strip()
        
        # Check for share link text
        if 'edit' in data_lower and 'can' in data_lower:
            self.current_link_permission = 'edit'
        
        # Check for expiration text
        if '30' in data_lower and ('day' in data_lower or 'expir' in data_lower):
            self.current_link_expiration = '30-day'
        
        # Capture shareable link URL
        if data.startswith('http') and 'share' in data_lower:
            self.shareable_links.append({
                'url': data,
                'permission': self.current_link_permission,
                'expiration': self.current_link_expiration
            })


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints for Query 8:
        # Create and save a document titled 'shared', then create a shareable can edit link with 30-day expiration
        self.checkpoints = {
            "cp1_document_created": False,
            "cp2_document_titled_shared": False,
            "cp3_document_saved": False,
            "cp4_shareable_link_created": False,
            "cp5_link_has_edit_permission": False,
            "cp6_link_has_30day_expiration": False,
        }
        
        self.issues = []
        self.details = {}
    
    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        if step_files:
            return step_files[-1]
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        return final_files[0] if final_files else None
    
    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data."""
        if not self.traj_file.exists():
            return []
        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def check_checkpoint_1(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if document was created."""
        # Check if user clicked to create new document
        for action in trajectory:
            action_data = action.get('action', {})
            if action_data.get('action_type') == 'CLICK':
                params = action_data.get('parameters', {})
                selector = params.get('selector', '')
                if 'blank' in selector.lower() or 'new' in selector.lower():
                    self.details['document_created'] = True
                    return True
        
        # Check if editor is visible in final HTML
        if 'editor-page' in html_content or 'editor-container' in html_content:
            self.details['document_created'] = True
            return True
        
        self.issues.append("Document was not created")
        return False
    
    def check_checkpoint_2(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if document is titled 'shared'."""
        parser = NoteHTMLParser()
        parser.feed(html_content)
        
        # Check if title is exactly 'shared'
        if parser.title and parser.title.strip().lower() == 'shared':
            self.details['document_title'] = parser.title.strip()
            return True
        
        # Check trajectory for TYPE action with 'shared'
        for action in trajectory:
            action_data = action.get('action', {})
            if action_data.get('action_type') == 'TYPE':
                text = action_data.get('parameters', {}).get('text', '')
                if text.strip().lower() == 'shared':
                    self.details['typed_shared'] = True
                    # But verify it stuck in final HTML
                    if 'shared' in html_content.lower():
                        return True
        
        self.issues.append(f"Document title is not 'shared' (found: {parser.title})")
        self.details['document_title'] = parser.title
        return False
    
    def check_checkpoint_3(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if document was saved."""
        # Check for save button click in trajectory
        for action in trajectory:
            action_data = action.get('action', {})
            if action_data.get('action_type') == 'CLICK':
                params = action_data.get('parameters', {})
                selector = params.get('selector', '')
                if 'save' in selector.lower():
                    result = action.get('result', {})
                    if result.get('success'):
                        self.details['save_clicked'] = True
                        return True
        
        # Check HTML for saved state indicators
        if 'draft saved' in html_content.lower() or 'saved' in html_content.lower():
            self.details['saved_indicator'] = True
            return True
        
        self.issues.append("Document was not saved")
        return False
    
    def check_checkpoint_4(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if shareable link was created."""
        # Check trajectory for share-related actions
        share_action_found = False
        for action in trajectory:
            action_data = action.get('action', {})
            reasoning = action_data.get('reasoning', '').lower()
            params = action_data.get('parameters', {})
            selector = params.get('selector', '').lower()
            
            if 'share' in reasoning or 'share' in selector:
                share_action_found = True
                self.details['share_action_attempted'] = True
        
        # Check HTML for share modal or link
        if 'share' in html_content.lower() and ('link' in html_content.lower() or 'url' in html_content.lower()):
            if 'http' in html_content.lower():
                self.details['share_link_in_html'] = True
                return True
        
        if share_action_found:
            self.issues.append("Share action attempted but link not visible in final state")
        else:
            self.issues.append("No shareable link creation attempted")
        
        return False
    
    def check_checkpoint_5(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if shareable link has 'can edit' permission."""
        # Check for 'edit' permission in HTML
        html_lower = html_content.lower()
        if 'can edit' in html_lower or 'edit permission' in html_lower:
            self.details['edit_permission'] = True
            return True
        
        # Check trajectory for edit permission setting
        for action in trajectory:
            action_data = action.get('action', {})
            reasoning = action_data.get('reasoning', '').lower()
            if 'edit' in reasoning and ('permission' in reasoning or 'can' in reasoning):
                self.details['edit_permission_attempted'] = True
        
        self.issues.append("Shareable link does not have 'can edit' permission")
        return False
    
    def check_checkpoint_6(self, html_content: str, trajectory: List[Dict]) -> bool:
        """Check if shareable link has 30-day expiration."""
        html_lower = html_content.lower()
        
        # Check for 30-day expiration in HTML
        if '30' in html_content and ('day' in html_lower or 'expir' in html_lower):
            self.details['30day_expiration'] = True
            return True
        
        # Check trajectory for 30-day expiration setting
        for action in trajectory:
            action_data = action.get('action', {})
            reasoning = action_data.get('reasoning', '').lower()
            if '30' in reasoning and ('day' in reasoning or 'expir' in reasoning):
                self.details['30day_expiration_attempted'] = True
        
        self.issues.append("Shareable link does not have 30-day expiration")
        return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Find final HTML
            final_html_path = self.find_final_html()
            if not final_html_path:
                raise FileNotFoundError("No final HTML state found")
            
            with open(final_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Load trajectory
            trajectory = self.get_trajectory()
            
            # Run checkpoints
            self.checkpoints['cp1_document_created'] = self.check_checkpoint_1(html_content, trajectory)
            self.checkpoints['cp2_document_titled_shared'] = self.check_checkpoint_2(html_content, trajectory)
            self.checkpoints['cp3_document_saved'] = self.check_checkpoint_3(html_content, trajectory)
            self.checkpoints['cp4_shareable_link_created'] = self.check_checkpoint_4(html_content, trajectory)
            self.checkpoints['cp5_link_has_edit_permission'] = self.check_checkpoint_5(html_content, trajectory)
            self.checkpoints['cp6_link_has_30day_expiration'] = self.check_checkpoint_6(html_content, trajectory)
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 8,
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
                'query_id': 8,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query8.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
