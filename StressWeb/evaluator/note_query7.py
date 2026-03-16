#!/usr/bin/env python3
# Evaluator for note Query 7

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
        self.in_title_input = False
        self.in_editor_content = False
        self.document_title = ""
        self.editor_content = ""
        self.current_tag = ""
        self.attrs_dict = {}
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.attrs_dict = dict(attrs)
        
        # Check for title input field
        if tag == "input" and self.attrs_dict.get("class") == "title-input":
            self.in_title_input = True
            self.document_title = self.attrs_dict.get("value", "")
        
        # Check for editor content div
        if tag == "div" and "editor-content" in self.attrs_dict.get("class", ""):
            self.in_editor_content = True
        
    def handle_endtag(self, tag):
        if tag == "input" and self.in_title_input:
            self.in_title_input = False
        if tag == "div" and self.in_editor_content:
            self.in_editor_content = False
            
    def handle_data(self, data):
        if self.in_editor_content:
            self.editor_content += data.strip()


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_document_created": False,
            "cp2_title_correct": False,
            "cp3_bullet_points_added": False,
            "cp4_document_saved": False,
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
    
    def find_step_html(self, step_pattern: str) -> List[Path]:
        """Find intermediate step HTML files."""
        return sorted(self.result_dir.glob(f"{step_pattern}_raw.html"))
    
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
    
    def parse_html_for_document(self, html_path: Path) -> Tuple[str, str]:
        """Parse HTML to extract document title and content."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = DocumentHTMLParser()
        parser.feed(html_content)
        return parser.document_title, parser.editor_content
    
    def check_bullet_points_in_html(self, html_path: Path) -> Tuple[bool, bool, bool]:
        """Check if all three bullet points exist in HTML."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for each bullet point
        has_social_media = "Social media strategy" in html_content
        has_email = "Email campaign" in html_content
        has_influencer = "Influencer partnerships" in html_content
        
        return has_social_media, has_email, has_influencer
    
    def check_checkpoint_1(self) -> bool:
        """Check if document was created (blank note clicked and editor opened)."""
        try:
            trajectory = self.get_trajectory()
            
            # Look for "Blank Note" click action
            for action_data in trajectory:
                action = action_data.get('action', {})
                if action.get('action_type') == 'CLICK':
                    params = action.get('parameters', {})
                    selector = params.get('selector', '')
                    if 'Blank Note' in selector or 'New Note' in selector:
                        result = action_data.get('result', {})
                        if result.get('success'):
                            self.details['document_created'] = True
                            return True
            
            self.issues.append("Document creation action (Blank Note click) not found")
            return False
        except Exception as e:
            self.issues.append(f"Error checking document creation: {str(e)}")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if document title is 'Technical Specification v2'."""
        try:
            # Check in intermediate steps for the title
            step_files = self.find_step_html("step_*")
            
            expected_title = "Technical Specification v2"
            
            for step_file in step_files:
                title, _ = self.parse_html_for_document(step_file)
                if title == expected_title:
                    self.details['title_found'] = expected_title
                    self.details['title_found_in'] = step_file.name
                    return True
            
            # Check final state
            final_html = self.find_final_html()
            if final_html:
                title, _ = self.parse_html_for_document(final_html)
                if title == expected_title:
                    self.details['title_found'] = expected_title
                    self.details['title_found_in'] = 'final'
                    return True
                else:
                    self.details['actual_title'] = title
            
            self.issues.append(f"Title not set to 'Technical Specification v2'")
            return False
        except Exception as e:
            self.issues.append(f"Error checking document title: {str(e)}")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if all three bullet points were added to the document."""
        try:
            # Check step files for bullet points
            step_files = self.find_step_html("step_*")
            
            best_match = (False, False, False)
            best_step = None
            
            for step_file in step_files:
                has_social, has_email, has_influencer = self.check_bullet_points_in_html(step_file)
                current = (has_social, has_email, has_influencer)
                
                # Update if this is better than previous
                if sum(current) > sum(best_match):
                    best_match = current
                    best_step = step_file.name
                
                # If all three found, we're done
                if all(current):
                    break
            
            has_social, has_email, has_influencer = best_match
            
            self.details['bullet_points'] = {
                'social_media_strategy': has_social,
                'email_campaign': has_email,
                'influencer_partnerships': has_influencer,
                'found_in_step': best_step
            }
            
            if not has_social:
                self.issues.append("Missing bullet point: '1. Social media strategy'")
            if not has_email:
                self.issues.append("Missing bullet point: '2. Email campaign'")
            if not has_influencer:
                # Note: Original query has typo "3Influencer" but should be "3. Influencer"
                self.issues.append("Missing bullet point: '3. Influencer partnerships'")
            
            return all([has_social, has_email, has_influencer])
            
        except Exception as e:
            self.issues.append(f"Error checking bullet points: {str(e)}")
            return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if document was saved (auto-save or manual save)."""
        try:
            trajectory = self.get_trajectory()
            
            # Check for save button click or auto-save indication
            for action_data in trajectory:
                action = action_data.get('action', {})
                
                # Check for explicit save button click
                if action.get('action_type') == 'CLICK':
                    params = action.get('parameters', {})
                    selector = params.get('selector', '')
                    if 'Save' in selector or 'save' in selector.lower():
                        result = action_data.get('result', {})
                        if result.get('success'):
                            self.details['save_method'] = 'manual_save'
                            return True
            
            # Check if auto-save was active (look for auto-save in HTML)
            step_files = self.find_step_html("step_*")
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Auto-save: ON' in content or 'Draft saved' in content:
                        self.details['save_method'] = 'auto_save'
                        return True
            
            # Check final state for saved content
            final_html = self.find_final_html()
            if final_html:
                # Check if we're on Notebooks page (indicating save completed)
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'notebooks-page' in content or 'Notebooks</h1>' in content:
                        self.details['save_method'] = 'navigated_to_notebooks'
                        return True
            
            self.issues.append("No evidence of document being saved")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking save status: {str(e)}")
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_document_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_title_correct'] = self.check_checkpoint_2()
            self.checkpoints['cp3_bullet_points_added'] = self.check_checkpoint_3()
            self.checkpoints['cp4_document_saved'] = self.check_checkpoint_4()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 7,
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
                'query_id': 7,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query7.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
