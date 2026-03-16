#!/usr/bin/env python3
# Evaluator for file Query 13

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
        self.in_script = False
        self.script_content = []
        self.shared_items = []
        self.file_items = []
        self.folder_items = []
        
    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.in_script = True
            
    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False
            
    def handle_data(self, data):
        if self.in_script:
            self.script_content.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_finance_folder_shared": False,
            "cp2_jpg_files_deleted": False,
            "cp3_welcome_kickoff_searched": False,
            "cp4_work_folder_created": False,
            "cp5_today_subfolder_created": False,
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
        """Checkpoint 1: Finance folder shared with sara.johnson@example.com with View & Copy permissions"""
        try:
            # Check step 40 where sharing was completed
            step_40_files = self.find_step_html("step_40_*")
            if not step_40_files:
                self.issues.append("Could not find step 40 HTML file to verify sharing")
                return False
            
            with open(step_40_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if email address is present
            has_email = 'sara.johnson@example.com' in html_content
            
            # Check if View & Copy permission is present
            has_permission = 'View &amp; Copy' in html_content or 'View & Copy' in html_content
            
            if has_email and has_permission:
                self.details['finance_shared'] = 'Finance folder shared with sara.johnson@example.com with View & Copy permissions'
                return True
            else:
                if not has_email:
                    self.issues.append("Email address sara.johnson@example.com not found in share dialog")
                if not has_permission:
                    self.issues.append("View & Copy permission not selected")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking Finance folder sharing: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: All JPG files deleted under Personal Files/Photos"""
        try:
            # Check step 71 which is after JPG deletion (back to Photos folder)
            step_71_files = self.find_step_html("step_71_*")
            if not step_71_files:
                self.issues.append("Could not find step 71 HTML file to verify JPG deletion")
                return False
            
            with open(step_71_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if JPG files are still present in the Photos folder
            jpg_patterns = [
                'family_photo_2025.jpg',
                'welcome_team_photo.jpg',
                r'\.jpg["\']'
            ]
            
            jpg_found = False
            for pattern in jpg_patterns:
                if re.search(pattern, html_content):
                    jpg_found = True
                    break
            
            # Also check trajectory to see if deletions were attempted
            trajectory = self.get_trajectory()
            delete_actions = [a for a in trajectory if a.get('action', {}).get('action_type') == 'CLICK' 
                            and 'Delete' in str(a.get('action', {}).get('parameters', {}).get('selector', ''))]
            
            if jpg_found:
                self.issues.append("JPG files still present in Photos folder - deletion appears incomplete")
                self.details['jpg_deletion'] = f'JPG files not fully deleted. {len(delete_actions)} delete attempts made'
                return False
            else:
                self.details['jpg_deletion'] = f'JPG files appear to be deleted. {len(delete_actions)} delete actions performed'
                return True
                
        except Exception as e:
            self.issues.append(f"Error checking JPG file deletion: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Searched for welcome_kickoff.pptx and viewed its details"""
        try:
            # Check step 74 where search results are shown
            step_74_files = self.find_step_html("step_74_*")
            if not step_74_files:
                self.issues.append("Could not find step 74 HTML file to verify search")
                return False
            
            with open(step_74_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if welcome_kickoff.pptx is in search results
            if 'welcome_kickoff.pptx' in html_content:
                self.details['file_search'] = 'Successfully searched for welcome_kickoff.pptx'
                
                # Check trajectory for viewing details (clicking info button)
                trajectory = self.get_trajectory()
                info_clicks = [a for a in trajectory if a.get('step', -1) >= 74 and a.get('step', -1) <= 77
                             and a.get('action', {}).get('action_type') == 'CLICK']
                
                if len(info_clicks) > 0:
                    self.details['file_details_viewed'] = f'File details viewed (info button clicked {len(info_clicks)} times)'
                    return True
                else:
                    self.issues.append("File details not viewed - no info button click detected")
                    return False
            else:
                self.issues.append("welcome_kickoff.pptx not found in search results")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking file search: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Created folder named 'work' under Work Projects"""
        try:
            # Check step 82 where work folder was created
            step_82_files = self.find_step_html("step_82_*")
            if not step_82_files:
                self.issues.append("Could not find step 82 HTML file to verify work folder creation")
                return False
            
            with open(step_82_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if 'work' folder exists in the view
            if re.search(r'work', html_content, re.IGNORECASE):
                self.details['work_folder'] = 'work folder created under Work Projects'
                return True
            else:
                self.issues.append("work folder not found after creation")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking work folder creation: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Created subfolder 'today' and uploaded 100MB file"""
        try:
            # Check final HTML to see if agent successfully navigated into work folder
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Could not find final HTML file")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check trajectory to see if the agent attempted to navigate into work folder
            trajectory = self.get_trajectory()
            work_clicks = [a for a in trajectory if a.get('step', -1) >= 82 
                          and a.get('action', {}).get('action_type') == 'CLICK'
                          and 'work' in str(a.get('action', {}).get('parameters', {}).get('selector', '')).lower()]
            
            # The agent clicked on work folder many times but never successfully entered it
            if len(work_clicks) > 10:
                self.issues.append(f"Agent attempted to open work folder {len(work_clicks)} times but failed to navigate into it")
                self.details['today_folder'] = 'Failed to create today subfolder - could not navigate into work folder'
                return False
            
            # Check if 'today' folder exists
            if 'today' in html_content:
                self.details['today_folder'] = 'today subfolder created'
                
                # Check for 100MB file upload
                if '100' in html_content and ('MB' in html_content or 'mb' in html_content):
                    self.details['file_upload'] = '100MB file uploaded'
                    return True
                else:
                    self.issues.append("100MB file not uploaded to today folder")
                    return False
            else:
                self.issues.append("today subfolder not created")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking today subfolder and file upload: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_finance_folder_shared'] = self.check_checkpoint_1()
            self.checkpoints['cp2_jpg_files_deleted'] = self.check_checkpoint_2()
            self.checkpoints['cp3_welcome_kickoff_searched'] = self.check_checkpoint_3()
            self.checkpoints['cp4_work_folder_created'] = self.check_checkpoint_4()
            self.checkpoints['cp5_today_subfolder_created'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 13,
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
                'query_id': 13,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query13.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
