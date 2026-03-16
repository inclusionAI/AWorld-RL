#!/usr/bin/env python3
# Evaluator for network Query 12

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class ProfileHTMLParser(HTMLParser):
    """Parse HTML to extract profile experience and skills information."""

    def __init__(self):
        super().__init__()
        self.experiences = []
        self.skills = []
        self.in_experience_item = False
        self.in_skill_item = False
        self.current_experience = {}
        self.current_tag = None
        self.current_class = None
        self.capture_company = False
        self.capture_title = False
        self.capture_dates = False
        self.capture_skill = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get('class', '')
        
        # Detect experience item
        if 'experience-item' in classes:
            self.in_experience_item = True
            self.current_experience = {}
        elif 'skill-item' in classes or 'skill-tag' in classes:
            self.in_skill_item = True
        
        # Capture experience details
        if self.in_experience_item:
            if 'experience-title' in classes or 'company-name' in classes:
                self.capture_company = True
            elif 'experience-subtitle' in classes or 'job-title' in classes:
                self.capture_title = True
            elif 'experience-date' in classes or 'date-range' in classes:
                self.capture_dates = True
        
        # Capture skill
        if self.in_skill_item:
            self.capture_skill = True
    
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.capture_company:
            self.current_experience['company'] = data
            self.capture_company = False
        elif self.capture_title:
            self.current_experience['title'] = data
            self.capture_title = False
        elif self.capture_dates:
            self.current_experience['dates'] = data
            self.capture_dates = False
        elif self.capture_skill:
            self.skills.append(data.lower())
            self.capture_skill = False
    
    def handle_endtag(self, tag):
        if self.in_experience_item and tag == 'div':
            if self.current_experience:
                self.experiences.append(self.current_experience)
                self.current_experience = {}
            self.in_experience_item = False
        elif self.in_skill_item and tag in ['span', 'div']:
            self.in_skill_item = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_profile": False,
            "cp2_added_experience_companyB": False,
            "cp3_experience_currently_working": False,
            "cp4_added_skill_python": False,
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
        """Checkpoint 1: Navigated to Profile page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if user navigated to profile page in trajectory
            for entry in trajectory:
                url = entry.get('url', '')
                if '/profile' in url.lower():
                    self.details['profile_page_reached'] = True
                    return True
            
            # Also check final HTML for profile page indicators
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for profile-specific elements
                    if 'experience-section' in content or 'skills-section' in content:
                        self.details['profile_page_reached'] = True
                        return True
            
            self.issues.append("Failed to navigate to Profile page")
            self.details['profile_page_reached'] = False
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking profile navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Added experience with companyB"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple text search for companyB
            if 'companyB' in content or 'companyb' in content.lower():
                self.details['companyB_found'] = True
                # Also verify it's in the context of experience
                if 'experience' in content.lower() or 'intern' in content.lower():
                    return True
            
            self.issues.append("Experience with companyB not found in profile")
            self.details['companyB_found'] = False
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking companyB experience: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Experience has "I currently work here" checked"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for indicators that "currently work here" is checked
            # This could be "Present", "Current", or similar text near companyB
            if 'companyB' in content or 'companyb' in content.lower():
                # Check for date patterns indicating current employment
                patterns = [
                    r'June 2021.*?Present',
                    r'June 2021.*?Current',
                    r'2021.*?Present',
                    r'currently.*?work',
                ]
                
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                        self.details['currently_working_indicator'] = True
                        return True
                
                # Also check if there's no end date specified
                if re.search(r'June 2021', content, re.IGNORECASE):
                    # If we find start date but no obvious end date pattern
                    if not re.search(r'June 2021.*?(?:June|July|August|September|October|November|December)\s+\d{4}', content, re.IGNORECASE):
                        self.details['currently_working_indicator'] = True
                        return True
            
            self.issues.append("'I currently work here' checkbox not properly set")
            self.details['currently_working_indicator'] = False
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking current work status: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Added skill 'python'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for python in skills section
            if 'python' in content.lower():
                # Verify it's in skills context
                if 'skill' in content.lower():
                    self.details['python_skill_found'] = True
                    return True
            
            self.issues.append("Skill 'python' not found in profile")
            self.details['python_skill_found'] = False
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking python skill: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_profile'] = self.check_checkpoint_1()
            self.checkpoints['cp2_added_experience_companyB'] = self.check_checkpoint_2()
            self.checkpoints['cp3_experience_currently_working'] = self.check_checkpoint_3()
            self.checkpoints['cp4_added_skill_python'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 12,
                'query': result_data.get('query', 'Query 12: Add experience and skill to profile'),
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
                'query_id': 12,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python network_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
