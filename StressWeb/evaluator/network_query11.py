#!/usr/bin/env python3
# Evaluator for network Query 11

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ExperienceHTMLParser(HTMLParser):
    """Parse HTML to extract work experience information."""

    def __init__(self):
        super().__init__()
        self.in_experience_section = False
        self.current_experience = {}
        self.experiences = []
        self.current_tag = None
        self.current_class = None
        self.data_buffer = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_class = attrs_dict.get('class', '')
        
        # Check if we're in experience item
        if 'experience-item' in self.current_class:
            self.in_experience_section = True
            self.current_experience = {}
        elif self.in_experience_section:
            # Track specific fields
            if 'experience-company' in self.current_class or 'company-name' in self.current_class:
                self.data_buffer = []
            elif 'experience-title' in self.current_class or 'job-title' in self.current_class:
                self.data_buffer = []
            elif 'experience-date' in self.current_class or 'date-range' in self.current_class or 'experience-duration' in self.current_class:
                self.data_buffer = []

    def handle_data(self, data):
        if self.in_experience_section and data.strip():
            self.data_buffer.append(data.strip())

    def handle_endtag(self, tag):
        if self.in_experience_section and self.data_buffer:
            data_text = ' '.join(self.data_buffer).strip()
            
            if self.current_class:
                if 'experience-company' in self.current_class or 'company-name' in self.current_class:
                    self.current_experience['company'] = data_text
                elif 'experience-title' in self.current_class or 'job-title' in self.current_class:
                    self.current_experience['title'] = data_text
                elif 'experience-date' in self.current_class or 'date-range' in self.current_class or 'experience-duration' in self.current_class:
                    self.current_experience['dates'] = data_text
            
            self.data_buffer = []
        
        # End of experience item
        if tag == 'div' and 'experience-item' in str(self.current_class):
            if self.current_experience:
                self.experiences.append(self.current_experience.copy())
            self.in_experience_section = False
            self.current_experience = {}
        
        self.current_class = None

    def get_experiences(self):
        return self.experiences


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for Query 11
        self.checkpoints = {
            "cp1_navigated_to_profile": False,
            "cp2_accessed_add_experience": False,
            "cp3_company_set_to_companyA": False,
            "cp4_job_title_set_to_intern": False,
            "cp5_dates_set_correctly": False,
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

    def parse_experiences_from_html(self, html_path: Path) -> List[Dict]:
        """Extract work experiences from HTML."""
        if not html_path.exists():
            return []
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Use parser
        parser = ExperienceHTMLParser()
        parser.feed(html_content)
        experiences = parser.get_experiences()
        
        # Also try regex patterns for different HTML structures
        # Pattern 1: Look for experience sections with company and title
        company_pattern = r'companyA'
        intern_pattern = r'intern'
        
        has_company = re.search(company_pattern, html_content, re.IGNORECASE)
        has_intern = re.search(intern_pattern, html_content, re.IGNORECASE)
        
        if has_company and has_intern:
            # Look for date patterns near companyA
            date_patterns = [
                r'June\s+2021.*?June\s+2022',
                r'Jun\s+2021.*?Jun\s+2022',
                r'06/2021.*?06/2022',
                r'2021-06.*?2022-06',
            ]
            
            for pattern in date_patterns:
                if re.search(pattern, html_content, re.IGNORECASE):
                    # Found a match with proper dates
                    experiences.append({
                        'company': 'companyA',
                        'title': 'intern',
                        'dates': 'June 2021 - June 2022'
                    })
                    break
        
        return experiences

    def check_trajectory_for_profile_navigation(self) -> bool:
        """Check if agent navigated to profile page."""
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            url = entry.get('url', '')
            if '/profile' in url and '/profile/add' not in url and '/profile/edit' not in url:
                return True
        
        return False

    def check_trajectory_for_add_experience(self) -> bool:
        """Check if agent accessed add work experience page."""
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            url = entry.get('url', '')
            if '/profile/add-work-experience' in url or '/add-work-experience' in url:
                return True
        
        return False

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Profile page"""
        navigated = self.check_trajectory_for_profile_navigation()
        
        if not navigated:
            self.issues.append("Agent did not navigate to the Profile page")
        else:
            self.details['profile_navigation'] = "Successfully navigated to profile page"
        
        return navigated

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Accessed Add Work Experience page"""
        accessed = self.check_trajectory_for_add_experience()
        
        if not accessed:
            self.issues.append("Agent did not access the Add Work Experience page")
        else:
            self.details['add_experience_access'] = "Successfully accessed add work experience page"
        
        return accessed

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Company set to 'companyA'"""
        final_html = self.find_final_html()
        
        if not final_html:
            self.issues.append("No final HTML found to verify company name")
            return False
        
        experiences = self.parse_experiences_from_html(final_html)
        
        # Check if any experience has companyA
        for exp in experiences:
            if 'company' in exp and exp['company'].lower() == 'companya':
                self.details['company_name'] = exp['company']
                return True
        
        # Also check in HTML content directly
        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        if 'companyA' in html_content:
            self.details['company_name'] = "companyA found in HTML"
            return True
        
        self.issues.append("Company 'companyA' not found in profile")
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Job title set to 'intern'"""
        final_html = self.find_final_html()
        
        if not final_html:
            self.issues.append("No final HTML found to verify job title")
            return False
        
        experiences = self.parse_experiences_from_html(final_html)
        
        # Check if any experience has intern as title
        for exp in experiences:
            if 'title' in exp and exp['title'].lower() == 'intern':
                self.details['job_title'] = exp['title']
                return True
        
        # Also check in HTML content directly
        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for intern near companyA
        if 'intern' in html_content.lower():
            self.details['job_title'] = "intern found in HTML"
            return True
        
        self.issues.append("Job title 'intern' not found in profile")
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Dates set correctly (June 2021 to June 2022)"""
        final_html = self.find_final_html()
        
        if not final_html:
            self.issues.append("No final HTML found to verify dates")
            return False
        
        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for various date formats
        date_patterns = [
            (r'June\s+2021.*?June\s+2022', 'June 2021 - June 2022'),
            (r'Jun\s+2021.*?Jun\s+2022', 'Jun 2021 - Jun 2022'),
            (r'06/2021.*?06/2022', '06/2021 - 06/2022'),
            (r'2021-06.*?2022-06', '2021-06 to 2022-06'),
        ]
        
        for pattern, display in date_patterns:
            if re.search(pattern, html_content, re.IGNORECASE | re.DOTALL):
                self.details['dates'] = display
                return True
        
        self.issues.append("Date range 'June 2021 to June 2022' not found in profile")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_profile'] = self.check_checkpoint_1()
            self.checkpoints['cp2_accessed_add_experience'] = self.check_checkpoint_2()
            self.checkpoints['cp3_company_set_to_companyA'] = self.check_checkpoint_3()
            self.checkpoints['cp4_job_title_set_to_intern'] = self.check_checkpoint_4()
            self.checkpoints['cp5_dates_set_correctly'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 11,
                'query': result_data.get('query', 'On the Profile page, add an experience with Company set to "companyA", Job Title set to "intern", and Date from June 2021 to June 2022.'),
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
                'query_id': 11,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python network_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
