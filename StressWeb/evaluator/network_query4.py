#!/usr/bin/env python3
# Evaluator for network Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class JobDetailHTMLParser(HTMLParser):
    """Parse HTML to extract job detail page information."""
    def __init__(self):
        super().__init__()
        self.in_job_detail = False
        self.current_tag = None
        self.job_title = None
        self.company_name = None
        self.job_description = []
        self.in_h1 = False
        self.in_h2 = False
        self.in_responsibilities = False
        self.in_requirements = False
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == 'h1':
            self.in_h1 = True
        elif tag == 'h2':
            self.in_h2 = True
        
    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_h1 = False
        elif tag == 'h2':
            self.in_h2 = False
            
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture job title from h1
        if self.in_h1 and not self.job_title:
            if 'Frontend Engineer' in data or 'Product Marketing Manager' in data:
                self.job_title = data
                self.in_job_detail = True
        
        # Capture detail sections
        if self.in_h2:
            if 'Responsibilities' in data or 'Requirements' in data or 'Benefits' in data:
                self.job_description.append(data)
        
        # If we found responsibilities or requirements text
        if self.in_job_detail and len(data) > 20:
            self.job_description.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_navigated_to_jobs_page": False,
            "cp2_viewed_frontend_engineer_details": False,
            "cp3_viewed_product_marketing_manager_details": False,
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
    
    def check_html_for_job_details(self, html_path: Path, job_title: str) -> bool:
        """Check if HTML contains job detail page for specific job."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for job detail indicators
            has_job_title = job_title in content
            has_responsibilities = 'Responsibilities' in content or 'responsibilities' in content
            has_requirements = 'Requirements' in content or 'requirements' in content
            has_back_link = 'Back to jobs' in content or 'back to jobs' in content
            
            # Job details page should have title + detail sections + back link
            if has_job_title and (has_responsibilities or has_requirements):
                return True
            
            return False
        except Exception as e:
            return False
    
    def check_checkpoint_1(self) -> bool:
        """Check if navigated to Jobs page."""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if first action was navigating to Jobs
            for action in action_history[:5]:  # Check first few actions
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Jobs' in selector or 'jobs' in selector:
                        self.details['jobs_navigation'] = f"Navigated to Jobs page at step {action.get('step')}"
                        return True
            
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if viewed Frontend Engineer details."""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Find steps where Frontend Engineer was clicked
            frontend_steps = []
            for action in action_history:
                if action.get('action', {}).get('action_type') in ['CLICK']:
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Frontend Engineer' in selector:
                        frontend_steps.append(action.get('step'))
            
            if not frontend_steps:
                self.issues.append("Frontend Engineer job was never clicked")
                return False
            
            # Check HTML files after these clicks for job details
            for step_num in frontend_steps:
                # Check next few steps for detail page
                for offset in range(1, 4):
                    check_step = step_num + offset
                    step_files = list(self.result_dir.glob(f"step_{check_step}_*_raw.html"))
                    if step_files:
                        if self.check_html_for_job_details(step_files[0], "Frontend Engineer"):
                            self.details['frontend_engineer_viewed'] = f"Viewed at step {check_step}"
                            return True
            
            self.issues.append("Frontend Engineer was clicked but details page not confirmed")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if viewed Product Marketing Manager details."""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Find steps where Product Marketing Manager was clicked
            pmm_steps = []
            for action in action_history:
                if action.get('action', {}).get('action_type') in ['CLICK']:
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Product Marketing Manager' in selector:
                        pmm_steps.append(action.get('step'))
            
            if not pmm_steps:
                self.issues.append("Product Marketing Manager job was never clicked")
                return False
            
            # Check HTML files after these clicks for job details
            for step_num in pmm_steps:
                # Check next few steps for detail page
                for offset in range(1, 4):
                    check_step = step_num + offset
                    step_files = list(self.result_dir.glob(f"step_{check_step}_*_raw.html"))
                    if step_files:
                        if self.check_html_for_job_details(step_files[0], "Product Marketing Manager"):
                            self.details['product_marketing_manager_viewed'] = f"Viewed at step {check_step}"
                            return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html and self.check_html_for_job_details(final_html, "Product Marketing Manager"):
                self.details['product_marketing_manager_viewed'] = "Viewed in final state"
                return True
            
            self.issues.append("Product Marketing Manager was clicked but details page not confirmed")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_navigated_to_jobs_page'] = self.check_checkpoint_1()
            self.checkpoints['cp2_viewed_frontend_engineer_details'] = self.check_checkpoint_2()
            self.checkpoints['cp3_viewed_product_marketing_manager_details'] = self.check_checkpoint_3()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 4,
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
                'query_id': 4,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python network_query4.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
