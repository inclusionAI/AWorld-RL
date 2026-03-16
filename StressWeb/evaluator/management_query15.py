#!/usr/bin/env python3
# Evaluator for management Query 15

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task information."""

    def __init__(self):
        super().__init__()
        self.task_titles = []
        self.in_task_title = False
        self.current_tag = None
        self.current_attrs = {}

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Check if this is a task title element
        if tag in ['h3', 'span', 'div']:
            for attr_name, attr_value in attrs:
                if attr_name == 'class' and attr_value and 'task' in attr_value.lower():
                    self.in_task_title = True
                    break

    def handle_data(self, data):
        # Capture task titles matching "Milestone X" pattern
        if data and 'Milestone' in data:
            match = re.search(r'Milestone\s+(\d+)', data.strip())
            if match:
                milestone_num = int(match.group(1))
                if milestone_num not in self.task_titles:
                    self.task_titles.append(milestone_num)

    def handle_endtag(self, tag):
        if tag == self.current_tag:
            self.in_task_title = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_bulk_create_used": False,
            "cp2_all_100_tasks_created": False,
            "cp3_correct_naming_pattern": False,
            "cp4_sequential_numbering": False,
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
        """Checkpoint 1: Verify bulk create was used"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if bulk create button was clicked
            for action in action_history:
                action_params = action.get('action', {}).get('parameters', {})
                selector = action_params.get('selector', '')
                if 'Bulk Create' in selector or 'bulk create' in selector.lower():
                    self.details['bulk_create_used'] = True
                    return True
            
            self.issues.append("Bulk Create feature was not used - inefficient approach")
            self.details['bulk_create_used'] = False
            return False
        except Exception as e:
            self.issues.append(f"Error checking bulk create usage: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Verify all 100 tasks were created"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML file not found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            milestone_numbers = parser.task_titles
            self.details['milestone_tasks_found'] = len(milestone_numbers)
            
            if len(milestone_numbers) >= 100:
                return True
            else:
                self.issues.append(f"Only {len(milestone_numbers)} tasks found, expected 100")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking task count: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Verify correct naming pattern (Milestone X)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Find all "Milestone X" patterns
            milestone_pattern = re.findall(r'Milestone\s+(\d+)', html_content)
            
            if not milestone_pattern:
                self.issues.append("No tasks with 'Milestone' naming pattern found")
                return False
            
            # Check if all found patterns are valid numbers
            milestone_numbers = []
            for num_str in milestone_pattern:
                try:
                    milestone_numbers.append(int(num_str))
                except ValueError:
                    continue
            
            if milestone_numbers:
                self.details['milestone_naming_correct'] = True
                self.details['unique_milestones'] = len(set(milestone_numbers))
                return True
            else:
                self.issues.append("Milestone naming pattern incorrect")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking naming pattern: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Verify sequential numbering from 1 to 100"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            milestone_numbers = sorted(set(parser.task_titles))
            
            # Check if we have all numbers from 1 to 100
            expected_range = list(range(1, 101))
            
            if milestone_numbers == expected_range:
                self.details['sequential_numbering'] = 'Complete (1-100)'
                return True
            else:
                missing = set(expected_range) - set(milestone_numbers)
                extra = set(milestone_numbers) - set(expected_range)
                
                if missing:
                    self.issues.append(f"Missing milestone numbers: {sorted(list(missing))[:10]}...")
                    self.details['missing_milestones'] = len(missing)
                if extra:
                    self.issues.append(f"Unexpected milestone numbers found: {sorted(list(extra))[:10]}...")
                    self.details['extra_milestones'] = len(extra)
                
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking sequential numbering: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_bulk_create_used'] = self.check_checkpoint_1()
            self.checkpoints['cp2_all_100_tasks_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_naming_pattern'] = self.check_checkpoint_3()
            self.checkpoints['cp4_sequential_numbering'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 15,
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
                'query_id': 15,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python management_query15.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
