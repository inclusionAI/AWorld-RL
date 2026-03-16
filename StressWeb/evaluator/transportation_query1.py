#!/usr/bin/env python3
# Evaluator for transportation Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.in_article = False
        self.article_title = None
        self.feedback_given = False
        self.current_text = []
        self.in_helpful_section = False

    def handle_starttag(self, tag, attrs):
        pass

    def handle_data(self, data):
        stripped = data.strip()
        if stripped:
            # Check if we found the article title
            if "Understanding fare estimates" in stripped:
                self.article_title = "Understanding fare estimates"
                self.in_article = True
            
            # Check if feedback was given
            if "Thank you for your feedback" in stripped:
                self.feedback_given = True
            
            # Check if we're in the helpful section
            if "Was this article helpful" in stripped:
                self.in_helpful_section = True


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_help": False,
            "cp2_selected_riders_category": False,
            "cp3_opened_fare_estimates_article": False,
            "cp4_marked_as_helpful": False,
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
        """Find intermediate step HTML files.

        Args:
            step_pattern: Pattern like "step_*", "step_10_*", "step_[12]*"
        Returns:
            List of matching HTML files sorted by step number
        """
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
        """Checkpoint 1: Navigated to Help Center"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if there was a click action on "Help"
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Help' in selector or 'help' in selector.lower():
                        self.details['help_clicked'] = True
                        return True
            
            self.issues.append("Did not navigate to Help Center")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Help navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Selected Riders category"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if there was a click action on "Riders"
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Riders' in selector or 'riders' in selector.lower():
                        self.details['riders_selected'] = True
                        return True
            
            self.issues.append("Did not select Riders category")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Riders selection: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Opened 'Understanding fare estimates' article"""
        try:
            # Check trajectory for article click
            trajectory = self.get_trajectory()
            article_clicked = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Understanding fare estimates' in selector:
                        article_clicked = True
                        break
            
            if not article_clicked:
                self.issues.append("Did not click on 'Understanding fare estimates' article")
                return False
            
            # Verify the article is shown in the final HTML
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            if parser.article_title:
                self.details['article_opened'] = True
                return True
            else:
                self.issues.append("Article 'Understanding fare estimates' not found in final state")
                return False
            
        except Exception as e:
            self.issues.append(f"Error checking article opening: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Marked article as helpful"""
        try:
            # Check trajectory for clicking Yes button
            trajectory = self.get_trajectory()
            yes_clicked = False
            
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Yes' in selector and ('helpful' in selector.lower() or 'button' in selector.lower()):
                        yes_clicked = True
                        break
            
            if not yes_clicked:
                self.issues.append("Did not click 'Yes' button to mark as helpful")
                return False
            
            # Verify feedback confirmation in final HTML
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            if parser.feedback_given:
                self.details['marked_helpful'] = True
                return True
            else:
                self.issues.append("Feedback confirmation 'Thank you for your feedback' not found in final state")
                return False
            
        except Exception as e:
            self.issues.append(f"Error checking helpful marking: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_help'] = self.check_checkpoint_1()
            self.checkpoints['cp2_selected_riders_category'] = self.check_checkpoint_2()
            self.checkpoints['cp3_opened_fare_estimates_article'] = self.check_checkpoint_3()
            self.checkpoints['cp4_marked_as_helpful'] = self.check_checkpoint_4()

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
            return {
                'query_id': 1,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python transportation_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
