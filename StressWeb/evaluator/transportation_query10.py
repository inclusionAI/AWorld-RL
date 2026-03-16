#!/usr/bin/env python3
# Evaluator for transportation Query 10

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RideHistoryHTMLParser(HTMLParser):
    """Parse HTML to extract ride history dates and rating information."""

    def __init__(self):
        super().__init__()
        self.in_ride_date = False
        self.ride_dates = []
        self.current_text = ""
        self.star_ratings = []
        self.comments = []
        self.in_comment = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        # Look for date elements in ride history
        if tag in ['div', 'span', 'p']:
            class_name = attrs_dict.get('class', '')
            if 'date' in class_name.lower() or 'time' in class_name.lower():
                self.in_ride_date = True
        
        # Look for star ratings
        if tag == 'i' or tag == 'svg':
            class_name = attrs_dict.get('class', '')
            if 'star' in class_name.lower() or 'fa-star' in class_name.lower():
                self.star_ratings.append(attrs_dict)

    def handle_data(self, data):
        data = data.strip()
        if data:
            if self.in_ride_date:
                # Check if this looks like a date
                if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', data):
                    self.ride_dates.append(data)
                    self.in_ride_date = False
            
            # Look for rating comments
            if 'excellent' in data.lower() and 'experience' in data.lower():
                self.comments.append(data)

    def handle_endtag(self, tag):
        if tag in ['div', 'span', 'p']:
            self.in_ride_date = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for Query 10
        self.checkpoints = {
            "cp1_accessed_ride_history": False,
            "cp2_found_feb2_trip": False,
            "cp3_submitted_5_star_rating": False,
            "cp4_submitted_comment": False,
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

    def check_html_for_pattern(self, html_path: Path, pattern: str) -> bool:
        """Check if a pattern exists in an HTML file."""
        if not html_path.exists():
            return False
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return bool(re.search(pattern, content, re.IGNORECASE))
        except Exception:
            return False

    def find_february_2_trip_in_steps(self) -> bool:
        """Check if February 2, 2026 trip appears in any step."""
        step_files = self.find_step_html("step_*")
        for step_file in step_files:
            if self.check_html_for_pattern(step_file, r'(Feb(?:ruary)?\s*2,?\s*2026|2026-02-02)'):
                return True
        return False

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: User accessed Ride History."""
        trajectory = self.get_trajectory()
        
        # Check if user clicked on "Ride History" button/link
        for action in trajectory:
            if action.get('action', {}).get('action_type') == 'CLICK':
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                if 'ride history' in selector.lower():
                    self.details['ride_history_accessed'] = True
                    return True
        
        # Also check if any HTML file contains ride history page indicators
        step_files = self.find_step_html("step_*")
        for step_file in step_files[20:]:  # Check after step 20 (after navigation)
            if self.check_html_for_pattern(step_file, r'(Ride\s+History|Trip\s+History|Past\s+Rides)'):
                # Look for actual trip list indicators
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check if there are multiple date entries (indicating a list of rides)
                    dates_found = re.findall(r'(Jan|Feb|Mar)\s+\d{1,2},\s+2026', content)
                    if len(dates_found) >= 3:
                        self.details['ride_history_accessed'] = True
                        return True
        
        self.issues.append("User did not successfully access Ride History page")
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: User found the trip on February 2, 2026."""
        # Check if February 2, 2026 appears in any step HTML
        if self.find_february_2_trip_in_steps():
            self.details['feb2_trip_found'] = True
            return True
        
        # The agent scrolled extensively but never found Feb 2, 2026
        # This suggests the trip doesn't exist in the data
        self.issues.append("February 2, 2026 trip not found - may not exist in ride history")
        self.details['feb2_trip_found'] = False
        return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: User submitted a 5-star rating."""
        trajectory = self.get_trajectory()
        
        # Check for clicking on 5th star or rating action
        for action in trajectory:
            if action.get('action', {}).get('action_type') == 'CLICK':
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                # Look for 5-star rating selector patterns
                if 'star' in selector.lower() and ('5' in selector or 'fifth' in selector):
                    self.details['rating_submitted'] = True
                    return True
        
        # Check HTML for submitted ratings
        step_files = self.find_step_html("step_*")
        for step_file in step_files:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for filled stars or rating confirmation
                if re.search(r'(fa-star\s+filled|star.*filled|rating.*5|5.*star)', content, re.IGNORECASE):
                    # Verify it's associated with Feb 2 trip
                    if 'feb' in content.lower() or 'february' in content.lower():
                        self.details['rating_submitted'] = True
                        return True
        
        self.issues.append("5-star rating not submitted")
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: User submitted comment 'Excellent Experience!'."""
        trajectory = self.get_trajectory()
        
        # Check for typing the comment
        for action in trajectory:
            if action.get('action', {}).get('action_type') == 'TYPE':
                text = action.get('action', {}).get('parameters', {}).get('text', '')
                if 'excellent experience' in text.lower():
                    self.details['comment_submitted'] = True
                    return True
        
        # Check HTML for the comment
        step_files = self.find_step_html("step_*")
        for step_file in step_files:
            if self.check_html_for_pattern(step_file, r'Excellent\s+Experience!?'):
                self.details['comment_submitted'] = True
                return True
        
        self.issues.append("Comment 'Excellent Experience!' not submitted")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_accessed_ride_history'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_feb2_trip'] = self.check_checkpoint_2()
            
            # Only check rating and comment if the trip was found
            if self.checkpoints['cp2_found_feb2_trip']:
                self.checkpoints['cp3_submitted_5_star_rating'] = self.check_checkpoint_3()
                self.checkpoints['cp4_submitted_comment'] = self.check_checkpoint_4()
            else:
                self.issues.append("Cannot verify rating/comment submission - Feb 2 trip not found")

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python transportation_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
