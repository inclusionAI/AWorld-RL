#!/usr/bin/env python3
# Evaluator for network Query 10

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_first_post_has_hello_comment": False,
            "cp2_second_post_has_world_comment": False,
            "cp3_third_post_has_exclamation_comment": False,
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
        """Checkpoint 1: First post has 'Hello' comment"""
        try:
            trajectory = self.get_trajectory()
            
            # Track which post IDs received which comments
            comments_by_post = {}
            
            for entry in trajectory:
                if 'action' in entry and entry.get('result', {}).get('success'):
                    action = entry['action']
                    # Look for TYPE actions with text
                    if action.get('action_type') == 'TYPE':
                        text = action.get('parameters', {}).get('text', '')
                        # Get the URL from page_info
                        url = entry.get('page_info', {}).get('url', '')
                        
                        # Extract post ID from URL
                        post_id_match = re.search(r'/posts/(\d+)', url)
                        if post_id_match and text:
                            post_id = post_id_match.group(1)
                            if post_id not in comments_by_post:
                                comments_by_post[post_id] = []
                            comments_by_post[post_id].append(text)
            
            self.details['comments_by_post'] = comments_by_post
            
            # The first post should have "Hello" comment
            # From trajectory, we see post 7234 was the first post clicked
            first_post_id = '7234'
            
            if first_post_id in comments_by_post:
                if 'Hello' in comments_by_post[first_post_id]:
                    return True
                else:
                    self.issues.append(f"First post (ID: {first_post_id}) does not have 'Hello' comment. Found: {comments_by_post[first_post_id]}")
            else:
                self.issues.append(f"No comments found for first post (ID: {first_post_id})")
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking first post comment: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Second post has 'world' comment"""
        try:
            trajectory = self.get_trajectory()
            
            # Track which post IDs received which comments in order
            comments_sequence = []
            
            for entry in trajectory:
                if 'action' in entry and entry.get('result', {}).get('success'):
                    action = entry['action']
                    # Look for TYPE actions with text
                    if action.get('action_type') == 'TYPE':
                        text = action.get('parameters', {}).get('text', '')
                        # Get the URL from page_info
                        url = entry.get('page_info', {}).get('url', '')
                        
                        # Extract post ID from URL
                        post_id_match = re.search(r'/posts/(\d+)', url)
                        if post_id_match and text:
                            post_id = post_id_match.group(1)
                            comments_sequence.append((post_id, text))
            
            self.details['comments_sequence'] = comments_sequence
            
            # The second post should have "world" comment
            # Based on the trajectory, we need to check if a DIFFERENT post got "world"
            # The agent added "Hello" to post 7234, then "world" to post 7234 again
            # This is incorrect - "world" should be on the second post
            
            # Count how many unique posts received comments
            unique_posts_with_comments = set([post_id for post_id, _ in comments_sequence])
            
            if len(unique_posts_with_comments) < 2:
                self.issues.append(f"Only {len(unique_posts_with_comments)} post(s) received comments. Expected at least 2 different posts.")
                return False
            
            # Check if second comment was on a different post
            if len(comments_sequence) >= 2:
                first_post = comments_sequence[0][0]
                second_post = comments_sequence[1][0]
                second_comment = comments_sequence[1][1]
                
                if second_post != first_post and second_comment == 'world':
                    return True
                elif second_post == first_post:
                    self.issues.append(f"Second comment 'world' was added to the same post as first comment (post ID: {second_post})")
                else:
                    self.issues.append(f"Second comment was not 'world'. Found: '{second_comment}'")
            else:
                self.issues.append(f"Less than 2 comments were posted")
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking second post comment: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Third post has '!' comment"""
        try:
            trajectory = self.get_trajectory()
            
            # Track which post IDs received which comments in order
            comments_sequence = []
            
            for entry in trajectory:
                if 'action' in entry and entry.get('result', {}).get('success'):
                    action = entry['action']
                    # Look for TYPE actions with text
                    if action.get('action_type') == 'TYPE':
                        text = action.get('parameters', {}).get('text', '')
                        # Get the URL from page_info
                        url = entry.get('page_info', {}).get('url', '')
                        
                        # Extract post ID from URL
                        post_id_match = re.search(r'/posts/(\d+)', url)
                        if post_id_match and text:
                            post_id = post_id_match.group(1)
                            comments_sequence.append((post_id, text))
            
            # Count how many unique posts received comments
            unique_posts_with_comments = set([post_id for post_id, _ in comments_sequence])
            
            if len(unique_posts_with_comments) < 3:
                self.issues.append(f"Only {len(unique_posts_with_comments)} post(s) received comments. Expected 3 different posts.")
                return False
            
            # Check if third comment was on a third different post
            if len(comments_sequence) >= 3:
                first_post = comments_sequence[0][0]
                second_post = comments_sequence[1][0]
                third_post = comments_sequence[2][0]
                third_comment = comments_sequence[2][1]
                
                if third_post != first_post and third_post != second_post and third_comment == '!':
                    return True
                elif third_post == first_post or third_post == second_post:
                    self.issues.append(f"Third comment '!' was added to a previously commented post (post ID: {third_post})")
                else:
                    self.issues.append(f"Third comment was not '!'. Found: '{third_comment}'")
            else:
                self.issues.append(f"Less than 3 comments were posted. Found {len(comments_sequence)} comments.")
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking third post comment: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_first_post_has_hello_comment'] = self.check_checkpoint_1()
            self.checkpoints['cp2_second_post_has_world_comment'] = self.check_checkpoint_2()
            self.checkpoints['cp3_third_post_has_exclamation_comment'] = self.check_checkpoint_3()

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
        print("Usage: python network_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
