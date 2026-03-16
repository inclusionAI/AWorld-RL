#!/usr/bin/env python3
# Evaluator for network Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class PostHTMLParser(HTMLParser):
    """Parse HTML to extract posts from Test User."""

    def __init__(self):
        super().__init__()
        self.posts = []
        self.current_post = {}
        self.in_post_card = False
        self.in_post_header = False
        self.in_author_info = False
        self.in_post_content = False
        self.capture_author = False
        self.capture_content = False
        self.current_author = ""
        self.current_content = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'article' and attrs_dict.get('class') == 'post-card':
            self.in_post_card = True
            self.current_post = {}
            self.current_author = ""
            self.current_content = ""
            
        elif self.in_post_card and tag == 'div' and attrs_dict.get('class') == 'post-header':
            self.in_post_header = True
            
        elif self.in_post_header and tag == 'div' and attrs_dict.get('class') == 'post-author-info':
            self.in_author_info = True
            
        elif self.in_author_info and tag == 'h4':
            self.capture_author = True
            
        elif self.in_post_card and tag == 'div' and attrs_dict.get('class') == 'post-content':
            self.in_post_content = True
            
        elif self.in_post_content and tag == 'p':
            self.capture_content = True

    def handle_endtag(self, tag):
        if tag == 'article' and self.in_post_card:
            if self.current_author and self.current_content:
                self.posts.append({
                    'author': self.current_author.strip(),
                    'content': self.current_content.strip()
                })
            self.in_post_card = False
            self.in_post_header = False
            self.in_author_info = False
            self.in_post_content = False
            self.capture_author = False
            self.capture_content = False
            
        elif tag == 'div' and self.in_post_header:
            self.in_post_header = False
            
        elif tag == 'div' and self.in_author_info:
            self.in_author_info = False
            
        elif tag == 'h4' and self.capture_author:
            self.capture_author = False
            
        elif tag == 'div' and self.in_post_content:
            self.in_post_content = False
            
        elif tag == 'p' and self.capture_content:
            self.capture_content = False

    def handle_data(self, data):
        if self.capture_author:
            self.current_author += data
        elif self.capture_content:
            self.current_content += data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_post_hello_created": False,
            "cp2_post_world_created": False,
            "cp3_post_exclamation_created": False,
            "cp4_correct_sequence": False,
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

    def parse_posts_from_html(self, html_path: Path) -> List[Dict]:
        """Parse posts from HTML file."""
        if not html_path.exists():
            return []
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = PostHTMLParser()
        parser.feed(html_content)
        return parser.posts

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Post with 'Hello' content created by Test User"""
        html_path = self.find_final_html()
        if not html_path:
            self.issues.append("No final HTML file found")
            return False
        
        posts = self.parse_posts_from_html(html_path)
        test_user_posts = [p for p in posts if p['author'] == 'Test User']
        
        hello_posts = [p for p in test_user_posts if p['content'] == 'Hello']
        
        if not hello_posts:
            self.issues.append("Post with content 'Hello' not found")
            return False
        
        self.details['hello_post_found'] = True
        return True

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Post with 'world' content created by Test User"""
        html_path = self.find_final_html()
        if not html_path:
            self.issues.append("No final HTML file found")
            return False
        
        posts = self.parse_posts_from_html(html_path)
        test_user_posts = [p for p in posts if p['author'] == 'Test User']
        
        world_posts = [p for p in test_user_posts if p['content'] == 'world']
        
        if not world_posts:
            self.issues.append("Post with content 'world' not found")
            return False
        
        self.details['world_post_found'] = True
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Post with '!' content created by Test User"""
        html_path = self.find_final_html()
        if not html_path:
            self.issues.append("No final HTML file found")
            return False
        
        posts = self.parse_posts_from_html(html_path)
        test_user_posts = [p for p in posts if p['author'] == 'Test User']
        
        exclamation_posts = [p for p in test_user_posts if p['content'] == '!']
        
        if not exclamation_posts:
            self.issues.append("Post with content '!' not found")
            return False
        
        self.details['exclamation_post_found'] = True
        return True

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: All three posts created in correct sequence"""
        html_path = self.find_final_html()
        if not html_path:
            self.issues.append("No final HTML file found")
            return False
        
        posts = self.parse_posts_from_html(html_path)
        test_user_posts = [p for p in posts if p['author'] == 'Test User']
        
        # Extract content from test user posts
        test_user_contents = [p['content'] for p in test_user_posts]
        
        # Count occurrences of each expected content
        hello_count = test_user_contents.count('Hello')
        world_count = test_user_contents.count('world')
        exclamation_count = test_user_contents.count('!')
        
        self.details['test_user_post_count'] = len(test_user_posts)
        self.details['test_user_posts'] = test_user_contents
        
        # Check if all three posts exist
        if hello_count != 1 or world_count != 1 or exclamation_count != 1:
            self.issues.append(
                f"Expected exactly 1 of each post content, found: "
                f"Hello={hello_count}, world={world_count}, !={exclamation_count}"
            )
            return False
        
        # Posts are in reverse chronological order (most recent first)
        # So we should see: !, world, Hello in the feed
        expected_sequence = ['!', 'world', 'Hello']
        
        # Find the positions of our three posts
        try:
            exclamation_idx = test_user_contents.index('!')
            world_idx = test_user_contents.index('world')
            hello_idx = test_user_contents.index('Hello')
            
            # Check if they appear in the correct order (most recent first)
            if exclamation_idx < world_idx < hello_idx:
                self.details['sequence_correct'] = True
                return True
            else:
                self.issues.append(
                    f"Posts not in correct chronological order. "
                    f"Expected sequence (newest first): !, world, Hello. "
                    f"Found at indices: !={exclamation_idx}, world={world_idx}, Hello={hello_idx}"
                )
                return False
        except ValueError as e:
            self.issues.append(f"Error finding post sequence: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_post_hello_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_post_world_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_post_exclamation_created'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_sequence'] = self.check_checkpoint_4()

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
        print("Usage: python network_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
