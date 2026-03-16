#!/usr/bin/env python3
# Evaluator for network Query 13

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract post information and like states."""

    def __init__(self):
        super().__init__()
        self.posts = []
        self.current_post = None
        self.in_post_card = False
        self.in_post_header = False
        self.in_author_info = False
        self.in_author_name = False
        self.in_post_actions = False
        self.in_like_button = False
        self.capture_text = False
        self.current_button_classes = []
        self.current_icon_class = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get('class', '')

        if tag == 'article' and 'post-card' in classes:
            self.in_post_card = True
            self.current_post = {
                'author': '',
                'is_liked': False,
                'like_count': 0
            }

        elif self.in_post_card:
            if tag == 'div' and 'post-header' in classes:
                self.in_post_header = True
            elif self.in_post_header and tag == 'div' and 'post-author-info' in classes:
                self.in_author_info = True
            elif self.in_author_info and tag == 'h4':
                self.in_author_name = True
                self.capture_text = True
            elif tag == 'div' and 'post-actions' in classes:
                self.in_post_actions = True
            elif self.in_post_actions and tag == 'button' and 'action-button' in classes:
                self.in_like_button = True
                self.current_button_classes = classes.split()
            elif self.in_like_button and tag == 'i':
                self.current_icon_class = classes
            elif self.in_like_button and tag == 'span' and 'count' in classes:
                self.capture_text = True

    def handle_data(self, data):
        if self.capture_text:
            text = data.strip()
            if self.in_author_name:
                self.current_post['author'] = text
            elif self.in_like_button and text.isdigit():
                self.current_post['like_count'] = int(text)

    def handle_endtag(self, tag):
        if tag == 'article' and self.in_post_card:
            # Check if post is liked based on icon class
            if self.current_post:
                self.posts.append(self.current_post)
            self.in_post_card = False
            self.current_post = None
            self.in_post_header = False
            self.in_author_info = False
            self.in_post_actions = False

        elif tag == 'h4' and self.in_author_name:
            self.in_author_name = False
            self.capture_text = False

        elif tag == 'div':
            if self.in_post_actions:
                self.in_post_actions = False
            if self.in_author_info:
                self.in_author_info = False
            if self.in_post_header:
                self.in_post_header = False

        elif tag == 'button' and self.in_like_button:
            self.in_like_button = False
            self.current_button_classes = []
            self.current_icon_class = ""

        elif tag == 'span' and self.capture_text:
            self.capture_text = False


def parse_posts_simple(html_content: str) -> List[Dict]:
    """Simple regex-based parser for posts and their like status."""
    posts = []
    
    # Find all post-card articles
    post_pattern = r'<article class="post-card">(.*?)</article>'
    post_matches = re.findall(post_pattern, html_content, re.DOTALL)
    
    for post_html in post_matches:
        # Extract author name
        author_match = re.search(r'<h4>(.*?)</h4>', post_html)
        author = author_match.group(1) if author_match else "Unknown"
        
        # Check if liked - fas fa-thumbs-up indicates liked, far fa-thumbs-up indicates not liked
        is_liked = bool(re.search(r'<i class="fas fa-thumbs-up"', post_html))
        
        # Extract like count
        count_match = re.search(r'<span class="count">(\d+)</span>', post_html)
        like_count = int(count_match.group(1)) if count_match else 0
        
        posts.append({
            'author': author,
            'is_liked': is_liked,
            'like_count': like_count
        })
    
    return posts


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task: "Like the first seven posts"
        self.checkpoints = {
            "cp1_at_least_7_posts_available": False,
            "cp2_first_post_liked": False,
            "cp3_seven_posts_liked": False,
            "cp4_only_first_seven_liked": False,
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
        """Checkpoint 1: At least 7 posts are available (either initially or after loading more)"""
        try:
            final_html_path = self.find_final_html()
            if not final_html_path:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html_path, 'r', encoding='utf-8') as f:
                html = f.read()

            posts = parse_posts_simple(html)
            total_posts = len(posts)
            
            self.details['total_posts_in_final_state'] = total_posts

            # Check if agent tried to load more posts
            trajectory = self.get_trajectory()
            clicked_load_more = any(
                entry.get('action', {}).get('action_type') == 'CLICK' and
                'load' in str(entry.get('action', {}).get('parameters', {}).get('selector', '')).lower()
                for entry in trajectory
            )
            
            self.details['clicked_load_more'] = clicked_load_more

            if total_posts < 7:
                self.issues.append(f"Only {total_posts} posts available in final state, need at least 7")
                if not clicked_load_more:
                    self.issues.append("Agent did not click 'Load more posts' button to reveal additional posts")
                return False

            return True

        except Exception as e:
            self.issues.append(f"Error in checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: First post is liked"""
        try:
            final_html_path = self.find_final_html()
            if not final_html_path:
                return False

            with open(final_html_path, 'r', encoding='utf-8') as f:
                html = f.read()

            posts = parse_posts_simple(html)
            
            if not posts:
                self.issues.append("No posts found in final HTML")
                return False

            first_post = posts[0]
            self.details['first_post_author'] = first_post['author']
            self.details['first_post_liked'] = first_post['is_liked']

            if not first_post['is_liked']:
                self.issues.append(f"First post by {first_post['author']} is not liked")
                return False

            return True

        except Exception as e:
            self.issues.append(f"Error in checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Exactly 7 posts are liked"""
        try:
            final_html_path = self.find_final_html()
            if not final_html_path:
                return False

            with open(final_html_path, 'r', encoding='utf-8') as f:
                html = f.read()

            posts = parse_posts_simple(html)
            liked_posts = [p for p in posts if p['is_liked']]
            liked_count = len(liked_posts)
            
            self.details['total_liked_posts'] = liked_count
            self.details['liked_posts_authors'] = [p['author'] for p in liked_posts]

            if liked_count < 7:
                self.issues.append(f"Only {liked_count} posts are liked, expected 7")
                
                # Analyze if agent clicked same button multiple times
                trajectory = self.get_trajectory()
                like_clicks = [
                    i for i, entry in enumerate(trajectory)
                    if entry.get('action', {}).get('action_type') == 'CLICK' and
                    'Like' in str(entry.get('action', {}).get('parameters', {}).get('selector', ''))
                ]
                
                self.details['like_click_count'] = len(like_clicks)
                
                if len(like_clicks) >= 7 and liked_count < 7:
                    self.issues.append(
                        f"Agent clicked Like button {len(like_clicks)} times but only {liked_count} posts are liked. "
                        "Agent likely clicked the same Like button multiple times (toggling it on/off) instead of "
                        "clicking different posts' Like buttons"
                    )
                
                return False

            return True

        except Exception as e:
            self.issues.append(f"Error in checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Only the first 7 posts are liked (not random posts)"""
        try:
            final_html_path = self.find_final_html()
            if not final_html_path:
                return False

            with open(final_html_path, 'r', encoding='utf-8') as f:
                html = f.read()

            posts = parse_posts_simple(html)
            
            if len(posts) < 7:
                # If less than 7 posts total, this checkpoint is N/A
                return False

            # Check if exactly the first 7 posts are liked
            first_seven_liked = all(p['is_liked'] for p in posts[:7])
            remaining_not_liked = all(not p['is_liked'] for p in posts[7:])

            if not first_seven_liked:
                liked_indices = [i+1 for i, p in enumerate(posts[:7]) if p['is_liked']]
                not_liked_indices = [i+1 for i, p in enumerate(posts[:7]) if not p['is_liked']]
                self.issues.append(
                    f"Not all of the first 7 posts are liked. "
                    f"Liked positions in first 7: {liked_indices}, "
                    f"Not liked positions: {not_liked_indices}"
                )
                return False

            if not remaining_not_liked:
                extra_liked_indices = [i+1 for i, p in enumerate(posts[7:], start=7) if p['is_liked']]
                self.issues.append(f"Posts beyond position 7 are also liked: positions {extra_liked_indices}")
                return False

            return True

        except Exception as e:
            self.issues.append(f"Error in checkpoint 4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations in order
            self.checkpoints['cp1_at_least_7_posts_available'] = self.check_checkpoint_1()
            self.checkpoints['cp2_first_post_liked'] = self.check_checkpoint_2()
            self.checkpoints['cp3_seven_posts_liked'] = self.check_checkpoint_3()
            self.checkpoints['cp4_only_first_seven_liked'] = self.check_checkpoint_4()

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
        print("Usage: python network_query13.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
