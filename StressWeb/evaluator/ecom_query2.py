#!/usr/bin/env python3
# Evaluator for ecom Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class CartHTMLParser(HTMLParser):
    """Parse HTML to extract cart information."""

    def __init__(self):
        super().__init__()
        self.in_cart_badge = False
        self.cart_count = None
        self.product_name = None
        self.in_product_title = False
        self.current_tag_attrs = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag_attrs = attrs_dict
        
        if tag == 'span' and attrs_dict.get('class') == 'cart-badge':
            self.in_cart_badge = True
        elif tag == 'h1' and 'product-title' in attrs_dict.get('class', ''):
            self.in_product_title = True

    def handle_data(self, data):
        if self.in_cart_badge:
            self.cart_count = data.strip()
            self.in_cart_badge = False
        elif self.in_product_title:
            self.product_name = data.strip()
            self.in_product_title = False


class SearchResultsParser(HTMLParser):
    """Parse HTML to extract search results."""

    def __init__(self):
        super().__init__()
        self.in_search_results = False
        self.in_product_card = False
        self.in_product_title = False
        self.product_titles = []
        self.current_product_title = None
        self.search_query = None
        self.in_search_query = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'search-results' in attrs_dict.get('class', ''):
            self.in_search_results = True
        elif self.in_search_results and tag == 'a' and 'product-card' in attrs_dict.get('class', ''):
            self.in_product_card = True
        elif self.in_product_card and tag == 'h3':
            self.in_product_title = True
        elif tag == 'strong':
            self.in_search_query = True

    def handle_endtag(self, tag):
        if tag == 'a' and self.in_product_card:
            self.in_product_card = False
        elif tag == 'div' and self.in_search_results:
            self.in_search_results = False

    def handle_data(self, data):
        if self.in_product_title:
            self.current_product_title = data.strip()
            if self.current_product_title:
                self.product_titles.append(self.current_product_title)
            self.in_product_title = False
        elif self.in_search_query:
            self.search_query = data.strip()
            self.in_search_query = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_first_product_selected": False,
            "cp3_product_added_to_cart": False,
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
        """Checkpoint 1: Search for 'laptop' was executed."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if search was performed in trajectory
            search_performed = False
            for entry in trajectory:
                if 'page_info' in entry and 'url' in entry['page_info']:
                    url = entry['page_info']['url']
                    if 'search?q=laptop' in url.lower():
                        search_performed = True
                        self.details['search_url'] = url
                        break
            
            if not search_performed:
                self.issues.append("Search for 'laptop' was not executed")
                return False
            
            # Verify search results page exists
            search_result_files = self.find_step_html("step_3_*")
            if not search_result_files:
                self.issues.append("Search results page not found")
                return False
            
            # Parse search results to verify correct query
            with open(search_result_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
                parser = SearchResultsParser()
                parser.feed(html_content)
                
                if parser.search_query and parser.search_query.lower() == 'laptop':
                    self.details['search_query'] = parser.search_query
                    self.details['search_results_count'] = len(parser.product_titles)
                    self.details['search_results'] = parser.product_titles
                    return True
                else:
                    self.issues.append(f"Search query mismatch: expected 'laptop', found '{parser.search_query}'")
                    return False
                    
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: First product in search results was selected."""
        try:
            # Get search results
            search_result_files = self.find_step_html("step_3_*")
            if not search_result_files:
                self.issues.append("Search results page not found for verification")
                return False
            
            # Parse search results to get first product
            with open(search_result_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
                parser = SearchResultsParser()
                parser.feed(html_content)
                
                if not parser.product_titles:
                    self.issues.append("No products found in search results")
                    return False
                
                first_product = parser.product_titles[0]
                self.details['first_product_in_results'] = first_product
            
            # Check trajectory to verify product was navigated to
            trajectory = self.get_trajectory()
            product_page_visited = False
            
            for entry in trajectory:
                if 'page_info' in entry and 'url' in entry['page_info']:
                    url = entry['page_info']['url']
                    if '/product/' in url:
                        product_page_visited = True
                        self.details['product_page_url'] = url
                        break
            
            if not product_page_visited:
                self.issues.append("Product page was not visited")
                return False
            
            # Verify the correct product page was opened
            product_page_files = self.find_step_html("step_4_*")
            if not product_page_files:
                self.issues.append("Product detail page not found")
                return False
            
            with open(product_page_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
                parser = CartHTMLParser()
                parser.feed(html_content)
                
                if parser.product_name:
                    self.details['selected_product'] = parser.product_name
                    # Verify it matches the first product from search results
                    if parser.product_name == first_product:
                        return True
                    else:
                        self.issues.append(f"Wrong product selected: expected '{first_product}', got '{parser.product_name}'")
                        return False
                else:
                    self.issues.append("Could not extract product name from product page")
                    return False
                    
        except Exception as e:
            self.issues.append(f"Error checking product selection: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Product was added to shopping cart."""
        try:
            # Check final HTML state for cart count
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML state not found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
                parser = CartHTMLParser()
                parser.feed(html_content)
                
                if parser.cart_count is None:
                    self.issues.append("Cart badge not found in final state")
                    return False
                
                try:
                    cart_count = int(parser.cart_count)
                    self.details['final_cart_count'] = cart_count
                    
                    if cart_count >= 1:
                        return True
                    else:
                        self.issues.append(f"Cart is empty (count: {cart_count})")
                        return False
                        
                except ValueError:
                    self.issues.append(f"Invalid cart count: {parser.cart_count}")
                    return False
                    
        except Exception as e:
            self.issues.append(f"Error checking cart state: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_first_product_selected'] = self.check_checkpoint_2()
            self.checkpoints['cp3_product_added_to_cart'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 2,
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
                'query_id': 2,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ecom_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
