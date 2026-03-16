#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive batch evaluation script - Evaluate all batches in test_results
Generate detailed statistical reports, support grouping analysis by model and mode
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import traceback


class ComprehensiveBatchEvaluator:
    """Comprehensive Batch Evaluator"""

    def __init__(self, test_results_dir: str, evaluator_dir: str, reports_dir: str):
        self.test_results_dir = Path(test_results_dir)
        self.evaluator_dir = Path(evaluator_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Supported projects and modes
        self.projects = ['calendar', 'ecom', 'email', 'file', 'food',
                        'management', 'network', 'note', 'reservation', 'transportation']
        self.modes = ['clean', 'chaos', 'failed', 'perturbed', 'semanticE', 'semanticS', 'dom']

    def parse_batch_name(self, batch_name: str) -> Optional[Dict[str, str]]:
        """Parse batch name, extract project, model, mode information"""
        # Format: batch_{timestamp}_{pname}_{model}_{mode}
        pattern = r'batch_(\d{8}_\d{6})_(\w+)_(.+?)_(clean|chaos|failed|perturbed|semanticE|semanticS|dom)$'
        match = re.match(pattern, batch_name)

        if match:
            return {
                'timestamp': match.group(1),
                'project': match.group(2),
                'model': match.group(3),
                'mode': match.group(4)
            }
        return None

    def find_evaluator(self, project: str, query_num: int) -> Optional[Path]:
        """Find corresponding evaluator script"""
        evaluator_file = self.evaluator_dir / f"{project}_query{query_num}.py"
        return evaluator_file if evaluator_file.exists() else None

    def run_single_evaluator(self, evaluator_path: Path, query_dir: Path, timeout: int = 60) -> Dict:
        """Run single evaluator"""
        try:
            result = subprocess.run(
                ['python3', str(evaluator_path), str(query_dir)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0 and result.stderr:
                return {
                    'error': 'Evaluator execution error',
                    'stderr': result.stderr[:500]
                }

            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as e:
                return {
                    'error': 'JSON decode error',
                    'message': str(e),
                    'stdout': result.stdout[:500]
                }

        except subprocess.TimeoutExpired:
            return {'error': 'Timeout'}
        except Exception as e:
            return {
                'error': 'Exception',
                'message': str(e),
                'traceback': traceback.format_exc()[:500]
            }

    def evaluate_single_batch(self, batch_dir: Path) -> Dict:
        """Evaluate single batch"""
        batch_info = self.parse_batch_name(batch_dir.name)
        if not batch_info:
            return {
                'error': 'Invalid batch name format',
                'batch_name': batch_dir.name
            }

        # Read batch config
        config_file = batch_dir / "batch_config.json"
        config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except:
                pass

        # Find all query directories
        query_dirs = sorted(
            [d for d in batch_dir.iterdir() if d.is_dir() and re.match(r'^query_\d+$', d.name)],
            key=lambda d: int(d.name.split('_')[1])
        )

        batch_result = {
            'batch_name': batch_dir.name,
            'timestamp': batch_info['timestamp'],
            'project': batch_info['project'],
            'model': batch_info['model'],
            'mode': batch_info['mode'],
            'config': config,
            'total_queries': len(query_dirs),
            'evaluated_queries': 0,
            'queries': []
        }

        # Evaluate each query
        for query_dir in query_dirs:
            query_num = int(query_dir.name.split('_')[1])
            evaluator_path = self.find_evaluator(batch_info['project'], query_num)

            if not evaluator_path:
                batch_result['queries'].append({
                    'query_num': query_num,
                    'error': 'Evaluator not found'
                })
                continue

            # Run evaluator
            eval_result = self.run_single_evaluator(evaluator_path, query_dir)

            # Extract key information
            query_result = {
                'query_num': query_num,
                'overall_success': eval_result.get('overall_success', False),
                'success_rate': eval_result.get('success_rate', 0),
                'total_steps': eval_result.get('total_steps', 0),
                'execution_time': eval_result.get('execution_time', 0),
                'agent_claimed_success': eval_result.get('agent_claimed_success', False),
                'checkpoints': eval_result.get('checkpoints', {}),
                'checkpoints_passed': eval_result.get('checkpoints_passed', '0/0'),
            }

            # If there's an error, add error information
            if 'error' in eval_result:
                query_result['error'] = eval_result['error']
                query_result['error_details'] = eval_result.get('message', '')

            batch_result['queries'].append(query_result)
            batch_result['evaluated_queries'] += 1

        # Calculate batch statistics
        batch_result['statistics'] = self.calculate_batch_statistics(batch_result['queries'])

        return batch_result

    def calculate_batch_statistics(self, queries: List[Dict]) -> Dict:
        """Calculate batch statistics"""
        if not queries:
            return {}

        valid_queries = [q for q in queries if 'error' not in q or not q['error']]

        if not valid_queries:
            return {'error': 'No valid query results'}

        # Task success statistics
        task_success_count = sum(1 for q in valid_queries if q.get('overall_success', False))
        task_success_rate = task_success_count / len(valid_queries) if valid_queries else 0

        # Agent self-claimed success statistics
        agent_success_count = sum(1 for q in valid_queries if q.get('agent_claimed_success', False))
        agent_success_rate = agent_success_count / len(valid_queries) if valid_queries else 0

        # Checkpoint statistics
        total_checkpoints = 0
        passed_checkpoints = 0
        for q in valid_queries:
            checkpoints = q.get('checkpoints', {})
            for cp_name, cp_value in checkpoints.items():
                total_checkpoints += 1
                # Handle boolean or dictionary format
                if isinstance(cp_value, bool):
                    if cp_value:
                        passed_checkpoints += 1
                elif isinstance(cp_value, dict):
                    if cp_value.get('passed', False):
                        passed_checkpoints += 1

        checkpoint_pass_rate = passed_checkpoints / total_checkpoints if total_checkpoints > 0 else 0

        # Step statistics
        steps_list = [q.get('total_steps', 0) for q in valid_queries]
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0

        # Partial success statistics (at least one checkpoint passed but not all)
        partial_success_count = sum(
            1 for q in valid_queries
            if not q.get('overall_success', False) and q.get('success_rate', 0) > 0
        )

        return {
            'total_queries': len(queries),
            'valid_queries': len(valid_queries),
            'task_success_count': task_success_count,
            'task_success_rate': task_success_rate,
            'partial_success_count': partial_success_count,
            'failed_count': len(valid_queries) - task_success_count - partial_success_count,
            'agent_success_count': agent_success_count,
            'agent_success_rate': agent_success_rate,
            'total_checkpoints': total_checkpoints,
            'passed_checkpoints': passed_checkpoints,
            'checkpoint_pass_rate': checkpoint_pass_rate,
            'avg_steps': avg_steps,
            'min_steps': min(steps_list) if steps_list else 0,
            'max_steps': max(steps_list) if steps_list else 0,
        }

    def evaluate_all_batches(self, filter_project: Optional[str] = None,
                           filter_model: Optional[str] = None,
                           filter_mode: Optional[str] = None,
                           filter_models: Optional[List[str]] = None) -> Dict:
        """Evaluate all batches"""
        # Find all batch directories
        all_batch_dirs = sorted(
            [d for d in self.test_results_dir.iterdir()
             if d.is_dir() and d.name.startswith('batch_')],
            key=lambda d: d.name
        )

        # Apply filters
        batch_dirs = []
        for batch_dir in all_batch_dirs:
            batch_info = self.parse_batch_name(batch_dir.name)
            if not batch_info:
                continue

            if filter_project and batch_info['project'] != filter_project:
                continue
            if filter_model and batch_info['model'] != filter_model:
                continue
            # Support model list filtering
            if filter_models and batch_info['model'] not in filter_models:
                continue
            if filter_mode and batch_info['mode'] != filter_mode:
                continue

            batch_dirs.append(batch_dir)

        print(f"\n{'='*80}")
        print(f"🚀 Starting Comprehensive Evaluation")
        print(f"{'='*80}")
        print(f"Test results directory: {self.test_results_dir}")
        print(f"Evaluator directory: {self.evaluator_dir}")
        print(f"Reports directory: {self.reports_dir}")
        print(f"Total batches: {len(all_batch_dirs)}")
        print(f"Batches to evaluate: {len(batch_dirs)}")
        if filter_project:
            print(f"Project filter: {filter_project}")
        if filter_model:
            print(f"Model filter: {filter_model}")
        if filter_models:
            print(f"Models filter: {', '.join(filter_models)}")
        if filter_mode:
            print(f"Mode filter: {filter_mode}")
        print(f"{'='*80}\n")

        all_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_batches': len(batch_dirs),
            'filters': {
                'project': filter_project,
                'model': filter_model,
                'models': filter_models,
                'mode': filter_mode
            },
            'batches': []
        }

        # Evaluate each batch
        for i, batch_dir in enumerate(batch_dirs, 1):
            print(f"[{i:3d}/{len(batch_dirs)}] Evaluating: {batch_dir.name}")

            try:
                batch_result = self.evaluate_single_batch(batch_dir)
                all_results['batches'].append(batch_result)

                # Print brief results
                if 'statistics' in batch_result:
                    stats = batch_result['statistics']
                    print(f"           Task success rate: {stats.get('task_success_rate', 0)*100:.1f}% "
                          f"({stats.get('task_success_count', 0)}/{stats.get('valid_queries', 0)})")
                    print(f"           Checkpoint pass rate: {stats.get('checkpoint_pass_rate', 0)*100:.1f}% "
                          f"({stats.get('passed_checkpoints', 0)}/{stats.get('total_checkpoints', 0)})")
                    print(f"           Average steps: {stats.get('avg_steps', 0):.1f}")
                elif 'error' in batch_result:
                    print(f"           ❌ Error: {batch_result['error']}")
                print()

            except Exception as e:
                print(f"           ❌ Exception: {str(e)}")
                all_results['batches'].append({
                    'batch_name': batch_dir.name,
                    'error': f'Exception during evaluation: {str(e)}'
                })
                print()

        # Generate summary statistics
        all_results['summary'] = self.generate_summary_statistics(all_results['batches'])

        return all_results

    def generate_summary_statistics(self, batches: List[Dict]) -> Dict:
        """Generate summary statistics"""
        # Group by model
        by_model = defaultdict(list)
        # Group by mode
        by_mode = defaultdict(list)
        # Group by project
        by_project = defaultdict(list)
        # Group by model and mode combination
        by_model_mode = defaultdict(list)

        for batch in batches:
            if 'error' in batch or 'statistics' not in batch:
                continue

            model = batch.get('model', 'unknown')
            mode = batch.get('mode', 'unknown')
            project = batch.get('project', 'unknown')
            stats = batch['statistics']

            by_model[model].append(stats)
            by_mode[mode].append(stats)
            by_project[project].append(stats)
            by_model_mode[f"{model}_{mode}"].append(stats)

        summary = {
            'by_model': {},
            'by_mode': {},
            'by_project': {},
            'by_model_mode': {}
        }

        # Aggregate statistics
        for model, stats_list in by_model.items():
            summary['by_model'][model] = self.aggregate_statistics(stats_list)

        for mode, stats_list in by_mode.items():
            summary['by_mode'][mode] = self.aggregate_statistics(stats_list)

        for project, stats_list in by_project.items():
            summary['by_project'][project] = self.aggregate_statistics(stats_list)

        for key, stats_list in by_model_mode.items():
            summary['by_model_mode'][key] = self.aggregate_statistics(stats_list)

        return summary

    def aggregate_statistics(self, stats_list: List[Dict]) -> Dict:
        """Aggregate statistics from multiple batches"""
        if not stats_list:
            return {}

        total_queries = sum(s.get('valid_queries', 0) for s in stats_list)
        total_success = sum(s.get('task_success_count', 0) for s in stats_list)
        total_agent_success = sum(s.get('agent_success_count', 0) for s in stats_list)
        total_checkpoints = sum(s.get('total_checkpoints', 0) for s in stats_list)
        total_passed_checkpoints = sum(s.get('passed_checkpoints', 0) for s in stats_list)

        avg_steps = sum(s.get('avg_steps', 0) * s.get('valid_queries', 0) for s in stats_list) / total_queries if total_queries > 0 else 0

        return {
            'batches_count': len(stats_list),
            'total_queries': total_queries,
            'task_success_count': total_success,
            'task_success_rate': total_success / total_queries if total_queries > 0 else 0,
            'agent_success_count': total_agent_success,
            'agent_success_rate': total_agent_success / total_queries if total_queries > 0 else 0,
            'total_checkpoints': total_checkpoints,
            'passed_checkpoints': total_passed_checkpoints,
            'checkpoint_pass_rate': total_passed_checkpoints / total_checkpoints if total_checkpoints > 0 else 0,
            'avg_steps': avg_steps,
        }

    def save_results(self, results: Dict, output_prefix: str = "comprehensive_evaluation") -> Tuple[Path, Path]:
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON format
        json_file = self.reports_dir / f"{output_prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save human-readable text format
        txt_file = self.reports_dir / f"{output_prefix}_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(self.format_summary_text(results))

        return json_file, txt_file

    def format_summary_text(self, results: Dict) -> str:
        """Format summary report as text"""
        lines = []
        lines.append("=" * 100)
        lines.append("Comprehensive Evaluation Report".center(100))
        lines.append("=" * 100)
        lines.append(f"\nEvaluation time: {results['evaluation_timestamp']}")
        lines.append(f"Total batches: {results['total_batches']}")
        lines.append("")

        summary = results.get('summary', {})

        # Summary by model
        if summary.get('by_model'):
            lines.append("\n" + "=" * 120)
            lines.append("Summary by Model".center(120))
            lines.append("=" * 120)
            lines.append(f"\n{'Model':<30} {'Batches':<8} {'Task Success Rate':<20} {'Checkpoint Pass Rate':<20} {'Agent Self-Rate':<15} {'Avg Steps':<10}")
            lines.append("-" * 120)

            for model in sorted(summary['by_model'].keys()):
                stats = summary['by_model'][model]
                lines.append(
                    f"{model:<30} "
                    f"{stats['batches_count']:<8} "
                    f"{stats['task_success_rate']*100:>6.1f}% ({stats['task_success_count']}/{stats['total_queries']})"
                    f"{stats['checkpoint_pass_rate']*100:>10.1f}%          "
                    f"{stats['agent_success_rate']*100:>6.1f}%         "
                    f"{stats['avg_steps']:>8.1f}"
                )

        # Summary by mode
        if summary.get('by_mode'):
            lines.append("\n" + "=" * 100)
            lines.append("Summary by Mode".center(100))
            lines.append("=" * 100)
            lines.append(f"\n{'Mode':<15} {'Batches':<8} {'Task Success Rate':<20} {'Checkpoint Pass Rate':<20} {'Avg Steps':<10}")
            lines.append("-" * 100)

            for mode in sorted(summary['by_mode'].keys()):
                stats = summary['by_mode'][mode]
                lines.append(
                    f"{mode:<15} "
                    f"{stats['batches_count']:<8} "
                    f"{stats['task_success_rate']*100:>6.1f}% ({stats['task_success_count']}/{stats['total_queries']})"
                    f"{stats['checkpoint_pass_rate']*100:>10.1f}%          "
                    f"{stats['avg_steps']:>8.1f}"
                )

        # Summary by project
        if summary.get('by_project'):
            lines.append("\n" + "=" * 100)
            lines.append("Summary by Project".center(100))
            lines.append("=" * 100)
            lines.append(f"\n{'Project':<20} {'Batches':<8} {'Task Success Rate':<20} {'Checkpoint Pass Rate':<20} {'Avg Steps':<10}")
            lines.append("-" * 100)

            for project in sorted(summary['by_project'].keys()):
                stats = summary['by_project'][project]
                lines.append(
                    f"{project:<20} "
                    f"{stats['batches_count']:<8} "
                    f"{stats['task_success_rate']*100:>6.1f}% ({stats['task_success_count']}/{stats['total_queries']})"
                    f"{stats['checkpoint_pass_rate']*100:>10.1f}%          "
                    f"{stats['avg_steps']:>8.1f}"
                )

        # Agent self-evaluation comparison
        lines.append("\n" + "=" * 100)
        lines.append("Agent Self-Evaluation vs Actual Success Rate (by Model)".center(100))
        lines.append("=" * 100)
        lines.append(f"\n{'Model':<30} {'Agent Self-Success Rate':<20} {'Actual Task Success Rate':<20} {'Difference':<10}")
        lines.append("-" * 100)

        for model in sorted(summary.get('by_model', {}).keys()):
            stats = summary['by_model'][model]
            agent_rate = stats['agent_success_rate'] * 100
            task_rate = stats['task_success_rate'] * 100
            diff = agent_rate - task_rate
            lines.append(
                f"{model:<30} "
                f"{agent_rate:>8.1f}%           "
                f"{task_rate:>8.1f}%           "
                f"{diff:>+7.1f}%"
            )

        lines.append("\n" + "=" * 100)
        return "\n".join(lines)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive batch evaluation tool')
    parser.add_argument('--test-results', '-t',
                       default='/Users/haoyuebai/SimWebBench/ConEnv/Agent/test_results',
                       help='Test results directory')
    parser.add_argument('--evaluator', '-e',
                       default='/Users/haoyuebai/SimWebBench/ConEnv/Agent/evaluator',
                       help='Evaluator directory')
    parser.add_argument('--output', '-o',
                       default='/Users/haoyuebai/SimWebBench/ConEnv/Agent/evaluation_reports',
                       help='Output reports directory')
    parser.add_argument('--project', '-p', help='Filter by project')
    parser.add_argument('--model', '-m', help='Filter by single model (mutually exclusive with --models)')
    parser.add_argument('--models', '-M', nargs='+',
                       help='Filter by model list, e.g.: --models claude-sonnet gpt-5.2 (mutually exclusive with --model)')
    parser.add_argument('--mode', '-d', help='Filter by mode')
    parser.add_argument('--output-prefix', default='comprehensive_evaluation',
                       help='Output file prefix')

    args = parser.parse_args()

    # Check mutually exclusive parameters
    if args.model and args.models:
        print("Error: --model and --models cannot be used together")
        sys.exit(1)

    # Create evaluator
    evaluator = ComprehensiveBatchEvaluator(
        test_results_dir=args.test_results,
        evaluator_dir=args.evaluator,
        reports_dir=args.output
    )

    # Run evaluation
    results = evaluator.evaluate_all_batches(
        filter_project=args.project,
        filter_model=args.model,
        filter_models=args.models,
        filter_mode=args.mode
    )

    # Save results
    json_file, txt_file = evaluator.save_results(results, args.output_prefix)

    print(f"\n{'='*80}")
    print("✅ Evaluation completed!")
    print(f"{'='*80}")
    print(f"JSON report: {json_file}")
    print(f"Text report: {txt_file}")
    print(f"{'='*80}\n")

    # Print brief summary
    print(evaluator.format_summary_text(results))


if __name__ == "__main__":
    main()
