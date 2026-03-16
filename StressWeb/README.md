# StressWeb

StressWeb is a benchmark framework for testing and evaluating the web interaction capabilities of large language models under stress conditions.

## Project Structure

```
StressWeb/
├── websites/           # Test websites (download from HuggingFace)
├── queries/            # Test queries (download from HuggingFace)
├── evaluator/          # Evaluator scripts
├── test_results/       # Test results output directory
├── environment_manager.py      # Environment manager
├── playwright_controller.py    # Playwright controller
├── web_actions.py             # Web action definitions
├── global_query_runner.py     # Global query runner
├── authorization.py           # API key configuration
├── install_node_modules.sh    # Node modules installation script
└── cleanup_node_modules.sh    # Node modules cleanup script
```

## Quick Start

### 1. Requirements

- Python 3.8+
- Node.js 16+
- npm or yarn

### 2. Download Dataset

Download test data from HuggingFace (link coming soon):

```bash
# Download and extract websites and queries datasets
# HuggingFace dataset link: coming soon
```

Place the downloaded data in the project root directory:
- `websites/` - Contains all test websites
- `queries/` - Contains all test query files

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Install Playwright browsers:

```bash
playwright install chromium
```

### 4. Install Node Modules

Navigate to the `websites` directory and run the installation script:

```bash
cd websites
bash ../install_node_modules.sh
```

The script provides the following options:
1. Install node_modules for all PRD directories
2. Install base version only
3. Install specific PRD directory
4. Count installations needed

### 5. Configure API Keys

Edit the `authorization.py` file and fill in your API keys:

```python
{
  "gpt-4o": {
    "api_key": "your-openai-api-key"
  },
  "claude-opus-4": {
    "api_key": "your-anthropic-api-key"
  },
  "gemini-2.5-flash": {
    "api_key": "your-google-api-key"
  },
  // ... other model configurations
}
```

## Usage Guide

### Basic Running

Run tests with predefined model groups:

```bash
# Run g1 group models (gemini-2.5-flash)
python global_query_runner.py --group g1 --max-workers 8

# Run g2 group models (gpt-5.2, claude-opus-4-5)
python global_query_runner.py --group g2 --max-workers 6
```

### Custom Running

Specify specific models:

```bash
python global_query_runner.py --models "gpt-4o,claude-sonnet-4" --max-workers 4
```

Specify specific modes:

```bash
python global_query_runner.py --group g1 --modes "clean,chaos" --max-workers 8
```

Specify specific query IDs:

```bash
python global_query_runner.py --group g1 --queries "1,3,5-7" --max-workers 4
```

Specify specific projects:

```bash
python global_query_runner.py --group g1 --projects "food,ecom,note" --max-workers 4
```

### Command-Line Arguments

- `--group`: Use predefined model groups (g1/g2/g3/g4)
- `--models`: Custom model list (comma-separated)
- `--modes`: Specify test modes (comma-separated)
  - `clean`: Standard mode
  - `chaos`: Chaos mode
  - `failed`: Failure mode
  - `perturbed`: Perturbation mode
  - `semanticE`: Semantic remapping E mode
  - `semanticS`: Semantic remapping S mode
  - `dom`: DOM noise mode
- `--max-workers`: Maximum concurrent worker processes (default: 8)
- `--max-steps`: Maximum steps per query (default: 100)
- `--queries`: Only run specified query IDs (e.g., 1,3,5-7)
- `--projects`: Only run specified projects (e.g., food,ecom,note)
- `--no-skip`: Force rerun all tasks
- `--no-skip-queries`: Don't skip completed individual queries
- `--dry-run`: List tasks only, don't execute
- `--headless`: Run in headless mode (default: True)

### View Results

Test results are saved in the `test_results/` directory:

```
test_results/
├── batch_{timestamp}_{project}_{model}_{mode}/
│   ├── query_1/
│   │   ├── result.json
│   │   ├── screenshots/
│   │   └── trace.json
│   ├── query_2/
│   └── batch_summary.json
├── global_summary_{timestamp}.json
└── progress_{timestamp}.log
```

### Evaluate Results

Run the comprehensive evaluator:

```bash
cd evaluator
python comprehensive_batch_evaluator.py
```

This generates detailed evaluation reports including:
- Pass rates for each model across different modes
- Checkpoint pass statistics
- Detailed error analysis

## Supported Test Scenarios

StressWeb includes 10 different test scenarios:

1. **calendar** - Calendar management (18 queries)
2. **ecom** - E-commerce (12 queries)
3. **email** - Email system (18 queries)
4. **file** - File management (13 queries)
5. **food** - Food ordering (10 queries)
6. **management** - Project management (18 queries)
7. **network** - Social network (15 queries)
8. **note** - Note-taking app (15 queries)
9. **reservation** - Reservation system (18 queries)
10. **transportation** - Transportation service (12 queries)

## Cleanup

Clean all node_modules (free disk space):

```bash
cd websites
bash ../cleanup_node_modules.sh
```

Clean port registry:

```python
from environment_manager import PortManager
PortManager().release_all_ports()
```

## Troubleshooting

### Port Conflict Issues

If you encounter port conflicts, manually release ports:

```python
from environment_manager import PortManager
pm = PortManager()
pm.release_port(3000)  # Release specific port
pm.release_all_ports()  # Release all ports
```

### npm Installation Failure

Try cleaning npm cache:

```bash
npm cache clean --force
```

### Playwright Browser Issues

Reinstall browsers:

```bash
playwright install --force chromium
```

