#!/usr/bin/env python3
"""
Script to load environment variables from .env file and run run_multienv_o3.py
with specified parameters.
"""

import os
import sys
import subprocess
from pathlib import Path

def load_env_file(env_path):
    """Load environment variables from .env file"""
    if not os.path.exists(env_path):
        print(f"Warning: .env file not found at {env_path}")
        return

    print(f"Loading environment variables from {env_path}")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

                os.environ[key] = value
                print(f"Set {key} = {value}")

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # Path to .env file
    env_path = script_dir / ".env"

    # Load environment variables from .env file
    load_env_file(env_path)

    # Path to run_multienv_o3.py
    run_script = script_dir / "run_multienv_o3.py"

    if not run_script.exists():
        print(f"Error: run_multienv_o3.py not found at {run_script}")
        sys.exit(1)

    # Prepare command arguments
    cmd_args = [
        sys.executable,  # Use current Python interpreter
        str(run_script),
        "--provider_name", "docker",
        # "--headless",
        "--observation_type", "screenshot",
        "--model", "o3",
        "--num_envs", "1",
        "--path_to_vm", "/root/AgenticLearning/V2P/OSWorld/vmware_vm_data/Ubuntu.qcow2",
        "--sleep_after_execution", "3",
        "--max_steps", "15",
        "--result_dir", "./results",
        "--client_password", "password",
        "--test_all_meta_path", "evaluation_examples/test_chrome_clear.json"
#        "--test_all_meta_path", "evaluation_examples/test_gimp.json"
    ]

    print("Executing command:")
    print(" ".join(cmd_args))
    print("-" * 50)

    try:
        # Change to the script directory
        os.chdir(script_dir)

        # Execute the command
        result = subprocess.run(cmd_args, check=True)

        print("-" * 50)
        print("Command executed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nCommand interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()