"""
Parameter search (grid or random) for emotion prediction model.

Usage:
    # Grid search
    python src/emotions/param_search.py --search_type grid
    python src/emotions/param_search.py --search_type grid --param_grid src/emotions/configs/param_search.yaml
    
    # Random search
    python src/emotions/param_search.py --search_type random
    python src/emotions/param_search.py --search_type random --param_grid src/emotions/configs/param_search_random.yaml --n_samples 50
"""

import os
import sys
import yaml
import argparse
import tempfile
import shutil
import random
from itertools import product
from datetime import datetime, timedelta
import subprocess
import pandas as pd


def load_yaml(path):
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    """Save YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def generate_grid_combinations(param_grid):
    """Generate all combinations from parameter grid."""
    keys = list(param_grid.keys())
    values = [param_grid[k] if isinstance(param_grid[k], list) else [param_grid[k]] for k in keys]
    
    for combination in product(*values):
        yield dict(zip(keys, combination))


def generate_random_combinations(param_grid, n_samples, random_seed=None):
    """Generate random combinations from parameter grid.
    
    For parameters with 2-element lists, samples random integers in [min, max].
    For parameters with >2 elements, samples random choice from list.
    
    Args:
        param_grid: Dictionary of parameter names to lists of values
        n_samples: Number of random samples to generate
        random_seed: Random seed for reproducibility
    
    Yields:
        Dictionary of parameter combinations
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    keys = list(param_grid.keys())
    
    for _ in range(n_samples):
        combination = {}
        for key in keys:
            value_list = param_grid[key] if isinstance(param_grid[key], list) else [param_grid[key]]
            
            # If list has exactly 2 elements and both are numeric (but not bool), treat as range [min, max]
            if (len(value_list) == 2 and 
                not isinstance(value_list[0], bool) and not isinstance(value_list[1], bool) and
                isinstance(value_list[0], (int, float)) and isinstance(value_list[1], (int, float))):
                min_val, max_val = value_list[0], value_list[1]
                # Handle case where min == max
                if min_val == max_val:
                    combination[key] = min_val
                else:
                    combination[key] = random.randint(min_val, max_val)
            else:
                # Otherwise, random choice from list
                combination[key] = random.choice(value_list)
        
        yield combination


def update_config_with_params(base_config, params):
    """Update base config with parameter combination."""
    config = base_config.copy()
    
    # Map parameters to their locations in config
    param_mapping = {
        'window_length': ('dataset', 'window_length'),
        'kt': ('dataset', 'kt'),
        'ks': ('dataset', 'ks'),
        'hidden_channels': ('model', 'hidden_channels'),
        'use_preprocess_mlp': ('model', 'use_preprocess_mlp'),
        'add_self_loops': ('model', 'add_self_loops'),
        'strategies': ('cross_validation', 'strategies'),
    }
    
    for param, value in params.items():
        if param in param_mapping:
            section, key = param_mapping[param]
            config[section][key] = value
    
    return config


def parse_train_results(results_dir, start_time=None):
    """Parse results from training run.
    
    Args:
        results_dir: Directory containing results CSV files
        start_time: Optional datetime of when training started, used to filter for recent files
        
    Returns:
        DataFrame of results or None if not found
    """
    # Find the all_configs_comparison CSV file
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('all_configs_comparison_') and f.endswith('.csv')]
    
    if not csv_files:
        return None
    
    # Sort by filename (which includes timestamp) to get the most recent file
    csv_files.sort(reverse=True)
    
    # If start_time provided, filter for files created after that time
    if start_time is not None:
        for csv_file in csv_files:
            csv_path = os.path.join(results_dir, csv_file)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(csv_path))
            # Allow for a small time buffer (1 minute before start_time)
            if file_mtime >= start_time - timedelta(minutes=1):
                df = pd.read_csv(csv_path)
                print(f"Using results from: {csv_file} (modified: {file_mtime})")
                return df
        # If no file found after start_time, warn and return None
        print(f"WARNING: No all_configs_comparison CSV file found after {start_time}")
        return None
    else:
        # No start_time filter, just use the most recent
        csv_path = os.path.join(results_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        return df


def main():
    parser = argparse.ArgumentParser(description="Parameter search for emotion prediction hyperparameters")
    parser.add_argument("--search_type", type=str, choices=['grid', 'random'], default='grid',
                       help="Search type: 'grid' for exhaustive grid search or 'random' for random sampling")
    parser.add_argument("--base_config", type=str, default="src/emotions/configs/train_config.yaml",
                       help="Path to base training config")
    parser.add_argument("--param_grid", type=str, default=None,
                       help="Path to parameter grid config (defaults based on search_type)")
    parser.add_argument("--n_samples", type=int, default=50,
                       help="Number of random samples (only used for random search)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for random search reproducibility")
    args = parser.parse_args()
    
    # Set default param_grid based on search_type if not provided
    if args.param_grid is None:
        if args.search_type == 'grid':
            args.param_grid = "src/emotions/configs/param_search_grid.yaml"
        else:
            args.param_grid = "src/emotions/configs/param_search_random.yaml"
    
    # Load base config and parameter grid
    base_config = load_yaml(args.base_config)
    param_grid_config = load_yaml(args.param_grid)
    
    print(f"Search type: {args.search_type.upper()}")
    print(f"Base config: {args.base_config}")
    print(f"Parameter grid: {args.param_grid}")
    
    # Extract parameters from config (handle both flat and nested structure)
    if 'random_samples' in param_grid_config:
        if 'random_samples' in param_grid_config:
            args.n_samples = param_grid_config['random_samples']
        param_grid = {k: v for k, v in param_grid_config.items() if k != 'random_samples'}
    else:
        param_grid = param_grid_config
    
    print(f"Number of samples: {args.n_samples}")
    print(f"Parameter grid: {param_grid}")
    
    # Generate parameter combinations based on search type
    if args.search_type == 'grid':
        combinations = list(generate_grid_combinations(param_grid))
        print(f"\nTotal combinations to evaluate: {len(combinations)}")
    else:
        combinations = list(generate_random_combinations(param_grid, args.n_samples, args.random_seed))
        print(f"\nRandom samples to evaluate: {len(combinations)} (seed: {args.random_seed})")
        for i_comb, combination in enumerate(combinations):
            print("Sampled config k,v pairs:", combination)
            if i_comb >= 9:
                print("...")  # Limit output for large number of samples
                break
    
    # Create temporary directory for config files
    temp_dir = tempfile.mkdtemp(prefix="param_search_configs_")
    print(f"Temporary config directory: {temp_dir}")
    
    try:
        # Generate config files
        config_files = []
        for i, params in enumerate(combinations):
            config = update_config_with_params(base_config, params)
            config_path = os.path.join(temp_dir, f"config_{i:04d}.yaml")
            save_yaml(config, config_path)
            config_files.append(config_path)
        
        print(f"Generated {len(config_files)} config files")
        
        # Run training with all configs
        print(f"\nStarting {args.search_type} search training...")
        start_time = datetime.now()
        
        cmd = [
            sys.executable,
            "src/emotions/train.py",
            "--configs"
        ] + config_files
        
        # Set environment to avoid MKL threading issues
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        
        result = subprocess.run(cmd, cwd=os.getcwd(), env=env)
        
        if result.returncode != 0:
            print(f"Training failed with exit code {result.returncode}")
            return
        
        end_time = datetime.now()
        print(f"\n{args.search_type.capitalize()} search completed in {end_time - start_time}")
        
        # Parse results
        results_dir = base_config['logging']['results_dir']
        df = parse_train_results(results_dir, start_time)
        
        if df is None:
            print("Could not find results CSV file")
            return
        
        # Add parameter columns to results via merge to guarantee alignment
        param_records = []
        for i, params in enumerate(combinations):
            record = {'config': f"config_{i:04d}"}
            record.update(params)
            param_records.append(record)
        param_df = pd.DataFrame(param_records)
        df = df.merge(param_df, on='config', how='left')
        param_columns = [c for c in param_df.columns if c != 'config']
        
        # Save summary CSV (in run order)
        search_type_prefix = f"{args.search_type}_search"
        summary_path = os.path.join(results_dir, f"{search_type_prefix}_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nSaved parameter search summary to: {summary_path}")
        
        # Create ordered summary by metrics
        metric_cols = ['mse', 'mae', 'pearson_r', 'r2']
        available_metrics = [m for m in metric_cols if m in df.columns]
        
        if available_metrics:
            # Sort by: (1) MSE, (2) Pearson R (descending), (3) R2 (descending), (4) MAE
            sort_cols = []
            sort_ascending = []
            
            if 'mse' in available_metrics:
                sort_cols.append('mse')
                sort_ascending.append(True)
            
            if 'pearson_r' in available_metrics:
                sort_cols.append('pearson_r')
                sort_ascending.append(False)
            
            if 'r2' in available_metrics:
                sort_cols.append('r2')
                sort_ascending.append(False)
            
            if 'mae' in available_metrics:
                sort_cols.append('mae')
                sort_ascending.append(True)
            
            df_sorted = df.sort_values(by=sort_cols, ascending=sort_ascending)
            
            summary_ordered_path = os.path.join(results_dir, f"{search_type_prefix}_summary_ordered_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
            df_sorted.to_csv(summary_ordered_path, index=False)
            print(f"Saved ordered parameter search summary to: {summary_ordered_path}")
            
            # Print top 5 configurations
            print("\n" + "="*100)
            print("TOP 5 CONFIGURATIONS (by MSE, Pearson R, R2, MAE)")
            print("="*100)
            display_cols = ['config', 'strategy'] + param_columns + available_metrics
            display_cols = [c for c in display_cols if c in df_sorted.columns]
            print(df_sorted[display_cols].head(5).to_string(index=False))
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
