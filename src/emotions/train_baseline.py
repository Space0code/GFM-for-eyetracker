"""
Train and evaluate baseline models.

Usage: python src/emotions/train_baseline.py
"""

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data_tabular import TabularDataset
from emotions.baseline_model import get_all_baselines


def train_and_evaluate(model, X_train, y_train, X_test, y_test, save_dir):
    """Train model and save results."""
    print(f"\nTraining {model.name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("Train metrics - MSE: {:.4f} | MAE: {:.4f} | SD Error: {:.4f} | Spearman: {:.4f} | CCC: {:.4f}".format(
        train_metrics['aggregated']['mse'],
        train_metrics['aggregated']['mae'],
        train_metrics['aggregated']['sd_error'],
        train_metrics['aggregated']['spearman'],
        train_metrics['aggregated']['ccc'],
    ))
    print("Test metrics  - MSE: {:.4f} | MAE: {:.4f} | SD Error: {:.4f} | Spearman: {:.4f} | CCC: {:.4f}".format(
        test_metrics['aggregated']['mse'],
        test_metrics['aggregated']['mae'],
        test_metrics['aggregated']['sd_error'],
        test_metrics['aggregated']['spearman'],
        test_metrics['aggregated']['ccc'],
    ))
    
    # Save model
    model_dir = os.path.join(save_dir, model.name)
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save predictions
    y_pred_test = model.predict(X_test)
    results = {
        'predictions': y_pred_test,
        'targets': y_test,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    with open(os.path.join(model_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return train_metrics, test_metrics


def main():
    data_dir = "./data/processed/eSEEd_v2/"
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"./results/Baseline/eSEEd_v2/{timestamp}/"
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    # Load tabular dataset
    print("Loading dataset...")
    dataset = TabularDataset(
        root_dir=data_dir,
        # file_list=["sample_01_recording_01_merged.csv",
        #            "sample_02_recording_01_merged.csv",
        #            "sample_03_recording_01_merged.csv",
        #            "sample_04_recording_01_merged.csv"],
        window_length=10,
        test_size=0.2,
        random_state=42
    )
    
    # Train all baselines
    print("\n" + "="*60)
    print("Training Baseline Models")
    print("="*60)
    
    results_summary = []
    
    for model in get_all_baselines():
        train_metrics, test_metrics = train_and_evaluate(
            model, 
            dataset.X_train, 
            dataset.y_train,
            dataset.X_test, 
            dataset.y_test,
            run_dir
        )
        results_summary.append({
            'model': model.name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
    
    # Print summary
    print("\n" + "="*100)
    print("Results Summary")
    print("="*100)
    print(f"{'Model':<15} | {'MSE':<8} | {'MAE':<8} | {'SD_Err':<8} | {'Spearman':<8} | {'CCC':<8}")
    print("-"*100)
    for res in results_summary:
        tm = res['test_metrics']
        print(f"{res['model']:<15} | {tm['aggregated']['mse']:<8.4f} | {tm['aggregated']['mae']:<8.4f} | {tm['aggregated']['sd_error']:<8.4f} | {tm['aggregated']['spearman']:<8.4f} | {tm['aggregated']['ccc']:<8.4f}")
    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
