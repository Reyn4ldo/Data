#!/usr/bin/env python3
"""Main script for AMR Machine Learning Pipeline.

This script orchestrates the complete pipeline for antimicrobial resistance
prediction using bacterial isolate data.

Tasks:
1. Multi-class classification: Predict bacterial species
2. Multi-class classification: Predict resistance category
3. Multi-class classification: Predict species from AST only
4. Binary classification: Predict MDR status

Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, get_results_dir, get_models_dir, save_metrics
from src.data_preprocessing import (
    load_raw_data, preprocess_data, save_processed_data
)
from src.feature_engineering import engineer_features
from src.model_training import (
    train_models_for_task, save_task_models
)
from src.model_evaluation import (
    evaluate_all_models, select_best_model, create_comparison_table,
    save_evaluation_results, plot_confusion_matrix, plot_roc_curves,
    plot_calibration_curve
)
from src.model_interpretation import interpret_models


def run_pipeline():
    """Run the complete AMR ML pipeline."""
    
    # Setup
    logger = setup_logging(logging.INFO)
    logger.info("=" * 60)
    logger.info("Starting AMR Machine Learning Pipeline")
    logger.info("=" * 60)
    
    # Create output directories
    results_dir = get_results_dir()
    figures_dir = results_dir / 'figures'
    metrics_dir = results_dir / 'metrics'
    models_dir = get_models_dir()
    
    for dir_path in [figures_dir, metrics_dir, models_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Phase 1 & 2: Data Loading and Preprocessing
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Phase 1 & 2: Data Loading and Preprocessing")
    logger.info("=" * 40)
    
    # Load raw data
    df_raw = load_raw_data()
    logger.info(f"Raw data shape: {df_raw.shape}")
    logger.info(f"Columns: {list(df_raw.columns)}")
    
    # Preprocess data
    df_processed = preprocess_data(df_raw, missing_threshold=0.5)
    
    # Feature engineering
    df_engineered = engineer_features(df_processed)
    
    # Save processed data
    save_processed_data(df_engineered)
    
    # Log data summary
    logger.info(f"\nProcessed data summary:")
    logger.info(f"  Total samples: {len(df_engineered)}")
    if 'bacterial_species' in df_engineered.columns:
        logger.info(f"  Bacterial species: {df_engineered['bacterial_species'].value_counts().to_dict()}")
    if 'resistance_category' in df_engineered.columns:
        logger.info(f"  Resistance categories: {df_engineered['resistance_category'].value_counts().to_dict()}")
    if 'mdr_flag' in df_engineered.columns:
        logger.info(f"  MDR status: {df_engineered['mdr_flag'].value_counts().to_dict()}")
    
    # =========================================================================
    # Phase 3-8: Model Training, Evaluation, and Interpretation
    # =========================================================================
    
    tasks = ['species', 'resistance', 'species_ast_only', 'mdr']
    all_task_results = {}
    
    for task in tasks:
        logger.info("\n" + "=" * 40)
        logger.info(f"Processing Task: {task}")
        logger.info("=" * 40)
        
        # Phase 3 & 4: Train models
        try:
            results = train_models_for_task(df_engineered, task, n_cv_splits=5)
            all_task_results[task] = results
            
            # Phase 5: Evaluate models
            all_metrics = evaluate_all_models(results, task)
            
            # Phase 6: Select best model
            best_model_name, selection_scores = select_best_model(
                all_metrics, results['cv_results'], task
            )
            
            # Create and save comparison table
            comparison_table = create_comparison_table(all_metrics, task)
            save_evaluation_results(all_metrics, comparison_table, task)
            
            logger.info(f"\nModel Comparison for {task}:")
            logger.info(comparison_table.to_string())
            
            # Plot confusion matrices for best model
            best_model = results['trained_models'][best_model_name]
            y_pred = best_model.predict(results['X_test_scaled'])
            
            from sklearn.metrics import confusion_matrix
            import numpy as np
            cm = confusion_matrix(results['y_test'], y_pred)
            
            if task == 'mdr':
                class_names = ['Non-MDR', 'MDR']
            else:
                class_names = list(results['label_encoder'].classes_)
            
            plot_confusion_matrix(
                cm, class_names,
                f'Confusion Matrix - {best_model_name} ({task})',
                str(figures_dir / f'{task}_{best_model_name}_confusion.png')
            )
            
            # Plot ROC curves
            plot_roc_curves(
                results['trained_models'],
                results['X_test_scaled'],
                results['y_test'],
                task,
                str(figures_dir / f'{task}_roc_curves.png')
            )
            
            # For MDR task, plot calibration curve
            if task == 'mdr':
                plot_calibration_curve(
                    best_model,
                    results['X_test_scaled'],
                    results['y_test'],
                    best_model_name,
                    str(figures_dir / f'{task}_{best_model_name}_calibration.png')
                )
            
            # Phase 7: Model interpretation
            interpretation_results = interpret_models(
                results, task, best_model_name, use_shap=True
            )
            
            # Phase 8: Save final models
            save_task_models(results, task)
            
            # Save best model info
            best_model_info = {
                'task': task,
                'best_model': best_model_name,
                'metrics': all_metrics[best_model_name],
                'selection_scores': selection_scores
            }
            save_metrics(best_model_info, f'{task}_best_model.json')
            
            logger.info(f"\nCompleted task: {task}")
            logger.info(f"Best model: {best_model_name}")
            
        except Exception as e:
            logger.error(f"Error processing task {task}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    
    logger.info("\nOutputs generated:")
    logger.info(f"  - Processed data: data/processed_data.csv")
    logger.info(f"  - Models: models/")
    logger.info(f"  - Metrics: results/metrics/")
    logger.info(f"  - Figures: results/figures/")
    
    # Print summary of best models
    logger.info("\nBest Models Summary:")
    for task in tasks:
        try:
            from src.utils import load_metrics
            info = load_metrics(f'{task}_best_model.json')
            f1_key = 'f1' if task == 'mdr' else 'f1_macro'
            f1 = info['metrics'].get(f1_key, 'N/A')
            logger.info(f"  {task}: {info['best_model']} (F1: {f1:.4f})")
        except Exception:
            pass
    
    return all_task_results


if __name__ == '__main__':
    run_pipeline()
