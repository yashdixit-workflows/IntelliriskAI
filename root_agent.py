import os
import sys
from typing import Dict, Any, Optional, Tuple
from agents.coordinator_agent import CoordinatorAgent

class InsightPilotPlatform:
    """
    Main interface for the InsightPilot AI platform.
    Programmatically coordinates multi-agent analysis, prediction, and reports.
    """
    def __init__(self):
        self.coordinator = CoordinatorAgent()

    def run_pipeline(self, file_path: str, model_type: str = "Random Forest") -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Runs the complete analysis pipeline on a CSV/Excel file.
        1. Loads & Clean Data
        2. Profiles Schema and Quality
        3. Generates EDA narration and plots
        4. Trains predicting model
        5. Generates PDF/Word report documents
        
        Returns:
            Tuple[success (bool), coordinator_state (dict), error_message (str)]
        """
        if not os.path.exists(file_path):
            return False, {}, f"Dataset file not found: {file_path}"
            
        print(f"[*] Initializing platform pipeline for: {os.path.basename(file_path)}")
        success, err = self.coordinator.initialize_pipeline(file_path)
        if not success:
            return False, {}, f"Failed to initialize pipeline: {err}"
            
        print("[*] Dataset loaded and schema inferred successfully.")
        
        # Check prediction compatibility
        schema = self.coordinator.state["schema"]
        target = schema.get("target_column")
        
        if target:
            print(f"[*] Training predictor model: '{model_type}' on target: '{target}'")
            try:
                self.coordinator.train_predictive_model(model_type)
                print("[*] Prediction model training and portfolio scoring completed.")
            except Exception as e:
                print(f"[!] Warning: Model training failed: {e}. Platform will continue in descriptive mode.")
        else:
            print("[*] No target column found. Platform running in descriptive EDA mode.")

        print("[*] Generating Word and PDF report documents...")
        try:
            self.coordinator.generate_reports()
            print("[*] Reports written successfully to outputs/ directory.")
        except Exception as e:
            print(f"[!] Warning: Report generation failed: {e}")
            
        return True, self.coordinator.state, None

if __name__ == "__main__":
    # Test script entry point
    platform = InsightPilotPlatform()
    dataset_path = r"x:\creditguard-ai\datasets\Delinquency_prediction_dataset.xlsx"
    if os.path.exists(dataset_path):
        success, state, err = platform.run_pipeline(dataset_path)
        if success:
            print("\n=== PIPELINE RUN COMPLETED SUCCESSFULLY ===")
            print("KPIs:", state.get("kpis"))
            print("Active Model:", state.get("active_model_name"))
            print("Generated Reports:", state.get("report_paths"))
        else:
            print("\n=== PIPELINE RUN FAILED ===")
            print("Error:", err)
    else:
        print("Test dataset not found at default path.")
