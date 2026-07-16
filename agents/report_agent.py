import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
from tools.report_generator import ReportGenerator

class ReportAgent:
    """
    Report Agent: Coordinates document creation and exports (PDF, Word, CSV, Excel).
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.output_dir = r"x:\creditguard-ai\outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_reports(self, coordinator_state: Dict[str, Any]) -> Dict[str, str]:
        """
        Creates PDF, Word, and Excel outputs and returns their saved paths.
        """
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "generation_time": generation_time
        }
        
        pdf_path = os.path.join(self.output_dir, "executive_report.pdf")
        docx_path = os.path.join(self.output_dir, "executive_report.docx")
        csv_path = os.path.join(self.output_dir, "risk_predictions.csv")
        
        results = {}
        
        # 1. Generate Word Report
        success_docx = ReportGenerator.generate_docx(docx_path, metadata, coordinator_state)
        if success_docx:
            results["docx"] = docx_path
            
        # 2. Generate PDF Report
        success_pdf = ReportGenerator.generate_pdf(pdf_path, metadata, coordinator_state)
        if success_pdf:
            results["pdf"] = pdf_path
            
        # 3. Export predictions to CSV
        predictions_df = coordinator_state.get("predictions_df")
        if predictions_df is not None:
            predictions_df.to_csv(csv_path, index=False)
            results["csv"] = csv_path
            
        return results
