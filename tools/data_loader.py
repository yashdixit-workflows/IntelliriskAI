import pandas as pd
import os
import re
import io
from typing import Optional, Tuple

class DataLoader:
    """
    Handles loading CSV, Excel, and Google Sheets datasets.
    """

    # ── Google Sheets helpers ────────────────────────────────────────────────

    @staticmethod
    def extract_sheet_id(url: str) -> Optional[str]:
        """Extracts the spreadsheet ID from a Google Sheets share URL."""
        patterns = [
            r"/spreadsheets/d/([a-zA-Z0-9-_]+)",
            r"id=([a-zA-Z0-9-_]+)",
        ]
        for pat in patterns:
            m = re.search(pat, url)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def extract_gid(url: str) -> Optional[str]:
        """Extracts the gid (sheet tab id) from the URL, if present."""
        m = re.search(r"[#&?]gid=(\d+)", url)
        return m.group(1) if m else None

    @staticmethod
    def load_google_sheet(
        url: str,
        creds_json_path: Optional[str] = None,
        sheet_name: Optional[str] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Loads data from a Google Sheets URL.

        - Public sheets  → CSV export URL (no auth required)
        - Private sheets → gspread with service account JSON (creds_json_path required)
        """
        sheet_id = DataLoader.extract_sheet_id(url)
        if not sheet_id:
            return None, (
                "Could not extract spreadsheet ID from URL. "
                "Please paste a full Google Sheets share link."
            )

        # ── Public sheet via CSV export ──────────────────────────────────────
        if creds_json_path is None:
            gid = DataLoader.extract_gid(url)
            export_url = (
                f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                + (f"&gid={gid}" if gid else "")
            )
            try:
                import requests
                resp = requests.get(export_url, timeout=30)
                if resp.status_code == 200:
                    df = pd.read_csv(io.StringIO(resp.text))
                    df.columns = [col.strip() for col in df.columns]
                    return df, None
                elif resp.status_code == 401:
                    return None, (
                        "This Google Sheet is private. Make it public "
                        "('Anyone with the link can view') or provide a "
                        "Service Account JSON credentials file."
                    )
                else:
                    return None, (
                        f"Google Sheets export failed (HTTP {resp.status_code}). "
                        "Check that the sheet is shared as 'Anyone with the link'."
                    )
            except Exception as e:
                return None, f"Failed to fetch Google Sheet: {str(e)}"

        # ── Private sheet via gspread + service account ──────────────────────
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_file(creds_json_path, scopes=scopes)
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(sheet_id)
            ws = sh.worksheet(sheet_name) if sheet_name else sh.get_worksheet(0)
            data = ws.get_all_records()
            df = pd.DataFrame(data)
            df.columns = [col.strip() for col in df.columns]
            return df, None
        except ImportError:
            return None, (
                "gspread is not installed. "
                "Run: pip install gspread google-auth"
            )
        except Exception as e:
            return None, f"Failed to open private Google Sheet: {str(e)}"

    # ── Local file loader ────────────────────────────────────────────────────

    @staticmethod
    def load_dataset(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Loads a CSV or Excel dataset into a pandas DataFrame.
        Returns:
            Tuple[DataFrame or None, error_message or None]
        """
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"

        _, ext = os.path.splitext(file_path.lower())
        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                return None, (
                    f"Unsupported file type: {ext}. "
                    "Only CSV and Excel (xls, xlsx) files are supported."
                )

            # Clean up column names by stripping trailing whitespaces
            df.columns = [col.strip() for col in df.columns]
            return df, None

        except Exception as e:
            return None, f"Error loading file {os.path.basename(file_path)}: {str(e)}"

