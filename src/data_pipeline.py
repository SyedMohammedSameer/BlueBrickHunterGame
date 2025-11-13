"""
Data acquisition and processing pipeline for the College Experience Dataset.

This script:
1. Downloads the dataset from Kaggle
2. Performs comprehensive EDA
3. Cleans and validates data
4. Splits into agent-specific buckets
5. Generates data quality report
"""

import sys
from pathlib import Path
from typing import Dict, Any
import polars as pl
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader
from src.utils.config_loader import load_config


class CollegeDataPipeline:
    """Pipeline for processing the College Experience Dataset."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_path = Path(self.config.data.raw_path)
        self.processed_path = Path(self.config.data.processed_path)
        self.agent_buckets_path = Path(self.config.data.agent_buckets_path)

        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.agent_buckets_path.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader(self.data_path)
        self.logger = logger.bind(name=__name__)

    def download_dataset(self) -> Path:
        """
        Download dataset from Kaggle.

        Returns:
            Path to downloaded dataset directory
        """
        self.logger.info("Downloading dataset from Kaggle...")

        try:
            import kagglehub
            dataset_path = kagglehub.dataset_download(self.config.data.kaggle_dataset)
            self.logger.info(f"Dataset downloaded to: {dataset_path}")
            return Path(dataset_path)
        except Exception as e:
            self.logger.error(f"Failed to download dataset: {str(e)}")
            self.logger.info("Please manually download the dataset from Kaggle")
            self.logger.info(f"URL: https://www.kaggle.com/datasets/{self.config.data.kaggle_dataset}")
            raise

    def load_all_data(self, dataset_path: Path) -> Dict[str, pl.DataFrame]:
        """
        Load all CSV files from the dataset.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary mapping file names to DataFrames
        """
        self.logger.info("Loading all data files...")

        dataframes = {}
        csv_files = list(dataset_path.glob("**/*.csv"))

        if not csv_files:
            self.logger.warning(f"No CSV files found in {dataset_path}")
            return dataframes

        for csv_file in csv_files:
            try:
                df = self.loader.load_csv(csv_file)
                file_key = csv_file.stem  # Filename without extension
                dataframes[file_key] = df
                self.logger.info(f"Loaded {file_key}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file}: {str(e)}")

        return dataframes

    def perform_eda(self, dataframes: Dict[str, pl.DataFrame]) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis.

        Args:
            dataframes: Dictionary of DataFrames

        Returns:
            Dictionary with EDA results
        """
        self.logger.info("Performing exploratory data analysis...")

        eda_results = {
            "num_files": len(dataframes),
            "total_rows": 0,
            "files": {},
        }

        for name, df in dataframes.items():
            self.logger.info(f"\nAnalyzing {name}...")

            # Basic statistics
            file_stats = {
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns,
                "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                "missing_values": {},
                "summary_stats": None,
            }

            # Missing values
            for col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    file_stats["missing_values"][col] = {
                        "count": null_count,
                        "percentage": round((null_count / len(df)) * 100, 2),
                    }

            # Summary statistics for numeric columns
            numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
            if numeric_cols:
                try:
                    file_stats["summary_stats"] = df.select(numeric_cols).describe().to_dict()
                except:
                    pass

            eda_results["files"][name] = file_stats
            eda_results["total_rows"] += len(df)

            # Print summary
            print(f"\n{'='*60}")
            print(f"File: {name}")
            print(f"{'='*60}")
            print(f"Rows: {len(df):,}")
            print(f"Columns: {len(df.columns)}")
            print(f"Missing Values: {len(file_stats['missing_values'])} columns with nulls")

            if file_stats["missing_values"]:
                print("\nTop columns with missing values:")
                sorted_missing = sorted(
                    file_stats["missing_values"].items(),
                    key=lambda x: x[1]["percentage"],
                    reverse=True
                )[:5]
                for col, stats in sorted_missing:
                    print(f"  {col}: {stats['percentage']:.1f}% ({stats['count']:,} nulls)")

        return eda_results

    def create_unified_dataset(self, dataframes: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Create a unified dataset from multiple files.

        This method attempts to merge all dataframes intelligently based on common columns.

        Args:
            dataframes: Dictionary of DataFrames

        Returns:
            Unified DataFrame
        """
        self.logger.info("Creating unified dataset...")

        if not dataframes:
            raise ValueError("No dataframes provided")

        # Identify common identifier columns
        common_id_cols = ["uid", "student_id", "user_id", "timestamp", "date"]

        # Start with the largest dataframe
        main_df_name = max(dataframes.items(), key=lambda x: len(x[1]))[0]
        unified_df = dataframes[main_df_name].clone()

        self.logger.info(f"Starting with {main_df_name} as base: {len(unified_df)} rows")

        # Merge other dataframes
        for name, df in dataframes.items():
            if name == main_df_name:
                continue

            # Find common columns for joining
            join_cols = [col for col in common_id_cols if col in unified_df.columns and col in df.columns]

            if join_cols:
                self.logger.info(f"Merging {name} on {join_cols}")
                try:
                    # Add suffix to avoid column name conflicts
                    df_renamed = df.rename({col: f"{col}_{name}" for col in df.columns if col not in join_cols})
                    unified_df = unified_df.join(df_renamed, on=join_cols, how="left")
                except Exception as e:
                    self.logger.warning(f"Failed to merge {name}: {str(e)}")
            else:
                self.logger.warning(f"No common columns found for {name}, skipping merge")

        self.logger.info(f"Unified dataset created: {len(unified_df)} rows, {len(unified_df.columns)} columns")
        return unified_df

    def split_into_agent_buckets(self, df: pl.DataFrame) -> Dict[str, Path]:
        """
        Split unified dataset into agent-specific buckets.

        Args:
            df: Unified DataFrame

        Returns:
            Dictionary mapping agent names to file paths
        """
        self.logger.info("Splitting data into agent buckets...")

        # Define column mappings for each agent
        # Note: These are generic mappings - will be adjusted based on actual columns
        available_cols = set(df.columns)

        column_mapping = {
            "spatial": [
                col for col in df.columns
                if any(keyword in col.lower() for keyword in ["gps", "lat", "lon", "location", "place", "semantic", "mobility", "distance"])
            ],
            "behavioral": [
                col for col in df.columns
                if any(keyword in col.lower() for keyword in ["activity", "sleep", "screen", "phone", "app", "usage", "steps", "exercise"])
            ],
            "mental_health": [
                col for col in df.columns
                if any(keyword in col.lower() for keyword in ["phq", "mental", "health", "anxiety", "depression", "stress", "mood", "esteem"])
            ],
            "temporal": [
                col for col in df.columns
                if any(keyword in col.lower() for keyword in ["time", "date", "hour", "day", "week", "month", "year", "timestamp", "duration"])
            ],
            "social": [
                col for col in df.columns
                if any(keyword in col.lower() for keyword in ["social", "communication", "call", "sms", "contact", "interaction", "friend"])
            ],
            "demographic": [
                col for col in df.columns
                if any(keyword in col.lower() for keyword in ["demo", "age", "gender", "cohort", "year", "major", "background", "student_id", "uid"])
            ],
        }

        # Ensure identifier columns are in all buckets
        id_cols = [col for col in ["uid", "student_id", "timestamp", "date"] if col in df.columns]

        for agent_name in column_mapping:
            column_mapping[agent_name] = list(set(id_cols + column_mapping[agent_name]))

        # Split and save
        output_files = self.loader.split_by_agent(df, column_mapping, self.agent_buckets_path)

        return output_files

    def generate_data_quality_report(self, eda_results: Dict[str, Any], output_path: Path) -> None:
        """
        Generate a comprehensive data quality report.

        Args:
            eda_results: EDA results dictionary
            output_path: Output file path
        """
        self.logger.info("Generating data quality report...")

        report = []
        report.append("# College Experience Dataset - Data Quality Report")
        report.append(f"\nGenerated: {pl.datetime('now')}")
        report.append(f"\n## Overview")
        report.append(f"- Total files: {eda_results['num_files']}")
        report.append(f"- Total rows: {eda_results['total_rows']:,}")

        report.append(f"\n## File Details")

        for file_name, stats in eda_results["files"].items():
            report.append(f"\n### {file_name}")
            report.append(f"- Rows: {stats['num_rows']:,}")
            report.append(f"- Columns: {stats['num_columns']}")

            if stats["missing_values"]:
                report.append(f"\n#### Missing Values")
                for col, mv_stats in stats["missing_values"].items():
                    report.append(f"- **{col}**: {mv_stats['percentage']:.1f}% ({mv_stats['count']:,} nulls)")

            report.append(f"\n#### Columns")
            for col, dtype in stats["dtypes"].items():
                report.append(f"- {col}: {dtype}")

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

        self.logger.info(f"Data quality report saved to {output_path}")

    def run_pipeline(self) -> None:
        """Run the complete data pipeline."""
        self.logger.info("Starting data pipeline...")

        try:
            # Step 1: Download dataset
            dataset_path = self.download_dataset()

            # Step 2: Load all data
            dataframes = self.load_all_data(dataset_path)

            if not dataframes:
                self.logger.error("No data loaded. Exiting pipeline.")
                return

            # Step 3: Perform EDA
            eda_results = self.perform_eda(dataframes)

            # Step 4: Generate data quality report
            report_path = Path("outputs/reports/data_quality_report.md")
            self.generate_data_quality_report(eda_results, report_path)

            # Step 5: Create unified dataset (if multiple files)
            if len(dataframes) > 1:
                unified_df = self.create_unified_dataset(dataframes)
            else:
                unified_df = list(dataframes.values())[0]

            # Step 6: Save processed data
            processed_file = self.processed_path / "unified_dataset.parquet"
            self.loader.save_parquet(unified_df, processed_file)

            # Step 7: Split into agent buckets
            agent_files = self.split_into_agent_buckets(unified_df)

            self.logger.info("\n" + "="*60)
            self.logger.info("Data pipeline completed successfully!")
            self.logger.info("="*60)
            self.logger.info(f"Processed data saved to: {processed_file}")
            self.logger.info(f"Agent buckets created: {len(agent_files)}")
            for agent_name, file_path in agent_files.items():
                self.logger.info(f"  - {agent_name}: {file_path}")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Set up logging
    from src.utils.logger import setup_logger
    setup_logger(log_file="logs/data_pipeline.log", level="INFO")

    # Run pipeline
    pipeline = CollegeDataPipeline()
    pipeline.run_pipeline()
