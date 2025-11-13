"""Data loading and processing utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import polars as pl
import numpy as np
from loguru import logger


class DataLoader:
    """Data loader for the college experience dataset."""

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.logger = logger.bind(name=__name__)

    def load_csv(
        self,
        file_path: Union[str, Path],
        lazy: bool = False,
        **kwargs,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Load CSV file using Polars.

        Args:
            file_path: Path to CSV file
            lazy: Whether to load lazily
            **kwargs: Additional arguments for pl.read_csv

        Returns:
            Polars DataFrame or LazyFrame
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self.logger.info(f"Loading {path.name}...")

        if lazy:
            return pl.scan_csv(path, **kwargs)
        else:
            return pl.read_csv(path, **kwargs)

    def load_parquet(
        self,
        file_path: Union[str, Path],
        lazy: bool = False,
        **kwargs,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Load Parquet file using Polars.

        Args:
            file_path: Path to Parquet file
            lazy: Whether to load lazily
            **kwargs: Additional arguments for pl.read_parquet

        Returns:
            Polars DataFrame or LazyFrame
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self.logger.info(f"Loading {path.name}...")

        if lazy:
            return pl.scan_parquet(path, **kwargs)
        else:
            return pl.read_parquet(path, **kwargs)

    def save_parquet(
        self,
        df: pl.DataFrame,
        file_path: Union[str, Path],
        compression: str = "snappy",
    ) -> None:
        """
        Save DataFrame to Parquet file.

        Args:
            df: Polars DataFrame
            file_path: Output file path
            compression: Compression method
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving {path.name}...")
        df.write_parquet(path, compression=compression)
        self.logger.info(f"Saved {len(df)} rows to {path}")

    def validate_dataframe(
        self,
        df: pl.DataFrame,
        required_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate DataFrame and return quality metrics.

        Args:
            df: Polars DataFrame
            required_columns: List of required columns

        Returns:
            Dictionary with validation results
        """
        validation = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns,
            "missing_values": {},
            "data_types": {},
            "issues": [],
        }

        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation["issues"].append(f"Missing required columns: {missing_cols}")

        # Check missing values
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                validation["missing_values"][col] = {
                    "count": null_count,
                    "percentage": (null_count / len(df)) * 100,
                }

        # Check data types
        for col, dtype in zip(df.columns, df.dtypes):
            validation["data_types"][col] = str(dtype)

        return validation

    def split_by_agent(
        self,
        df: pl.DataFrame,
        column_mapping: Dict[str, List[str]],
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """
        Split DataFrame into agent-specific buckets.

        Args:
            df: Full DataFrame
            column_mapping: Mapping of agent name to column list
            output_dir: Output directory for agent buckets

        Returns:
            Dictionary mapping agent name to output file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = {}

        for agent_name, columns in column_mapping.items():
            # Get common identifier columns
            id_cols = [col for col in ["student_id", "uid", "timestamp"] if col in df.columns]

            # Select relevant columns
            selected_cols = list(set(id_cols + columns))
            available_cols = [col for col in selected_cols if col in df.columns]

            if not available_cols:
                self.logger.warning(f"No columns found for {agent_name}")
                continue

            agent_df = df.select(available_cols)

            # Save to parquet
            output_file = output_path / f"{agent_name}_data.parquet"
            self.save_parquet(agent_df, output_file)
            output_files[agent_name] = output_file

            self.logger.info(
                f"Created {agent_name} bucket: {len(agent_df)} rows, {len(agent_df.columns)} columns"
            )

        return output_files

    def merge_dataframes(
        self,
        dfs: List[pl.DataFrame],
        on: Union[str, List[str]],
        how: str = "inner",
    ) -> pl.DataFrame:
        """
        Merge multiple DataFrames.

        Args:
            dfs: List of DataFrames to merge
            on: Column(s) to join on
            how: Join strategy (inner, left, outer, etc.)

        Returns:
            Merged DataFrame
        """
        if not dfs:
            raise ValueError("No DataFrames provided")

        if len(dfs) == 1:
            return dfs[0]

        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on=on, how=how)

        self.logger.info(f"Merged {len(dfs)} DataFrames: {len(result)} rows")
        return result

    def clean_dataframe(
        self,
        df: pl.DataFrame,
        drop_duplicates: bool = True,
        fill_strategy: Optional[Dict[str, Any]] = None,
    ) -> pl.DataFrame:
        """
        Clean DataFrame by handling duplicates and missing values.

        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            fill_strategy: Dictionary mapping column names to fill values/strategies

        Returns:
            Cleaned DataFrame
        """
        cleaned = df.clone()

        # Drop duplicates
        if drop_duplicates:
            original_len = len(cleaned)
            cleaned = cleaned.unique()
            dropped = original_len - len(cleaned)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped} duplicate rows")

        # Fill missing values
        if fill_strategy:
            for col, strategy in fill_strategy.items():
                if col not in cleaned.columns:
                    continue

                if isinstance(strategy, (int, float, str)):
                    cleaned = cleaned.with_columns(
                        pl.col(col).fill_null(strategy)
                    )
                elif strategy == "mean":
                    cleaned = cleaned.with_columns(
                        pl.col(col).fill_null(pl.col(col).mean())
                    )
                elif strategy == "median":
                    cleaned = cleaned.with_columns(
                        pl.col(col).fill_null(pl.col(col).median())
                    )
                elif strategy == "forward":
                    cleaned = cleaned.with_columns(
                        pl.col(col).fill_null(strategy="forward")
                    )
                elif strategy == "backward":
                    cleaned = cleaned.with_columns(
                        pl.col(col).fill_null(strategy="backward")
                    )

        return cleaned

    def get_summary_statistics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Get summary statistics for DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with summary statistics
        """
        return df.describe()

    def filter_by_date_range(
        self,
        df: pl.DataFrame,
        date_col: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Filter DataFrame by date range.

        Args:
            df: Input DataFrame
            date_col: Name of date column
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Filtered DataFrame
        """
        filtered = df

        if start_date:
            filtered = filtered.filter(pl.col(date_col) >= start_date)

        if end_date:
            filtered = filtered.filter(pl.col(date_col) <= end_date)

        return filtered
