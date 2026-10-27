from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class SalesDataAnalyzer:
    """Load, clean, analyze, combine, split, and visualize sales data."""

    def __init__(self, file_path="data/sales_data.csv"):
        self.file_path = Path(file_path)
        self.data = pd.DataFrame()
        self.load_data(self.file_path)

    def __del__(self):
        pass

    def load_data(self, file_path=None):
        """Load a CSV file into the DataFrame."""
        if file_path:
            self.file_path = Path(file_path)
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.data)} rows from: {self.file_path}")
            return True
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Could not parse CSV: {error}")
        except Exception as error:
            print(f"Unable to load file: {error}")
        return False

    def explore_data(self):
        self._require_data()
        print("\n--- First 5 rows ---")
        print(self.data.head())
        print("\n--- Last 5 rows ---")
        print(self.data.tail())
        print("\n--- Columns ---")
        print(self.data.columns.tolist())
        print("\n--- Data types ---")
        print(self.data.dtypes)
        print("\n--- DataFrame info ---")
        self.data.info()
        print("\n--- Descriptive summary ---")
        print(self.data.describe(include="all").to_string())

    def clean_data(self, strategy="mean"):
        """Convert common types and handle missing values."""
        self._require_data()
        before = self.data.isna().sum().sum()
        if "Date" in self.data.columns:
            self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        text_cols = self.data.select_dtypes(include=["object", "string"]).columns

        for column in numeric_cols:
            if self.data[column].isna().any():
                if strategy == "median":
                    fill_value = self.data[column].median()
                elif strategy == "zero":
                    fill_value = 0
                else:
                    fill_value = self.data[column].mean()
                self.data[column] = self.data[column].fillna(fill_value)

        for column in text_cols:
            if self.data[column].isna().any():
                mode = self.data[column].mode()
                self.data[column] = self.data[column].fillna(
                    mode.iloc[0] if not mode.empty else "Unknown"
                )

        self.data = self.data.drop_duplicates().reset_index(drop=True)
        after = self.data.isna().sum().sum()
        print(f"Missing cells before cleaning: {before}; after cleaning: {after}")
        print("Duplicate rows removed; data types checked.")

    def mathematical_operations(self):
        self._require_data()
        if "Sales" not in self.data.columns:
            print("This operation requires a 'Sales' column.")
            return
        self.data["SalesWithTax"] = (self.data["Sales"] * 1.18).round(2)
        self.data["SalesRounded"] = self.data["Sales"].round(0)
        print(self.data[["Sales", "SalesWithTax", "SalesRounded"]].head())
        values = self.data["Sales"].to_numpy()
        print("\nNumPy array first 5 values:", values[:5])
        print("First value by indexing:", values[0] if len(values) else "No data")
        print("First 3 values by slicing:", values[:3])
        print("Array + 100 (first 5):", (values + 100)[:5])
        print("Array * 2 (first 5):", (values * 2)[:5])

    def combine_data(self, other_dataframe=None):
        self._require_data()
        if other_dataframe is None:
            other_dataframe = self.data.tail(3).copy()
            print("Demonstrating concat by appending a copy of the final 3 rows.")
        combined = pd.concat([self.data, other_dataframe], ignore_index=True)
        print(f"Combined DataFrame shape: {combined.shape}")
        print(combined.tail())
        return combined

    def split_data(self, column="Region"):
        self._require_data()
        if column not in self.data.columns:
            print(f"Column '{column}' was not found.")
            return {}
        groups = {str(key): value.copy() for key, value in self.data.groupby(column)}
        print(f"Split into {len(groups)} group(s) by '{column}':")
        for key, frame in groups.items():
            print(f"  {key}: {len(frame)} rows")
        return groups

    def search_sort_filter(self):
        self._require_data()
        print("\n1. Search product  2. Filter minimum sales  3. Sort by a column")
        choice = input("Choose: ").strip()
        if choice == "1":
            product = input("Product name (partial match allowed): ").strip()
            if "Product" not in self.data.columns:
                print("No 'Product' column.")
                return
            result = self.data[self.data["Product"].astype(str).str.contains(
                product, case=False, na=False
            )]
        elif choice == "2":
            if "Sales" not in self.data.columns:
                print("No 'Sales' column.")
                return
            try:
                minimum = float(input("Minimum sales: "))
            except ValueError:
                print("Please enter a number.")
                return
            result = self.data[self.data["Sales"] >= minimum]
        elif choice == "3":
            column = input(f"Column name {self.data.columns.tolist()}: ").strip()
            if column not in self.data.columns:
                print("Unknown column.")
                return
            ascending = input("Ascending? (y/n): ").strip().lower() != "n"
            result = self.data.sort_values(column, ascending=ascending)
        else:
            print("Invalid choice.")
            return
        print(result.to_string(index=False) if not result.empty else "No matching rows.")

    def aggregate_functions(self):
        self._require_data()
        if not {"Sales", "Product"}.issubset(self.data.columns):
            print("Aggregation requires 'Sales' and 'Product' columns.")
            return
        print("\nTotal sales:", self.data["Sales"].sum())
        print("Average sale:", round(self.data["Sales"].mean(), 2))
        print("Maximum sale:", self.data["Sales"].max())
        print("Minimum sale:", self.data["Sales"].min())
        print("Record count:", self.data["Sales"].count())
        print("\nSales by product:")
        print(self.data.groupby("Product")["Sales"].agg(["sum", "mean", "count"]))
        if "Region" in self.data.columns:
            print("\nSales by region:")
            print(self.data.groupby("Region")["Sales"].agg(["sum", "mean", "count"]))

    def statistical_analysis(self):
        self._require_data()
        numeric = self.data.select_dtypes(include=np.number)
        if numeric.empty:
            print("No numeric columns available.")
            return
        print("\n--- describe() ---")
        print(numeric.describe())
        print("\n--- Standard deviation ---")
        print(numeric.std(numeric_only=True))
        print("\n--- Variance ---")
        print(numeric.var(numeric_only=True))
        print("\n--- 25th, 50th, 75th percentiles ---")
        print(numeric.quantile([0.25, 0.50, 0.75]))

    def create_pivot_table(self):
        self._require_data()
        required = {"Product", "Region", "Sales"}
        if not required.issubset(self.data.columns):
            print("Pivot table requires Product, Region, and Sales columns.")
            return
        pivot = pd.pivot_table(
            self.data, values="Sales", index="Product", columns="Region",
            aggfunc="sum", fill_value=0, margins=True
        )
        print(pivot)
        return pivot

    def groupby_transform(self):
        self._require_data()
        if not {"Region", "Sales"}.issubset(self.data.columns):
            print("This operation requires Region and Sales columns.")
            return
        self.data["RegionTotalSales"] = self.data.groupby("Region")["Sales"].transform("sum")
        self.data["RegionSalesSharePct"] = (
            self.data["Sales"] / self.data["RegionTotalSales"] * 100
        ).round(2)
        print(self.data[["Region", "Sales", "RegionTotalSales", "RegionSalesSharePct"]].head(10))

    def visualize_data(self, chart_type=None):
        self._require_data()
        if chart_type is None:
            print("\n1. Bar  2. Line  3. Scatter  4. Pie  5. Histogram")
            print("6. Stack plot  7. Seaborn heatmap  8. Seaborn box plot")
            chart_type = input("Choose chart: ").strip()
        chart_type = str(chart_type).strip().lower()
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5))

        try:
            if chart_type in ("1", "bar"):
                if not {"Product", "Sales"}.issubset(self.data.columns):
                    raise ValueError("Bar chart requires Product and Sales columns.")
                grouped = self.data.groupby("Product")["Sales"].sum().sort_values(ascending=False)
                grouped.plot(kind="bar", ax=ax, title="Total Sales by Product")
                ax.set_ylabel("Sales")
            elif chart_type in ("2", "line"):
                if not {"Date", "Sales"}.issubset(self.data.columns):
                    raise ValueError("Line chart requires Date and Sales columns.")
                dated = self.data.copy()
                dated["Date"] = pd.to_datetime(dated["Date"], errors="coerce")
                dated = dated.dropna(subset=["Date"]).sort_values("Date")
                ax.plot(dated["Date"], dated["Sales"], marker="o", label="Sales")
                ax.set_title("Sales Over Time")
                ax.set_ylabel("Sales")
                ax.legend()
                fig.autofmt_xdate()
            elif chart_type in ("3", "scatter"):
                if not {"Sales", "Profit"}.issubset(self.data.columns):
                    raise ValueError("Scatter chart requires Sales and Profit columns.")
                sns.scatterplot(data=self.data, x="Sales", y="Profit", hue="Region" if "Region" in self.data else None, ax=ax)
                ax.set_title("Sales vs Profit")
            elif chart_type in ("4", "pie"):
                if not {"Product", "Sales"}.issubset(self.data.columns):
                    raise ValueError("Pie chart requires Product and Sales columns.")
                grouped = self.data.groupby("Product")["Sales"].sum()
                ax.pie(grouped, labels=grouped.index, autopct="%1.1f%%", startangle=90)
                ax.set_title("Sales Share by Product")
            elif chart_type in ("5", "histogram"):
                if "Sales" not in self.data.columns:
                    raise ValueError("Histogram requires a Sales column.")
                sns.histplot(data=self.data, x="Sales", bins=8, kde=True, ax=ax)
                ax.set_title("Sales Distribution")
            elif chart_type in ("6", "stack", "stack plot"):
                if not {"Region", "Product", "Sales"}.issubset(self.data.columns):
                    raise ValueError("Stack plot requires Region, Product, and Sales columns.")
                pivot = self.data.pivot_table(index="Region", columns="Product", values="Sales", aggfunc="sum", fill_value=0)
                pivot.plot(kind="bar", stacked=True, ax=ax, title="Sales by Region and Product")
                ax.set_ylabel("Sales")
                ax.legend(title="Product", bbox_to_anchor=(1.02, 1), loc="upper left")
            elif chart_type in ("7", "heatmap"):
                numeric = self.data.select_dtypes(include=np.number)
                if numeric.empty:
                    raise ValueError("Heatmap requires numeric columns.")
                sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                ax.set_title("Correlation Heatmap")
            elif chart_type in ("8", "box", "boxplot"):
                if not {"Product", "Sales"}.issubset(self.data.columns):
                    raise ValueError("Box plot requires Product and Sales columns.")
                sns.boxplot(data=self.data, x="Product", y="Sales", ax=ax)
                ax.set_title("Sales Distribution by Product")
                ax.tick_params(axis="x", rotation=20)
            else:
                plt.close(fig)
                print("Unknown chart type.")
                return
            ax.set_xlabel(ax.get_xlabel() or "")
            fig.tight_layout()
            plt.show()
            return fig
        except Exception as error:
            plt.close(fig)
            print(f"Could not create chart: {error}")

    def save_visualization(self, filename="sales_chart.png"):
        fig = plt.gcf()
        if not plt.get_fignums():
            print("Create a chart first.")
            return
        output = Path("outputs")
        output.mkdir(exist_ok=True)
        path = output / filename
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Chart saved to: {path}")
        except Exception as error:
            print(f"Could not save chart: {error}")

    def export_data(self):
        self._require_data()
        output = Path("outputs")
        output.mkdir(exist_ok=True)
        path = output / "sales_analysis_export.csv"
        self.data.to_csv(path, index=False)
        print(f"Current data exported to: {path}")

    def _require_data(self):
        if self.data.empty:
            raise ValueError("No dataset is loaded. Select 'Load Dataset' first.")
