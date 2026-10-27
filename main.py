from pathlib import Path
from sales_analyzer import SalesDataAnalyzer


def show_menu():
    print("\n========== Pandas Analyzer & Data Visualization ==========")
    print("1. Load Dataset")
    print("2. Explore Data")
    print("3. Clean Missing Data / Duplicates")
    print("4. Mathematical Operations + NumPy")
    print("5. Search, Sort, Filter")
    print("6. Aggregation Functions")
    print("7. Statistical Analysis")
    print("8. Combine DataFrames")
    print("9. Split DataFrame by Region/Product")
    print("10. Create Pivot Table")
    print("11. GroupBy + Transform")
    print("12. Data Visualization")
    print("13. Save Current Chart")
    print("14. Export Analyzed Data")
    print("0. Exit")
    print("==========================================================")


def main():
    default_csv = Path("data/sales_data.csv")
    analyzer = SalesDataAnalyzer(default_csv)

    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()
        try:
            if choice == "1":
                path = input("CSV path (Enter for sample data): ").strip()
                analyzer.load_data(path or default_csv)
            elif choice == "2":
                analyzer.explore_data()
            elif choice == "3":
                print("Fill strategy: 1. Mean  2. Median  3. Zero")
                strategy = input("Choose: ").strip()
                analyzer.clean_data({"1": "mean", "2": "median", "3": "zero"}.get(strategy, "mean"))
            elif choice == "4":
                analyzer.mathematical_operations()
            elif choice == "5":
                analyzer.search_sort_filter()
            elif choice == "6":
                analyzer.aggregate_functions()
            elif choice == "7":
                analyzer.statistical_analysis()
            elif choice == "8":
                analyzer.combine_data()
            elif choice == "9":
                column = input("Split column (Region/Product; default Region): ").strip() or "Region"
                analyzer.split_data(column)
            elif choice == "10":
                analyzer.create_pivot_table()
            elif choice == "11":
                analyzer.groupby_transform()
            elif choice == "12":
                analyzer.visualize_data()
            elif choice == "13":
                filename = input("Filename (e.g. sales_chart.png): ").strip() or "sales_chart.png"
                analyzer.save_visualization(filename)
            elif choice == "14":
                analyzer.export_data()
            elif choice == "0":
                print("Thank you for using Pandas Analyzer. Goodbye!")
                break
            else:
                print("Invalid choice. Please select an option from the menu.")
        except ValueError as error:
            print(f"Input/data error: {error}")
        except KeyboardInterrupt:
            print("\nOperation interrupted.")
        except Exception as error:
            print(f"Unexpected error: {error}")


if __name__ == "__main__":
    main()
