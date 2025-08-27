# BiobankTidy - Advanced Biobanking Data Analysis Platform

## 🧬 Overview

BiobankTidy is a sophisticated data analysis platform designed for biobanking research, featuring advanced filtering capabilities and interactive visualizations. Built with Streamlit and connected to Snowflake, it provides researchers with powerful tools to explore and analyze large-scale biobanking datasets.

## ✨ Key Features

### 🔍 Advanced Filtering System
- **Column Categorization**: Automatically organizes 1500+ columns into logical categories:
  - Demographics (Age, Gender, Race, etc.)
  - Medical History (Hx_* columns)
  - Symptoms (Sx_* columns)
  - Medications (Med_* columns)
  - Vital Signs
  - Lab Results
  - Dates & Times
  - Other

- **Smart Column Discovery**:
  - Fuzzy search across all columns
  - Category-based browsing
  - Recently used columns tracking
  - Column count indicators

- **Advanced Filter Builder**:
  - **AND/OR Logic**: Create complex filter groups with different logical operators
  - **Multiple Conditions**: Add multiple conditions within each filter group
  - **Type-Aware Operators**: Different operators based on column data type:
    - Numeric: equals, not equals, greater than, less than, between, is null, is not null
    - Date/Time: equals, not equals, greater than, less than, between, is null, is not null
    - Text: equals, not equals, contains, not contains, starts with, ends with, in list, is null, is not null

### 📊 Interactive Visualizations
- **Histograms**: For numeric data distribution
- **Bar Charts**: For categorical data analysis
- **Scatter Plots**: For correlation analysis between numeric variables
- **Pie Charts**: For demographic and categorical breakdowns

### 🔐 Security Features
- Password protection for data access
- No data download capabilities
- Text selection prevention
- Secure Snowflake connection

### 🎨 Modern UI/UX
- Dark theme with Van Heron Labs branding
- Responsive design
- Intuitive navigation
- Professional styling

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Snowflake account with proper credentials
- Required Python packages (see requirements.txt)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/thisisarminahmadi/BiobankingAI_ASN_StreamlitDemo.git
cd BiobankingAI_ASN_StreamlitDemo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Snowflake connection in `secrets.toml`:
```toml
[connections.snowflake]
account = "your_account"
user = "your_user"
password = "your_password"
database = "your_database"
schema = "your_schema"
warehouse = "your_warehouse"
role = "your_role"
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

## 📖 Usage Guide

### 1. Column Selection
- Use the search box to find specific columns
- Browse categories to discover related columns
- Click the ➕ button to add columns to your selection

### 2. Building Filters
1. **Add Filter Groups**: Click "➕ Add Filter Group" to create a new filter group
2. **Select Columns**: Choose from your selected columns
3. **Choose Operators**: Select appropriate operators based on data type
4. **Set Values**: Enter filter values (ranges, lists, text, etc.)
5. **Add Conditions**: Click "Add Condition" to add to the group
6. **Apply Logic**: Groups use OR logic between them, conditions within groups use AND logic

### 3. Filter Examples
```
Group 1 (AND logic within group):
- Age > 50
- Gender = "Female"

Group 2 (OR logic between groups):
- Hx_Diabetes = "Yes"
- Sx_Vision_Problems = "Yes"
```

This would find patients who are:
- (Age > 50 AND Gender = "Female") OR (Hx_Diabetes = "Yes") OR (Sx_Vision_Problems = "Yes")

## 🔧 Technical Architecture

### Data Flow
1. **Snowflake Connection**: Secure connection to ASN database
2. **Data Loading**: Cached data loading with type conversion
3. **Column Categorization**: Automatic categorization based on naming patterns
4. **Filter Processing**: Advanced filter engine with AND/OR logic
5. **Visualization**: Interactive charts using Altair

### Performance Optimizations
- Cached data loading
- Efficient filter processing
- Lazy column loading
- Optimized visualization rendering

## 🛠️ Customization

### Adding New Column Categories
Modify the `categorize_columns()` function to add new categories:

```python
def categorize_columns(columns: List[str]) -> Dict[str, List[str]]:
    categories = {
        'Your New Category': [],
        # ... existing categories
    }
    
    for col in columns:
        # Add your categorization logic
        if your_condition:
            categories['Your New Category'].append(col)
```

### Custom Operators
Extend the `create_filter_condition()` function to add new operators for specific data types.

## 🔒 Security Considerations

- All data access is password-protected
- No data export capabilities
- Secure Snowflake connection with encrypted credentials
- Session-based access control

## 📈 Future Enhancements

- [ ] Filter presets and saved queries
- [ ] Advanced statistical analysis
- [ ] Machine learning insights
- [ ] Real-time data updates
- [ ] Collaborative filtering
- [ ] Export capabilities (with proper permissions)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions, please contact the development team or create an issue in the repository.

---

**Built with ❤️ by the BiobankingAI Team** 