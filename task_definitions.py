from crewai import Task


def _compose_data_context(data_context, data_path):
    """Create a readable data snapshot string shared with all agents."""
    sections = []
    labels = [
        ("Dataset Overview", "overview"),
        ("Sample Rows", "sample_rows"),
        ("Numeric Summary", "numeric_summary"),
        ("Categorical Summary", "categorical_summary"),
        ("Missing Values", "missing_summary"),
        ("Top Correlations", "correlations"),
    ]
    for label, key in labels:
        payload = data_context.get(key)
        if payload:
            sections.append(f"{label}:\n{payload}")
    joined = "\n\n".join(sections)
    return (
        f"Grounded data snapshot generated directly from {data_path}.\n"
        f"Use these statistics as the single source of truth.\n\n{joined}"
    )


def create_tasks(agents, data_path, data_context):
    """Create tasks for the crew"""

    shared_context = _compose_data_context(data_context, data_path)
    
    # Task 1: Data Analysis
    task1 = Task(
        description=f"""
        Analyze the Zomato restaurant dataset at: {data_path}
        
        Your analysis must include:
        1. Load and inspect the data structure
        2. Identify the target variable (Aggregate rating → High/Medium/Low categories)
        3. Analyze feature distributions and relationships
        4. Identify data quality issues (missing values, outliers, duplicates)
        5. Discover patterns related to restaurant success
        6. Perform statistical tests to validate findings
        
        VALIDATION REQUIREMENTS:
        - All statistics must be calculated from actual data
        - Include confidence intervals where appropriate
        - Note any assumptions or limitations
        - Provide evidence for all claims
        
        Output a comprehensive EDA report with visualizations and statistical evidence.

        DATA CONTEXT (derived from the actual CSV):
        {shared_context}
        """,
        agent=agents['data_analyst'],
        expected_output="Detailed exploratory data analysis report with validated statistical findings"
    )
    
    # Task 2: Feature Engineering
    task2 = Task(
        description=f"""
        Based on the EDA findings, engineer features for restaurant success prediction.
        
        Create features that capture:
        1. Service capabilities (online delivery, table booking)
        2. Cuisine diversity and popularity
        3. Price positioning and competitiveness
        4. Geographic factors (city, locality clustering)
        5. Customer engagement (votes as proxy for visits)
        6. Feature interactions that make business sense
        
        VALIDATION REQUIREMENTS:
        - Justify each feature with business logic
        - Check for multicollinearity
        - Validate feature importance with statistical tests
        - Ensure no data leakage
        
        Document your feature engineering process and rationale.

        DATA CONTEXT (derived from the actual CSV):
        {shared_context}
        """,
        agent=agents['feature_engineer'],
        expected_output="Feature engineering report with created features, validation metrics, and rationale",
        context=[task1]
    )
    
    # Task 3: Model Building
    task3 = Task(
        description=f"""
        Build classification models to predict restaurant rating category (High/Medium/Low).
        
        Requirements:
        1. Create the target variable from Aggregate rating
        2. Split data properly (train/validation/test)
        3. Try multiple algorithms (Random Forest, XGBoost, Gradient Boosting)
        4. Use cross-validation for robust evaluation
        5. Optimize hyperparameters
        6. Compare models on multiple metrics (accuracy, F1, precision, recall)
        7. Generate SHAP values for interpretability
        
        VALIDATION REQUIREMENTS:
        - Report metrics on held-out test set
        - Check for overfitting
        - Validate model stability across multiple runs
        - Ensure predictions are sensible
        
        Provide comprehensive model evaluation and comparison.

        DATA CONTEXT (derived from the actual CSV):
        {shared_context}
        """,
        agent=agents['model_builder'],
        expected_output="Model building report with performance metrics, comparison, and best model selection",
        context=[task2]
    )
    
    # Task 4: Validation
    task4 = Task(
        description=f"""
        Validate all previous outputs for correctness and accuracy.
        
        Validation checklist:
        1. Verify EDA statistics against raw data
        2. Check feature engineering logic and calculations
        3. Validate model performance claims
        4. Test for fairness across geographic locations
        5. Ensure no hallucinations in insights
        6. Verify all claims are grounded in data
        
        CRITICAL: Flag any unsupported claims or statistical errors.
        
        Produce a validation report with pass/fail status for each check.

        DATA CONTEXT (derived from the actual CSV):
        {shared_context}
        """,
        agent=agents['validator'],
        expected_output="Comprehensive validation report with verification of all claims",
        context=[task1, task2, task3]
    )
    
    # Task 5: Insight Generation
    task5 = Task(
        description=f"""
        Generate actionable business insights from the analysis and models.
        
        Your insights should answer:
        1. What factors most strongly predict restaurant success?
        2. What can struggling restaurants do to improve?
        3. What patterns distinguish high-performing restaurants?
        4. Are there geographic or cuisine-specific patterns?
        5. What specific recommendations can you make?
        
        VALIDATION REQUIREMENTS:
        - Every insight must cite supporting evidence
        - Include confidence levels for recommendations
        - Acknowledge limitations and uncertainties
        
        Create a business-ready insights report.

        DATA CONTEXT (derived from the actual CSV):
        {shared_context}
        """,
        agent=agents['insight_generator'],
        expected_output="Business insights report with actionable recommendations grounded in validated findings",
        context=[task3, task4]
    )
    
    return [task1, task2, task3, task4, task5]