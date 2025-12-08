import os
import json
import pandas as pd
import numpy as np

from dotenv import load_dotenv

from crewai import Crew, Process
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# get validator
from multi_layer_validation import MultiLayerValidation

# create agents
from agent_definitions import create_agents

# implement tasks
from task_definitions import create_tasks

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')
model_name = os.getenv('OPENAI_MODEL_NAME')


def build_data_context(df: pd.DataFrame) -> dict:
    """Create structured dataset snippets for agents and validators."""
    overview = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
    }
    numeric_summary = df.select_dtypes(include=[np.number]).describe().round(3).to_dict()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    categorical_summary = {
        col: df[col].value_counts().head(10).to_dict() for col in categorical_cols
    }
    missing = df.isnull().sum()
    missing_summary = {col: int(val) for col, val in missing[missing > 0].items()}
    correlation_series = (
        df.select_dtypes(include=[np.number])
        .corrwith(df["Aggregate rating"], method="pearson")
        .dropna()
        .sort_values(ascending=False)
        .head(10)
        .round(3)
    )
    correlations = correlation_series.to_dict()
    sample_rows = df.head(5).to_dict(orient="records")

    def _pretty(payload):
        return json.dumps(payload, indent=2, default=str)

    return {
        "overview": _pretty(overview),
        "numeric_summary": _pretty(numeric_summary),
        "categorical_summary": _pretty(categorical_summary),
        "missing_summary": _pretty(missing_summary) if missing_summary else "None",
        "correlations": _pretty(correlations),
        "sample_rows": _pretty(sample_rows),
    }


def train_baseline_model(df: pd.DataFrame):
    """Train a lightweight baseline model for validation layer 3."""
    feature_df = df.drop(columns=["Rating_Category"])
    target = df["Rating_Category"].astype(str)

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return y_test, predictions

# -------main-------
def main():
    print("="*60)
    print("Restaurant Success Prediction Utilizing Agentic AI")
    print("="*60)
    
    # initialize validation system
    validator = MultiLayerValidation()
    
    # load data
    print("\nLoading dataset...")
    data_path = "zomato.csv"  # Update with your path
    
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
        print(f"<3 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ Error: {data_path} not found. Please download the dataset.")
        return
    
    # Layer 1: Input Validation
    passed, checks = validator.layer1_input_validation(df)
    if not passed:
        print("\n⚠ Warning: Some input validation checks failed. Review and proceed with caution.")
    
    # Create target variable
    print("\nCreating target variable...")
    df['Rating_Category'] = pd.cut(
        df['Aggregate rating'],
        bins=[0, 2.5, 4.0, 5.0],
        labels=['Low', 'Medium', 'High']
    )
    df = df.dropna(subset=['Rating_Category'])
    print(f"<3 Target variable created")
    print(f"  Distribution: {df['Rating_Category'].value_counts().to_dict()}")

    # Build structured context for agents
    print("\nPreparing grounded data context for agents...")
    data_context = build_data_context(df)
    print("<3 Data context prepared and will be shared with every agent")
    
    # Create agents
    print("\nInitializing AI agents...")
    agents = create_agents()
    print(f"<3 Created {len(agents)} specialized agents")
    
    # Create tasks
    print("\nDefining tasks...")
    tasks = create_tasks(agents, data_path, data_context)
    print(f"<3 Created {len(tasks)} sequential tasks")
    
    # Create and run crew
    print("\nAssembling crew...")
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("STARTING AGENTIC WORKFLOW")
    print("="*60)
    print("\nNote: This may take several minutes depending on your LLM setup...")
    print("The agents will work sequentially through all tasks.\n")
    
    # Execute the crew
    try:
        result = crew.kickoff()
        
        print("\n" + "="*60)
        print("WORKFLOW COMPLETED")
        print("="*60)
        print("\nFinal Result:")
        print(result)
        
        # Layer 2: Process validation on structured outputs
        structured_result = {
            "final_report": result,
            "data_evidence": data_context,
        }
        validator.layer2_process_validation(
            structured_result,
            expected_keys=["final_report", "data_evidence"],
        )

        # Layer 3: Output validation using baseline model
        try:
            y_true, y_pred = train_baseline_model(df)
            validator.layer3_output_validation(y_true, y_pred)
        except Exception as model_error:
            print(f"\n⚠ Baseline model validation skipped: {model_error}")

        # Layer 4: Hallucination prevention leveraging data context
        validator.layer4_hallucination_check(result, data_context)

        # Generate validation report
        validator.generate_report()
        
        print("\n<3 Assignment completed successfully!")
        print("Check the output above for detailed results from each agent.")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        print("This might be due to LLM configuration. Check your API keys and endpoints.")
    
if __name__ == "__main__":
    main()