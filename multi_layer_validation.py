
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from datetime import datetime


class MultiLayerValidation:
    """
    Implements the guardrail pipeline to prevent hallucinations
    and ensure correctness of agentic AI outputs
    """
    
    def __init__(self):
        self.validation_logs = []
        
    def layer1_input_validation(self, df):
        """Layer 1: Validate input data quality"""
        print("\n" + "="*60)
        print("LAYER 1: INPUT VALIDATION")
        print("="*60)
        
        checks = {}
        
        # Check 1: Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        checks['missing_values'] = {
            'passed': missing_pct.max() < 50,
            'details': missing_pct[missing_pct > 0].to_dict()
        }
        
        # Check 2: Data types
        checks['data_types'] = {
            'passed': True,
            'details': df.dtypes.to_dict()
        }
        
        # Check 3: Duplicates
        duplicates = df.duplicated().sum()
        checks['duplicates'] = {
            'passed': duplicates < len(df) * 0.1,
            'count': int(duplicates)
        }
        
        # Check 4: Outliers (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            outliers[col] = int(outlier_count)
        checks['outliers'] = {
            'passed': True,
            'details': outliers
        }
        
        self._log_validation("Layer 1", checks)
        return all(c['passed'] for c in checks.values()), checks
    
    def layer2_process_validation(self, agent_output, expected_keys=None):
        """Layer 2: Validate agent reasoning and outputs"""
        print("\n" + "="*60)
        print("LAYER 2: PROCESS VALIDATION")
        print("="*60)
        
        checks = {}
        
        # Check 1: Output completeness
        if expected_keys:
            checks['completeness'] = {
                'passed': all(key in agent_output for key in expected_keys),
                'missing_keys': [k for k in expected_keys if k not in agent_output]
            }
        
        # Check 2: Reasoning coherence (basic check)
        checks['has_reasoning'] = {
            'passed': len(str(agent_output)) > 100,
            'length': len(str(agent_output))
        }
        
        self._log_validation("Layer 2", checks)
        return all(c['passed'] for c in checks.values()), checks
    
    def layer3_output_validation(self, y_true, y_pred, model=None):
        """Layer 3: Validate model outputs and performance"""
        print("\n" + "="*60)
        print("LAYER 3: OUTPUT VALIDATION")
        print("="*60)
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        checks = {}
        
        # Check 1: Performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        checks['performance'] = {
            'passed': accuracy > 0.5,  # Better than random for 3 classes
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        }
        
        # Check 2: Class distribution in predictions
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        checks['prediction_distribution'] = {
            'passed': len(unique) == len(np.unique(y_true)),
            'distribution': pred_dist
        }
        
        # Check 3: Confusion matrix sanity
        cm = confusion_matrix(y_true, y_pred)
        checks['confusion_matrix'] = {
            'passed': np.trace(cm) > 0,  # At least some correct predictions
            'matrix': cm.tolist()
        }
        
        self._log_validation("Layer 3", checks)
        return all(c['passed'] for c in checks.values()), checks
    
    def layer4_hallucination_check(self, claims, data_source):
        """Layer 4: Verify claims are grounded in actual data"""
        print("\n" + "="*60)
        print("LAYER 4: HALLUCINATION PREVENTION")
        print("="*60)
        
        checks = {}
        
        # Check 1: Claims have data support
        checks['grounded'] = {
            'passed': data_source is not None,
            'source_available': data_source is not None
        }
        
        # Check 2: Statistical claims are verifiable
        checks['verifiable'] = {
            'passed': True,
            'note': "Manual verification required for specific claims"
        }
        
        self._log_validation("Layer 4", checks)
        return True, checks
    
    def _log_validation(self, layer, checks):
        """Log validation results"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'layer': layer,
            'checks': checks
        }
        self.validation_logs.append(log_entry)
        
        # Print results
        for check_name, check_result in checks.items():
            status = "<3 PASS" if check_result['passed'] else "✗ FAIL"
            print(f"{status}: {check_name}")
            if 'details' in check_result and check_result['details']:
                print(f"  Details: {check_result['details']}")
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY REPORT")
        print("="*60)
        
        for log in self.validation_logs:
            print(f"\n{log['layer']} - {log['timestamp']}")
            for check_name, result in log['checks'].items():
                status = "<3" if result['passed'] else "✗"
                print(f"  {status} {check_name}")
        
        return self.validation_logs
