import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleToxicityDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_simple_descriptors(self, smiles):
        """Calculate simple molecular descriptors from SMILES string"""
        try:
            if not smiles or not isinstance(smiles, str):
                return None
            
            # Clean SMILES
            smiles_upper = smiles.upper()
            
            descriptors = {
                'length': len(smiles),
                'carbon_count': smiles_upper.count('C'),
                'oxygen_count': smiles_upper.count('O'),
                'nitrogen_count': smiles_upper.count('N'),
                'sulfur_count': smiles_upper.count('S'),
                'phosphorus_count': smiles_upper.count('P'),
                'fluorine_count': smiles_upper.count('F'),
                'chlorine_count': smiles_upper.count('CL'),
                'bromine_count': smiles_upper.count('BR'),
                'iodine_count': smiles_upper.count('I'),
                'ring_count': sum(1 for c in smiles if c.isdigit()),
                'aromatic_count': sum(1 for c in smiles if c in 'cnops'),
                'double_bond_count': smiles.count('='),
                'triple_bond_count': smiles.count('#'),
                'branch_count': smiles.count('('),
                'charge_count': smiles.count('+') + smiles.count('-'),
                'hydroxyl_count': smiles_upper.count('OH'),
                'carbonyl_count': smiles.count('C=O') + smiles.count('O=C')
            }
            
            # Additional calculated features
            total_atoms = (descriptors['carbon_count'] + descriptors['oxygen_count'] + 
                          descriptors['nitrogen_count'] + descriptors['sulfur_count'] +
                          descriptors['phosphorus_count'])
            
            if total_atoms > 0:
                descriptors['oxygen_ratio'] = descriptors['oxygen_count'] / total_atoms
                descriptors['nitrogen_ratio'] = descriptors['nitrogen_count'] / total_atoms
                descriptors['halogen_ratio'] = (descriptors['fluorine_count'] + 
                                               descriptors['chlorine_count'] + 
                                               descriptors['bromine_count']) / total_atoms
                descriptors['complexity'] = descriptors['length'] / max(total_atoms, 1)
                descriptors['heteroatom_ratio'] = (total_atoms - descriptors['carbon_count']) / total_atoms
            else:
                descriptors['oxygen_ratio'] = 0
                descriptors['nitrogen_ratio'] = 0
                descriptors['halogen_ratio'] = 0
                descriptors['complexity'] = descriptors['length']
                descriptors['heteroatom_ratio'] = 0
                
            return descriptors
        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            return None
    
    def create_sample_data(self, n_samples=1000):
        """Create sample training data with known patterns"""
        
        # Sample compounds with their toxicity labels
        known_compounds = [
            # Toxic compounds (label = 1)
            ('CCO', 1),                                    # Ethanol
            ('CC(=O)Oc1ccccc1C(=O)O', 1),                # Aspirin
            ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 1),         # Caffeine
            ('c1ccccc1', 1),                              # Benzene
            ('CCl4', 1),                                  # Carbon tetrachloride
            ('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 1),       # Ibuprofen
            ('COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1', 1),
            
            # Safe compounds (label = 0)
            ('OC1C(O)C(O)C(CO)OC1O', 0),                 # Glucose
            ('CC(C)(C)C(=O)O', 0),                       # Simple organic acid
            ('CCCCCCCCCCCCCCCC(=O)O', 0),                # Fatty acid
            ('NC(CC(=O)O)C(=O)O', 0),                    # Aspartic acid
            ('NC(CCC(=O)O)C(=O)O', 0),                   # Glutamic acid
            ('CC(C)CCCC(C)CCCC(C)CCCC(C)C', 0),         # Vitamin E derivative
            ('OC(CO)(CO)CO', 0),                         # Glucose derivative
        ]
        
        np.random.seed(42)
        data = []
        
        # Process known compounds
        for smiles, toxicity in known_compounds:
            descriptors = self.calculate_simple_descriptors(smiles)
            if descriptors:
                descriptors['toxicity'] = toxicity
                data.append(descriptors)
        
        # Generate synthetic variations
        base_toxic = [comp for comp, tox in known_compounds if tox == 1]
        base_safe = [comp for comp, tox in known_compounds if tox == 0]
        
        # Add variations for toxic compounds
        for _ in range((n_samples - len(known_compounds)) // 2):
            base_smiles = np.random.choice(base_toxic)
            descriptors = self.calculate_simple_descriptors(base_smiles)
            if descriptors:
                # Add noise to simulate molecular variations
                for key in descriptors:
                    noise = np.random.normal(0, 0.2)
                    descriptors[key] = max(0, descriptors[key] + noise)
                descriptors['toxicity'] = 1
                data.append(descriptors)
        
        # Add variations for safe compounds
        for _ in range((n_samples - len(known_compounds)) // 2):
            base_smiles = np.random.choice(base_safe)
            descriptors = self.calculate_simple_descriptors(base_smiles)
            if descriptors:
                # Add noise to simulate molecular variations
                for key in descriptors:
                    noise = np.random.normal(0, 0.2)
                    descriptors[key] = max(0, descriptors[key] + noise)
                descriptors['toxicity'] = 0
                data.append(descriptors)
        
        return pd.DataFrame(data)
    
    def train_model(self, data):
        """Train the toxicity detection model"""
        # Separate features and target
        X = data.drop('toxicity', axis=1)
        y = data['toxicity']
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Print evaluation metrics
        print("Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Toxic']))
        
        # Feature importance
        self.plot_feature_importance()
        
        return X_test, y_test, y_pred
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        try:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance for Toxicity Prediction')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    
    def predict_toxicity(self, smiles):
        """Predict toxicity for a given SMILES string"""
        descriptors = self.calculate_simple_descriptors(smiles)
        if descriptors is None:
            return None, "Invalid SMILES string"
        
        # Convert to DataFrame and ensure same feature order
        features_df = pd.DataFrame([descriptors])
        features_df = features_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'Toxic' if prediction == 1 else 'Safe',
            'confidence': max(probability),
            'probabilities': {
                'safe': probability[0],
                'toxic': probability[1]
            },
            'descriptors': descriptors
        }
        
        return result, None
    
    def analyze_compound(self, smiles, compound_name="Unknown"):
        """Detailed analysis of a compound"""
        result, error = self.predict_toxicity(smiles)
        
        if error:
            print(f"Error analyzing {compound_name}: {error}")
            return
        
        print(f"\n=== Analysis for {compound_name} ===")
        print(f"SMILES: {smiles}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Safe probability: {result['probabilities']['safe']:.3f}")
        print(f"Toxic probability: {result['probabilities']['toxic']:.3f}")
        
        print("\nKey Molecular Features:")
        important_features = ['length', 'carbon_count', 'oxygen_count', 'nitrogen_count', 
                            'ring_count', 'aromatic_count', 'complexity']
        for feature in important_features:
            if feature in result['descriptors']:
                print(f"  {feature}: {result['descriptors'][feature]:.2f}")


# Main execution
if __name__ == "__main__":
    print("Simple Drug Toxicity Detector")
    print("=" * 50)
    print("Note: This version uses simplified molecular descriptors")
    print("For better accuracy, install RDKit for advanced descriptors")
    
    # Initialize the detector
    detector = SimpleToxicityDetector()
    
    # Create sample training data
    print("\nCreating sample training data...")
    training_data = detector.create_sample_data(n_samples=800)
    print(f"Created {len(training_data)} training samples")
    
    # Train the model
    print("\nTraining the model...")
    X_test, y_test, y_pred = detector.train_model(training_data)
    
    # Test with some example compounds
    print("\n" + "=" * 50)
    print("Testing with example compounds:")
    
    test_compounds = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Glucose", "OC1C(O)C(O)C(CO)OC1O"),
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Water", "O"),
        ("Methane", "C"),
    ]
    
    for name, smiles in test_compounds:
        detector.analyze_compound(smiles, name)
    
    print("\n" + "=" * 50)
    print("Interactive Mode: Enter SMILES strings to analyze")
    print("Type 'quit' to exit")
    print("\nTry these examples:")
    print("  CCO (ethanol)")
    print("  c1ccccc1 (benzene)")
    print("  CC(=O)O (acetic acid)")
    
    while True:
        user_input = input("\nEnter SMILES string: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            detector.analyze_compound(user_input)
        else:
            print("Please enter a valid SMILES string or 'quit' to exit")
