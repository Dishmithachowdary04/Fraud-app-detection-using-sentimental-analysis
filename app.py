import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from io import StringIO

# Import custom modules
from modules.data_loader import load_data, display_data_preview
from modules.data_preprocessor import preprocess_data, handle_missing_values, encode_categorical_features
from modules.exploratory_analysis import perform_eda, calculate_churn_rate
from modules.statistical_analysis import perform_correlation_analysis, perform_hypothesis_testing
from modules.visualization import plot_churn_distribution, plot_feature_distributions, plot_correlation_heatmap
from modules.modeling import train_model, evaluate_model, get_feature_importance
from modules.segmentation import perform_customer_segmentation, visualize_segments

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Customer Churn Analysis Dashboard")
st.markdown("""
This application helps you analyze customer churn patterns, identify key predictors, 
and visualize insights from your customer data.
""")

# Initialize session state for storing data and analysis results
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'segments' not in st.session_state:
    st.session_state.segments = None

# Sidebar for navigation and data upload
with st.sidebar:
    st.header("Navigation")
    
    # Data Upload Section
    st.subheader("1. Data Upload")
    uploaded_file = st.file_uploader("Upload your customer data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = load_data(uploaded_file)
            st.session_state.data = df
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    # Only show navigation options if data is loaded
    if st.session_state.data is not None:
        st.subheader("2. Analysis Options")
        page = st.radio(
            "Select Analysis",
            ["Data Overview", "Data Preprocessing", "Exploratory Analysis", 
             "Statistical Analysis", "Feature Importance", 
             "Customer Segmentation", "Churn Prediction"]
        )
    else:
        page = "Data Upload"
        st.info("Please upload data to continue.")

# Main content area
if st.session_state.data is not None:
    # Data Overview
    if page == "Data Overview":
        st.header("Data Overview")
        display_data_preview(st.session_state.data)
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(st.session_state.data.describe())
        
        # Display data types and missing values
        st.subheader("Data Types and Missing Values")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Data Types:")
            st.write(st.session_state.data.dtypes)
        
        with col2:
            st.write("Missing Values:")
            missing_values = st.session_state.data.isnull().sum()
            st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
    
    # Data Preprocessing
    elif page == "Data Preprocessing":
        st.header("Data Preprocessing")
        
        st.subheader("Identify Target Column (Churn)")
        target_column = st.selectbox(
            "Select the column that indicates churn status:", 
            st.session_state.data.columns
        )
        
        st.subheader("Handle Missing Values")
        numeric_strategy = st.selectbox(
            "Strategy for numeric columns with missing values:",
            ["mean", "median", "mode", "remove"]
        )
        categorical_strategy = st.selectbox(
            "Strategy for categorical columns with missing values:",
            ["mode", "most_frequent", "new_category", "remove"]
        )
        
        st.subheader("Feature Engineering")
        categorical_columns = st.multiselect(
            "Select categorical columns for encoding:",
            st.session_state.data.select_dtypes(include=['object']).columns
        )
        
        if st.button("Preprocess Data"):
            # First handle missing values
            df_clean = handle_missing_values(
                st.session_state.data, 
                numeric_strategy, 
                categorical_strategy
            )
            
            # Then encode categorical features
            preprocessed_df = encode_categorical_features(df_clean, categorical_columns)
            
            # Final preprocessing steps
            preprocessed_df, _ = preprocess_data(preprocessed_df, target_column)
            
            # Store the preprocessed data
            st.session_state.preprocessed_data = preprocessed_df
            st.session_state.target_column = target_column
            
            st.success("Data preprocessing completed!")
            st.write("Preprocessed Data Preview:")
            st.write(preprocessed_df.head())
    
    # Exploratory Analysis
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        if st.session_state.preprocessed_data is None:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
        else:
            df = st.session_state.preprocessed_data
            target = st.session_state.target_column
            
            # Calculate churn rate
            churn_rate, churn_counts = calculate_churn_rate(df, target)
            
            # Display churn rate
            st.subheader("Overall Churn Rate")
            st.metric("Churn Rate", f"{churn_rate:.2%}")
            
            # Plot churn distribution
            st.subheader("Churn Distribution")
            plot_churn_distribution(df, target)
            
            # Feature distribution analysis
            st.subheader("Feature Distributions by Churn Status")
            
            # Let user select features to visualize
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_features = st.multiselect(
                "Select features to visualize:",
                [col for col in numeric_cols if col != target],
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if selected_features:
                plot_feature_distributions(df, selected_features, target)
    
    # Statistical Analysis
    elif page == "Statistical Analysis":
        st.header("Statistical Analysis")
        
        if st.session_state.preprocessed_data is None:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
        else:
            df = st.session_state.preprocessed_data
            target = st.session_state.target_column
            
            # Correlation Analysis
            st.subheader("Correlation Analysis")
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            
            if numeric_df.shape[1] > 1:
                plot_correlation_heatmap(numeric_df)
                
                # Correlation with target
                if target in numeric_df.columns:
                    st.subheader(f"Correlation with {target}")
                    correlations = numeric_df.corr()[target].sort_values(ascending=False)
                    correlations = correlations[correlations.index != target]
                    
                    fig, ax = plt.subplots(figsize=(10, len(correlations) * 0.3))
                    bars = ax.barh(correlations.index, correlations.values)
                    for i, bar in enumerate(bars):
                        if correlations.values[i] > 0:
                            bar.set_color('blue')
                        else:
                            bar.set_color('red')
                    plt.title(f"Feature Correlation with {target}")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Not enough numeric columns for correlation analysis.")
            
            # Hypothesis Testing
            st.subheader("Hypothesis Testing")
            st.write("Testing the significance of feature relationships with churn:")
            
            results = perform_hypothesis_testing(df, target)
            st.write(results)
    
    # Feature Importance
    elif page == "Feature Importance":
        st.header("Feature Importance Analysis")
        
        if st.session_state.preprocessed_data is None:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
        else:
            df = st.session_state.preprocessed_data
            target = st.session_state.target_column
            
            # Feature importance with a Random Forest model
            st.subheader("Identifying Important Churn Predictors")
            
            if st.button("Calculate Feature Importance"):
                with st.spinner("Training model and calculating feature importance..."):
                    X = df.drop(columns=[target])
                    y = df[target]
                    
                    # Train model and get feature importance
                    model, importance_df = get_feature_importance(X, y)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.feature_importance = importance_df
                
                st.success("Feature importance calculated!")
            
            # Display feature importance if available
            if st.session_state.feature_importance is not None:
                imp_df = st.session_state.feature_importance
                
                # Plot feature importance
                st.subheader("Feature Importance Ranking")
                fig = px.bar(
                    imp_df, 
                    x='importance', 
                    y='feature', 
                    orientation='h',
                    title='Feature Importance',
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                    color='importance'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig)
                
                # Display table with feature importance values
                st.write("Feature Importance Values:")
                st.write(imp_df)
    
    # Customer Segmentation
    elif page == "Customer Segmentation":
        st.header("Customer Segmentation")
        
        if st.session_state.preprocessed_data is None:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
        else:
            df = st.session_state.preprocessed_data
            target = st.session_state.target_column
            
            st.subheader("Segment Customers based on Behavior")
            
            # Parameters for segmentation
            n_clusters = st.slider("Number of customer segments:", 2, 10, 3)
            
            # Features for segmentation
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            default_features = [col for col in numeric_cols if col != target][:min(5, len(numeric_cols))]
            
            segmentation_features = st.multiselect(
                "Select features for segmentation:",
                [col for col in numeric_cols if col != target],
                default=default_features
            )
            
            if st.button("Perform Customer Segmentation") and segmentation_features:
                with st.spinner("Performing customer segmentation..."):
                    segmented_df = perform_customer_segmentation(
                        df, 
                        target, 
                        segmentation_features, 
                        n_clusters
                    )
                    
                    st.session_state.segments = segmented_df
                
                st.success("Customer segmentation completed!")
            
            # Display segmentation results if available
            if st.session_state.segments is not None:
                st.subheader("Customer Segments")
                
                # Basic segment statistics
                segment_stats = st.session_state.segments.groupby('Segment').agg({
                    target: ['mean', 'count']
                })
                segment_stats.columns = ['Churn Rate', 'Count']
                segment_stats = segment_stats.reset_index()
                
                # Show segment statistics
                st.write("Segment Overview:")
                st.write(segment_stats)
                
                # Visualization of segments
                st.subheader("Segment Visualization")
                visualize_segments(st.session_state.segments, target)
                
                # Display segment profiles
                st.subheader("Segment Profiles")
                
                # Calculate mean values for each feature by segment
                profile_cols = segmentation_features + [target]
                segment_profiles = st.session_state.segments.groupby('Segment')[profile_cols].mean()
                
                st.write("Average values for each feature by segment:")
                st.write(segment_profiles)
                
                # Create radar chart or parallel coordinates for segment comparison
                st.subheader("Segment Comparison")
                
                # Convert to long format for visualization
                profile_long = segment_profiles.reset_index().melt(
                    id_vars=['Segment'],
                    value_vars=profile_cols,
                    var_name='Feature',
                    value_name='Value'
                )
                
                # Create parallel coordinates plot
                fig = px.parallel_coordinates(
                    profile_long, 
                    color='Segment',
                    labels={'Segment': 'Segment', 'Value': 'Average Value', 'Feature': 'Customer Attribute'},
                    title='Customer Segment Profiles Comparison',
                    color_continuous_scale=px.colors.diverging.Tealrose
                )
                st.plotly_chart(fig)
    
    # Churn Prediction
    elif page == "Churn Prediction":
        st.header("Churn Prediction")
        
        if st.session_state.preprocessed_data is None:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
        else:
            df = st.session_state.preprocessed_data
            target = st.session_state.target_column
            
            # Create tabs for training and prediction
            train_tab, predict_tab, explain_tab = st.tabs(["Train Model", "Predict Churn", "Model Explanation"])
            
            with train_tab:
                st.subheader("Train Churn Prediction Model")
                
                # Model selection
                model_type = st.selectbox(
                    "Select model type:",
                    ["Random Forest", "Logistic Regression", "Gradient Boosting"]
                )
                
                # Training parameters
                test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
                
                # Train model button
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        X = df.drop(columns=[target])
                        y = df[target]
                        
                        # Train the model
                        model, accuracy, precision, recall, f1, auc, conf_matrix = train_model(
                            X, y, model_type, test_size
                        )
                        
                        # Store model and metrics in session state
                        st.session_state.churn_model = model
                        st.session_state.model_metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc,
                            'conf_matrix': conf_matrix
                        }
                    
                    st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                
                # Display model metrics if available
                if 'model_metrics' in st.session_state:
                    metrics = st.session_state.model_metrics
                    
                    st.subheader("Model Evaluation Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    col2.metric("Precision", f"{metrics['precision']:.2%}")
                    col3.metric("Recall", f"{metrics['recall']:.2%}")
                    col4.metric("F1 Score", f"{metrics['f1']:.2%}")
                    
                    st.subheader("ROC AUC Score")
                    st.metric("AUC", f"{metrics['auc']:.2%}")
                    
                    st.subheader("Confusion Matrix")
                    conf_matrix = metrics['conf_matrix']
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                              xticklabels=['Predicted No Churn', 'Predicted Churn'],
                              yticklabels=['Actual No Churn', 'Actual Churn'])
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with predict_tab:
                st.subheader("Predict Customer Churn")
                
                if 'churn_model' not in st.session_state:
                    st.warning("Please train a model in the 'Train Model' tab first.")
                else:
                    st.write("Enter customer information to predict churn probability:")
                    
                    # Get feature inputs
                    X = df.drop(columns=[target])
                    features = X.columns
                    
                    # Create a form for user input
                    with st.form("prediction_form"):
                        # Create columns for a more compact form
                        cols = st.columns(3)
                        
                        # Dictionary to store input values
                        input_values = {}
                        
                        # Generate input fields for each feature
                        for i, feature in enumerate(features):
                            col_idx = i % 3
                            feature_type = X[feature].dtype
                            
                            if feature_type in ['int64', 'float64']:
                                min_val = float(X[feature].min())
                                max_val = float(X[feature].max())
                                mean_val = float(X[feature].mean())
                                
                                if feature_type == 'int64':
                                    input_values[feature] = cols[col_idx].number_input(
                                        f"{feature}:", 
                                        min_value=int(min_val),
                                        max_value=int(max_val),
                                        value=int(mean_val),
                                        step=1
                                    )
                                else:
                                    input_values[feature] = cols[col_idx].number_input(
                                        f"{feature}:", 
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=mean_val,
                                        format="%.2f"
                                    )
                            else:
                                # For categorical features, create a selectbox
                                unique_values = X[feature].unique().tolist()
                                input_values[feature] = cols[col_idx].selectbox(
                                    f"{feature}:", 
                                    unique_values,
                                    key=f"select_{feature}"
                                )
                        
                        # Submit button
                        submitted = st.form_submit_button("Predict Churn")
                    
                    if submitted:
                        # Create input DataFrame for prediction
                        input_df = pd.DataFrame([input_values])
                        
                        # Predict
                        prediction = st.session_state.churn_model.predict(input_df)[0]
                        prediction_proba = st.session_state.churn_model.predict_proba(input_df)[0][1]
                        
                        # Display prediction
                        st.subheader("Churn Prediction Result")
                        
                        # Create gauge chart for probability visualization
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prediction_proba * 100,
                            title = {'text': "Churn Probability"},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred" if prediction_proba > 0.5 else "green"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "lightyellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "#ff6666"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig)
                        
                        if prediction == 1:
                            st.error(f"⚠️ This customer is likely to churn (Probability: {prediction_proba:.2%})")
                            st.write("Consider taking proactive retention actions for this customer.")
                        else:
                            st.success(f"✅ This customer is likely to stay (Probability of not churning: {1-prediction_proba:.2%})")
                            st.write("This customer appears stable, but continued engagement is recommended.")
            
            with explain_tab:
                st.subheader("Model Explanation")
                
                if 'churn_model' not in st.session_state:
                    st.warning("Please train a model in the 'Train Model' tab first.")
                else:
                    # Get feature importance
                    if 'feature_importance' not in st.session_state:
                        with st.spinner("Calculating feature importance..."):
                            X = df.drop(columns=[target])
                            y = df[target]
                            _, feature_importance_df = get_feature_importance(X, y)
                            st.session_state.feature_importance = feature_importance_df
                    
                    # Display feature importance
                    if 'feature_importance' in st.session_state:
                        imp_df = st.session_state.feature_importance
                        
                        st.write("### Key Factors Influencing Churn")
                        st.write("The chart below shows which customer attributes most strongly predict churn behavior.")
                        
                        # Plot feature importance with Plotly
                        fig = px.bar(
                            imp_df.head(10), 
                            x='importance', 
                            y='feature', 
                            orientation='h',
                            title='Top 10 Churn Predictors',
                            labels={'importance': 'Importance Score', 'feature': 'Customer Attribute'},
                            color='importance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig)
                        
                        # Interpretation guidance
                        st.write("### Interpretation Guide")
                        st.write("- **Higher importance scores** indicate stronger influence on churn prediction")
                        st.write("- **Business Context:** Features like contract type, tenure, and monthly charges typically have significant impact on churn")
                        st.write("- **Application:** Focus retention efforts on addressing issues related to the most important factors")
else:
    # Initial page when no data is loaded
    st.header("Welcome to the Customer Churn Analysis Dashboard")
    
    st.markdown("""
    ### Upload your customer data to get started
    
    This application helps you:
    - Analyze customer churn patterns
    - Identify key churn predictors
    - Visualize important insights
    - Segment customers based on behavior
    - Build predictive models for churn
    
    #### Data Format Requirements:
    - CSV file format
    - One row per customer
    - Must include a column indicating churn status (1 for churned, 0 for not churned)
    - Other columns should represent customer attributes and behaviors
    
    #### Example Columns:
    - Customer demographics (age, gender, etc.)
    - Account information (tenure, contract type, etc.)
    - Usage metrics (frequency of use, spending, etc.)
    - Customer service interactions
    - And a target column indicating whether the customer has churned
    
    Use the upload button in the sidebar to begin your analysis.
    """)
    
    # Example Dataset Structure
    st.subheader("Example Dataset Structure")
    
    example_data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Age': [42, 35, 58, 29, 47],
        'Tenure': [24, 6, 36, 12, 18],
        'MonthlyCharges': [89.50, 65.30, 112.45, 75.60, 98.20],
        'TotalCharges': [2156.0, 391.8, 4048.2, 907.2, 1767.6],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Contract': ['Month-to-month', 'Month-to-month', 'One year', 'Month-to-month', 'Two year'],
        'Churn': [0, 1, 0, 1, 0]
    }
    
    st.dataframe(pd.DataFrame(example_data))