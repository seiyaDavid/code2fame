import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import optuna
import io
import torch


# Define all functions first
def create_metadata_from_data(data):
    """Create a metadata object from the given data"""
    try:
        # Create metadata object
        metadata = SingleTableMetadata()

        # Detect field types
        metadata.detect_from_dataframe(data)

        # Verify all columns are in the metadata
        data_columns = set(data.columns)
        metadata_columns = set(metadata.columns)

        missing_columns = data_columns - metadata_columns
        if missing_columns:
            st.warning(
                f"Adding missing columns to metadata: {', '.join(missing_columns)}"
            )

            # For each missing column, add it to the metadata
            for column in missing_columns:
                # Determine column type
                if pd.api.types.is_numeric_dtype(data[column]):
                    sdtype = "numerical"
                elif pd.api.types.is_datetime64_any_dtype(data[column]):
                    sdtype = "datetime"
                else:
                    sdtype = "categorical"

                # Add column to metadata
                metadata.update_column(column, sdtype=sdtype)

        # Double-check all columns are now in metadata
        if set(data.columns) != set(metadata.columns):
            st.error("Failed to add all columns to metadata")
            return None

        return metadata
    except Exception as e:
        st.error(f"Error creating metadata: {str(e)}")
        return None


def calculate_privacy_metrics(original_data, synthetic_data):
    """Calculate privacy metrics between original and synthetic data"""
    metrics = {}

    # Calculate uniqueness
    orig_uniqueness = len(original_data.drop_duplicates()) / len(original_data)
    synth_uniqueness = len(synthetic_data.drop_duplicates()) / len(synthetic_data)

    metrics["orig_uniqueness"] = orig_uniqueness
    metrics["synth_uniqueness"] = synth_uniqueness

    # Calculate exact matches
    merged = pd.merge(original_data, synthetic_data, how="inner")
    exact_matches = len(merged)
    exact_match_pct = (
        exact_matches / len(synthetic_data) if len(synthetic_data) > 0 else 0
    )

    metrics["exact_matches"] = exact_matches
    metrics["exact_match_pct"] = exact_match_pct

    # Calculate nearest neighbor distance for numerical columns
    numerical_cols = original_data.select_dtypes(include=["number"]).columns

    if len(numerical_cols) > 0:
        # Get numerical data
        orig_numeric = original_data[numerical_cols].fillna(0)
        synth_numeric = synthetic_data[numerical_cols].fillna(0)

        # Scale the data
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(orig_numeric)
        synth_scaled = scaler.transform(synth_numeric)

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(orig_scaled)
        distances, _ = nn.kneighbors(synth_scaled)

        metrics["avg_min_distance"] = float(distances.mean())
        metrics["min_distance"] = float(distances.min())
        metrics["max_distance"] = float(distances.max())
    else:
        metrics["avg_min_distance"] = "N/A"
        metrics["min_distance"] = "N/A"
        metrics["max_distance"] = "N/A"

    return metrics


def optimize_model_parameters(model_type, data, metadata, n_trials=10):
    """Use Optuna to find the best hyperparameters for the model"""

    # Check if metadata is None and create it if needed
    if metadata is None:
        metadata = create_metadata_from_data(data)
        if metadata is None:
            st.error("Failed to create metadata from data. Cannot optimize parameters.")
            return {}

    # Get table name from metadata
    table_name = "data"

    def objective(trial):
        # Common parameters
        epochs = trial.suggest_int("epochs", 50, 300)

        # For CTGAN and CopulaGAN, batch size must be divisible by PAC (default=10)
        # Use batch sizes that are multiples of 10 to avoid the assertion error
        if model_type in ["CTGAN", "CopulaGAN"]:
            batch_size = trial.suggest_categorical("batch_size", [60, 120, 240, 480])
        else:
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        embedding_dim = trial.suggest_int("embedding_dim", 64, 256)

        # Model-specific parameters
        if model_type in ["CTGAN", "CopulaGAN"]:
            # Generator dimensions
            n_layers_g = trial.suggest_int("n_layers_g", 1, 3)
            generator_dim = [
                trial.suggest_int(f"g_dim_{i}", 64, 512) for i in range(n_layers_g)
            ]

            # Discriminator dimensions
            n_layers_d = trial.suggest_int("n_layers_d", 1, 3)
            discriminator_dim = [
                trial.suggest_int(f"d_dim_{i}", 64, 512) for i in range(n_layers_d)
            ]

            # Create model with explicit pac parameter
            if model_type == "CTGAN":
                model = CTGANSynthesizer(
                    metadata,
                    epochs=epochs,
                    batch_size=batch_size,
                    embedding_dim=embedding_dim,
                    generator_dim=generator_dim,
                    discriminator_dim=discriminator_dim,
                    pac=10,  # Explicitly set PAC to 10 (default value)
                    verbose=False,
                )
            else:  # CopulaGAN
                model = CopulaGANSynthesizer(
                    metadata,
                    epochs=epochs,
                    batch_size=batch_size,
                    embedding_dim=embedding_dim,
                    generator_dim=generator_dim,
                    discriminator_dim=discriminator_dim,
                    pac=10,  # Explicitly set PAC to 10 (default value)
                    verbose=False,
                )
        else:  # TVAE
            # Encoder dimensions
            n_layers_e = trial.suggest_int("n_layers_e", 1, 3)
            compress_dims = [
                trial.suggest_int(f"e_dim_{i}", 64, 512) for i in range(n_layers_e)
            ]

            # Decoder dimensions
            n_layers_d = trial.suggest_int("n_layers_d", 1, 3)
            decompress_dims = [
                trial.suggest_int(f"d_dim_{i}", 64, 512) for i in range(n_layers_d)
            ]

            # Create model
            model = TVAESynthesizer(
                metadata,
                epochs=epochs,
                batch_size=batch_size,
                embedding_dim=embedding_dim,
                compress_dims=compress_dims,
                decompress_dims=decompress_dims,
                verbose=False,
            )

        # Train the model with a small number of epochs for optimization
        try:
            model.fit(data)

            # Generate synthetic data
            synthetic_data = model.sample(min(1000, len(data)))

            # Evaluate quality
            try:
                quality_report = evaluate_quality(synthetic_data, data, metadata)
                score = quality_report.get_score()
                return score
            except Exception as e:
                print(f"Error in evaluation: {e}")
                return 0.0
        except Exception as e:
            print(f"Error in model training: {e}")
            return 0.0

    # Create and run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params

    # Convert layer dimensions to lists
    if model_type in ["CTGAN", "CopulaGAN"]:
        n_layers_g = best_params.pop("n_layers_g")
        generator_dim = [best_params.pop(f"g_dim_{i}") for i in range(n_layers_g)]
        best_params["generator_dim"] = generator_dim

        n_layers_d = best_params.pop("n_layers_d")
        discriminator_dim = [best_params.pop(f"d_dim_{i}") for i in range(n_layers_d)]
        best_params["discriminator_dim"] = discriminator_dim
    else:  # TVAE
        n_layers_e = best_params.pop("n_layers_e")
        compress_dims = [best_params.pop(f"e_dim_{i}") for i in range(n_layers_e)]
        best_params["compress_dims"] = compress_dims

        n_layers_d = best_params.pop("n_layers_d")
        decompress_dims = [best_params.pop(f"d_dim_{i}") for i in range(n_layers_d)]
        best_params["decompress_dims"] = decompress_dims

    return best_params


def initialize_and_train_model(
    model_type, data, metadata, params, optimize=False, n_trials=10
):
    """Initialize and train the selected model with the given parameters"""
    status_text = st.empty()
    progress_bar = st.progress(0)

    status_text.text("Initializing model...")
    progress_bar.progress(10)

    # Check if metadata is None and create it if needed
    if metadata is None:
        status_text.text("Creating metadata from data...")
        metadata = create_metadata_from_data(data)
        if metadata is None:
            status_text.text("Failed to create metadata. Aborting.")
            return None

    # Verify metadata matches data columns
    data_columns = set(data.columns)
    metadata_columns = set(metadata.columns)

    if data_columns != metadata_columns:
        status_text.text(
            "Metadata doesn't match data columns. Creating new metadata..."
        )
        metadata = create_metadata_from_data(data)
        if metadata is None:
            status_text.text("Failed to create metadata. Aborting.")
            return None

    # Optimize parameters if requested
    if optimize:
        status_text.text("Optimizing hyperparameters with Optuna...")
        progress_bar.progress(20)

        # Run optimization
        best_params = optimize_model_parameters(model_type, data, metadata, n_trials)

        # Update params with optimized values
        params.update(best_params)

        status_text.text(f"Optimization complete. Best parameters: {best_params}")
        progress_bar.progress(40)

    # Parse dimension parameters if provided as strings
    if "generator_dim" in params and isinstance(params["generator_dim"], str):
        params["generator_dim"] = [
            int(x.strip()) for x in params["generator_dim"].split(",")
        ]

    if "discriminator_dim" in params and isinstance(params["discriminator_dim"], str):
        params["discriminator_dim"] = [
            int(x.strip()) for x in params["discriminator_dim"].split(",")
        ]

    if "compress_dims" in params and isinstance(params["compress_dims"], str):
        params["compress_dims"] = [
            int(x.strip()) for x in params["compress_dims"].split(",")
        ]

    if "decompress_dims" in params and isinstance(params["decompress_dims"], str):
        params["decompress_dims"] = [
            int(x.strip()) for x in params["decompress_dims"].split(",")
        ]

    # For CTGAN and CopulaGAN, ensure batch size is divisible by PAC (default=10)
    if model_type in ["CTGAN", "CopulaGAN"] and "batch_size" in params:
        # Round batch size to nearest multiple of 10
        params["batch_size"] = (params["batch_size"] // 10) * 10
        # Ensure it's at least 10
        params["batch_size"] = max(10, params["batch_size"])

    try:
        # Try with the requested model first
        status_text.text(f"Training {model_type} model...")
        progress_bar.progress(50)

        # Initialize the appropriate model
        if model_type == "CTGAN":
            model = CTGANSynthesizer(
                metadata,
                epochs=params.get("epochs", 100),
                batch_size=params.get(
                    "batch_size", 120
                ),  # Default to 120 (multiple of 10)
                embedding_dim=params.get("embedding_dim", 128),
                generator_dim=params.get("generator_dim", [256, 256]),
                discriminator_dim=params.get("discriminator_dim", [256, 256]),
                pac=10,  # Explicitly set PAC to 10
                verbose=True,
            )
        elif model_type == "TVAE":
            model = TVAESynthesizer(
                metadata,
                epochs=params.get("epochs", 100),
                batch_size=params.get("batch_size", 128),
                embedding_dim=params.get("embedding_dim", 128),
                compress_dims=params.get("compress_dims", [128, 128]),
                decompress_dims=params.get("decompress_dims", [128, 128]),
                verbose=True,
            )
        elif model_type == "CopulaGAN":
            model = CopulaGANSynthesizer(
                metadata,
                epochs=params.get("epochs", 100),
                batch_size=params.get(
                    "batch_size", 120
                ),  # Default to 120 (multiple of 10)
                embedding_dim=params.get("embedding_dim", 128),
                generator_dim=params.get("generator_dim", [256, 256]),
                discriminator_dim=params.get("discriminator_dim", [256, 256]),
                pac=10,  # Explicitly set PAC to 10
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train the model
        model.fit(data)
        progress_bar.progress(90)
        status_text.text("Model training complete!")
        return model

    except Exception as e:
        status_text.text(f"Error during training: {str(e)}")
        st.warning(f"Training {model_type} failed: {str(e)}")

        # Try with a simple TVAE model as fallback
        try:
            status_text.text("Trying with a simple TVAE model instead...")

            # Create fresh metadata
            fresh_metadata = create_metadata_from_data(data)
            if fresh_metadata is None:
                status_text.text(
                    "Failed to create metadata for fallback model. Aborting."
                )
                return None

            # Create a simple TVAE model with minimal parameters
            fallback_model = TVAESynthesizer(
                fresh_metadata,
                epochs=20,
                batch_size=128,
                verbose=True,
            )

            # Train the fallback model
            fallback_model.fit(data)
            status_text.text("Fallback model training complete!")
            return fallback_model

        except Exception as e2:
            status_text.text(f"Fallback model also failed: {str(e2)}")
            st.error(
                "Could not train any model. Please try with different data or parameters."
            )
            return None


def generate_synthetic_data(model, n_samples):
    """Generate synthetic data using the trained model"""
    status_text = st.empty()
    progress_bar = st.progress(90)

    status_text.text("Generating synthetic data...")
    synthetic_data = model.sample(n_samples)

    progress_bar.progress(100)
    status_text.text("Synthetic data generation complete!")

    return synthetic_data


def evaluate_data_quality(original_data, synthetic_data, metadata):
    """Evaluate the quality of the synthetic data"""
    try:
        quality_report = evaluate_quality(synthetic_data, original_data, metadata)
        return quality_report
    except Exception as e:
        st.error(f"Error evaluating data quality: {str(e)}")
        return None


# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Set page configuration
st.set_page_config(
    page_title="SDV Synthetic Data Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown(
    """
<style>
/* Risk level styling with darker colors */
.metric-card {
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 20px;
    color: white;
    font-weight: 500;
}
.metric-card h3 {
    margin-top: 0;
    color: white;
}
.high {
    background-color: #1e7d1e;  /* Darker green */
}
.medium {
    background-color: #b86e00;  /* Darker orange */
}
.low {
    background-color: #a30000;  /* Darker red */
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "model" not in st.session_state:
    st.session_state.model = None
if "quality_report" not in st.session_state:
    st.session_state.quality_report = None
if "privacy_metrics" not in st.session_state:
    st.session_state.privacy_metrics = {}

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data & Generation", "Visualization", "Evaluation", "Help"]
)

with tab1:
    st.header("Data Upload and Generation")

    # File upload section
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Select CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data

            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(5))
            st.text(f"Total rows: {len(data)}")

            # Model selection and parameters
            st.subheader("Generate Synthetic Data")

            col1, col2 = st.columns(2)

            with col1:
                model_type = st.selectbox(
                    "Select Model", ["CTGAN", "TVAE", "CopulaGAN"]
                )

                # Add optimization option
                optimize_model = st.checkbox(
                    "Optimize model hyperparameters", value=True
                )

            # Model parameters section
            with col2:
                # Common parameters for all models
                epochs = st.slider(
                    "Training Epochs", min_value=10, max_value=500, value=100, step=10
                )
                batch_size = st.selectbox(
                    "Batch Size", options=[32, 64, 128, 256, 512], index=2
                )
                n_samples = st.slider(
                    "Number of Samples to Generate",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                )

                # Advanced parameters (shown only when optimization is disabled)
                if not optimize_model:
                    with st.expander("Advanced Parameters"):
                        if model_type == "CTGAN":
                            embedding_dim = st.slider(
                                "Embedding Dimension",
                                min_value=32,
                                max_value=256,
                                value=128,
                            )
                            generator_dim = st.text_input(
                                "Generator Dimensions (comma-separated)",
                                value="256,256",
                            )
                            discriminator_dim = st.text_input(
                                "Discriminator Dimensions (comma-separated)",
                                value="256,256",
                            )
                        elif model_type == "TVAE":
                            embedding_dim = st.slider(
                                "Embedding Dimension",
                                min_value=32,
                                max_value=256,
                                value=128,
                            )
                            compress_dims = st.text_input(
                                "Encoder Dimensions (comma-separated)", value="128,128"
                            )
                            decompress_dims = st.text_input(
                                "Decoder Dimensions (comma-separated)", value="128,128"
                            )
                        elif model_type == "CopulaGAN":
                            embedding_dim = st.slider(
                                "Embedding Dimension",
                                min_value=32,
                                max_value=256,
                                value=128,
                            )
                            generator_dim = st.text_input(
                                "Generator Dimensions (comma-separated)",
                                value="256,256",
                            )
                            discriminator_dim = st.text_input(
                                "Discriminator Dimensions (comma-separated)",
                                value="256,256",
                            )

            # Show optimization settings if optimization is enabled
            if optimize_model:
                n_trials = st.slider(
                    "Number of Optimization Trials",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="More trials may find better parameters but will take longer",
                )
                st.session_state.n_trials = n_trials

            # Generate button
            generate_button = st.button("Generate Synthetic Data")
            if generate_button:
                try:
                    # Create fresh metadata for each run to avoid mismatch issues
                    with st.spinner("Creating metadata from data..."):
                        metadata = create_metadata_from_data(st.session_state.data)

                        if metadata is None:
                            st.error(
                                "Failed to create metadata from data. Please check your data format."
                            )
                        else:
                            st.session_state.metadata = metadata

                            # Collect parameters
                            params = {"epochs": epochs, "batch_size": batch_size}

                            # Add model-specific parameters if not optimizing
                            if not optimize_model:
                                if model_type in ["CTGAN", "CopulaGAN"]:
                                    params.update(
                                        {
                                            "embedding_dim": embedding_dim,
                                            "generator_dim": generator_dim,
                                            "discriminator_dim": discriminator_dim,
                                        }
                                    )
                                elif model_type == "TVAE":
                                    params.update(
                                        {
                                            "embedding_dim": embedding_dim,
                                            "compress_dims": compress_dims,
                                            "decompress_dims": decompress_dims,
                                        }
                                    )

                            # Add a placeholder for optimization progress
                            if optimize_model:
                                st.info(
                                    "Optimizing model hyperparameters. This may take a while..."
                                )
                                n_trials = st.session_state.get("n_trials", 10)
                            else:
                                n_trials = 0

                            # Initialize and train model with fresh metadata
                            with st.spinner("Training model..."):
                                model = initialize_and_train_model(
                                    model_type,
                                    st.session_state.data,
                                    metadata,  # Use the fresh metadata
                                    params,
                                    optimize=optimize_model,
                                    n_trials=n_trials,
                                )

                                if model is None:
                                    st.error(
                                        "Failed to train model. Please try with different parameters or data."
                                    )
                                else:
                                    st.session_state.model = model

                                    # Generate synthetic data
                                    with st.spinner("Generating synthetic data..."):
                                        synthetic_data = generate_synthetic_data(
                                            model, n_samples
                                        )
                                        st.session_state.synthetic_data = synthetic_data

                                        # Display synthetic data preview
                                        st.subheader("Synthetic Data Preview")
                                        st.dataframe(synthetic_data.head(5))
                                        st.text(
                                            f"Generated {len(synthetic_data)} rows of synthetic data"
                                        )

                                        # Evaluate quality
                                        with st.spinner("Evaluating data quality..."):
                                            quality_report = evaluate_data_quality(
                                                st.session_state.data,
                                                synthetic_data,
                                                metadata,  # Use the fresh metadata
                                            )
                                            st.session_state.quality_report = (
                                                quality_report
                                            )

                                            # Calculate privacy metrics
                                            privacy_metrics = calculate_privacy_metrics(
                                                st.session_state.data, synthetic_data
                                            )
                                            st.session_state.privacy_metrics = (
                                                privacy_metrics
                                            )

                                            # Success message
                                            st.success(
                                                "Synthetic data generated successfully!"
                                            )

                except Exception as e:
                    st.error(f"Error generating synthetic data: {str(e)}")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)
    else:
        st.info("Please upload a CSV file to get started.")

with tab2:
    st.header("Data Visualization")

    if (
        st.session_state.data is not None
        and st.session_state.synthetic_data is not None
    ):
        # Select visualization type
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Distribution Comparison", "Correlation Heatmap", "Pair Plot"],
        )

        # Select columns for visualization
        numerical_cols = st.session_state.data.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_cols = st.session_state.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if viz_type == "Distribution Comparison":
            # Select column for distribution comparison
            if numerical_cols:
                selected_col = st.selectbox(
                    "Select Column for Distribution", numerical_cols
                )

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # Original data distribution
                sns.histplot(st.session_state.data[selected_col], kde=True, ax=ax[0])
                ax[0].set_title("Original Data")

                # Synthetic data distribution
                sns.histplot(
                    st.session_state.synthetic_data[selected_col], kde=True, ax=ax[1]
                )
                ax[1].set_title("Synthetic Data")

                st.pyplot(fig)
            else:
                st.warning("No numerical columns found for distribution comparison.")

        elif viz_type == "Correlation Heatmap":
            if len(numerical_cols) > 1:
                fig, ax = plt.subplots(1, 2, figsize=(15, 7))

                # Original data correlation
                corr_orig = st.session_state.data[numerical_cols].corr()
                sns.heatmap(
                    corr_orig, annot=True, cmap="coolwarm", ax=ax[0], vmin=-1, vmax=1
                )
                ax[0].set_title("Original Data Correlation")

                # Synthetic data correlation
                corr_synth = st.session_state.synthetic_data[numerical_cols].corr()
                sns.heatmap(
                    corr_synth, annot=True, cmap="coolwarm", ax=ax[1], vmin=-1, vmax=1
                )
                ax[1].set_title("Synthetic Data Correlation")

                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numerical columns for correlation heatmap.")

        elif viz_type == "Pair Plot":
            if len(numerical_cols) > 1:
                # Select columns for pair plot (limit to 3-4 for performance)
                cols_for_plot = st.multiselect(
                    "Select Columns for Pair Plot (2-4 recommended)",
                    numerical_cols,
                    default=numerical_cols[: min(3, len(numerical_cols))],
                )

                if len(cols_for_plot) >= 2:
                    # Create a combined dataframe with a 'source' column
                    orig_df = st.session_state.data[cols_for_plot].copy()
                    orig_df["source"] = "Original"

                    synth_df = st.session_state.synthetic_data[cols_for_plot].copy()
                    synth_df["source"] = "Synthetic"

                    combined_df = pd.concat([orig_df, synth_df])

                    # Create pair plot
                    fig = plt.figure(figsize=(10, 8))
                    g = sns.pairplot(combined_df, hue="source", palette=["blue", "red"])
                    st.pyplot(g.fig)
                else:
                    st.warning("Please select at least 2 columns for the pair plot.")
            else:
                st.warning("Need at least 2 numerical columns for pair plot.")
    else:
        st.info("Please generate synthetic data first to enable visualization.")

with tab3:
    st.header("Data Evaluation")

    if st.session_state.quality_report is not None:
        # Display overall quality score
        score = st.session_state.quality_report.get_score()
        st.subheader("Overall Quality Score")

        # Create a gauge-like visualization for the score
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(0, score, color="green", height=0.5)
        ax.barh(0, 1, color="lightgray", height=0.5, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.text(
            score / 2,
            0,
            f"{score:.2f}",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="white",
        )
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
        ax.set_xlabel("Score (higher is better)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        st.pyplot(fig)

        # Display detailed quality metrics
        st.subheader("Detailed Quality Metrics")

        # Column shape metrics
        st.write("#### Column Shapes")
        column_shapes = st.session_state.quality_report.get_details("Column Shapes")
        column_shape_df = pd.DataFrame(column_shapes).T
        st.dataframe(column_shape_df)

        # Column pair trends
        st.write("#### Column Pair Trends")
        column_pairs = st.session_state.quality_report.get_details("Column Pair Trends")
        if isinstance(column_pairs, pd.DataFrame) and not column_pairs.empty:
            column_pairs_df = pd.DataFrame(column_pairs).T
            st.dataframe(column_pairs_df)
        else:
            st.info("No column pair data available for visualization.")

        # Privacy metrics
        st.subheader("Privacy Metrics")

        # Calculate privacy metrics if not already done
        if not st.session_state.privacy_metrics:
            try:
                # 1. Uniqueness - percentage of unique records
                orig_uniqueness = len(st.session_state.data.drop_duplicates()) / len(
                    st.session_state.data
                )
                synth_uniqueness = len(
                    st.session_state.synthetic_data.drop_duplicates()
                ) / len(st.session_state.synthetic_data)

                # 2. Exact matches - count of synthetic records that exactly match original records
                merged = pd.merge(
                    st.session_state.data, st.session_state.synthetic_data, how="inner"
                )
                exact_matches = len(merged)
                exact_match_pct = exact_matches / len(st.session_state.synthetic_data)

                # 3. Distance-based metric - average minimum distance between synthetic and original records
                # Only use numeric columns for distance calculation
                numeric_cols = [
                    col
                    for col in st.session_state.data.columns
                    if pd.api.types.is_numeric_dtype(st.session_state.data[col])
                ]

                if numeric_cols:
                    # Normalize data for distance calculation
                    scaler = StandardScaler()

                    orig_numeric = st.session_state.data[numeric_cols].fillna(0)
                    synth_numeric = st.session_state.synthetic_data[
                        numeric_cols
                    ].fillna(0)

                    orig_scaled = scaler.fit_transform(orig_numeric)
                    synth_scaled = scaler.transform(synth_numeric)

                    # Find nearest neighbors
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(orig_scaled)
                    distances, _ = nn.kneighbors(synth_scaled)

                    avg_min_distance = distances.mean()
                    min_distance = distances.min()
                    max_distance = distances.max()
                else:
                    avg_min_distance = "N/A"
                    min_distance = "N/A"
                    max_distance = "N/A"

                # Store metrics in session state
                st.session_state.privacy_metrics = {
                    "orig_uniqueness": orig_uniqueness,
                    "synth_uniqueness": synth_uniqueness,
                    "exact_matches": exact_matches,
                    "exact_match_pct": exact_match_pct,
                    "avg_min_distance": avg_min_distance,
                    "min_distance": min_distance,
                    "max_distance": max_distance,
                }

            except Exception as e:
                st.error(f"Error calculating privacy metrics: {str(e)}")

        # Display privacy metrics if available
        if st.session_state.privacy_metrics:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Original Data Uniqueness",
                    f"{st.session_state.privacy_metrics['orig_uniqueness']:.2%}",
                )
                st.metric(
                    "Synthetic Data Uniqueness",
                    f"{st.session_state.privacy_metrics['synth_uniqueness']:.2%}",
                )

            with col2:
                st.metric(
                    "Exact Matches", st.session_state.privacy_metrics["exact_matches"]
                )
                st.metric(
                    "Exact Match Percentage",
                    f"{st.session_state.privacy_metrics['exact_match_pct']:.2%}",
                )

            with col3:
                if isinstance(
                    st.session_state.privacy_metrics["avg_min_distance"], float
                ):
                    st.metric(
                        "Avg. Min Distance",
                        f"{st.session_state.privacy_metrics['avg_min_distance']:.4f}",
                    )
                    st.metric(
                        "Min Distance",
                        f"{st.session_state.privacy_metrics['min_distance']:.4f}",
                    )
                else:
                    st.metric("Avg. Min Distance", "N/A")
                    st.metric("Min Distance", "N/A")

            # Privacy risk assessment
            st.subheader("Privacy Risk Assessment")

            # Determine risk level based on metrics
            if isinstance(st.session_state.privacy_metrics["exact_match_pct"], float):
                if st.session_state.privacy_metrics["exact_match_pct"] > 0.05:
                    risk_level = "High"
                    risk_class = "low"  # CSS class (inverted for color coding)
                    risk_description = "The synthetic data contains a significant number of exact matches with the original data, which may pose privacy risks."
                elif (
                    isinstance(
                        st.session_state.privacy_metrics["avg_min_distance"], float
                    )
                    and st.session_state.privacy_metrics["avg_min_distance"] < 0.1
                ):
                    risk_level = "Medium"
                    risk_class = "medium"
                    risk_description = "The synthetic data points are very close to the original data, which may allow for some inference of the original data."
                else:
                    risk_level = "Low"
                    risk_class = "high"  # CSS class (inverted for color coding)
                    risk_description = "The synthetic data appears to be sufficiently different from the original data, with low privacy risk."

                st.markdown(
                    f"""
                <div class="metric-card {risk_class}">
                    <h3>Risk Level: {risk_level}</h3>
                    <p>{risk_description}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Unable to assess privacy risk due to missing metrics.")
    else:
        st.info("Please generate synthetic data first to enable evaluation.")

with tab4:
    st.header("Help & Documentation")

    st.write(
        """
        ## About This Application
        
        This application uses the Synthetic Data Vault (SDV) library to generate synthetic data that statistically resembles your original dataset while preserving privacy.
        """
    )

    st.subheader("Available Models")

    # CTGAN model explanation
    with st.expander("CTGAN (Conditional Tabular GAN)", expanded=True):
        st.write(
            """
            **CTGAN** is a GAN-based model specifically designed for tabular data with mixed categorical and numerical columns.
            
            **How it works**: CTGAN uses a conditional generator and training-by-sampling approach to handle the non-Gaussian and multimodal distribution of tabular data. It employs mode-specific normalization to overcome the imbalanced discrete columns often present in tabular data.
            
            **Best suited for**:
            - Datasets with complex relationships between variables
            - Data with both categorical and numerical features
            - Datasets where preserving conditional distributions is important
            - Medium to large datasets (1,000+ rows)
            
            **Advantages**:
            - Excellent at capturing complex relationships between variables
            - Good at handling mixed data types
            - Preserves the statistical properties of the original data
            - Can generate high-quality synthetic data for diverse applications
            
            **Limitations**:
            - Requires more training time than simpler models
            - May struggle with very small datasets
            - Can be sensitive to hyperparameter choices
            """
        )

    # TVAE model explanation
    with st.expander("TVAE (Tabular Variational Autoencoder)"):
        st.write(
            """
            **TVAE** is a deep learning model based on variational autoencoders, adapted specifically for tabular data.
            
            **How it works**: TVAE learns to encode the input data into a lower-dimensional latent space and then decode it back to the original space. By sampling from this latent space, it can generate new, synthetic data points that maintain the statistical properties of the original data.
            
            **Best suited for**:
            - Datasets where preserving the overall distribution is more important than exact relationships
            - Data with many continuous variables
            - Cases where faster training is needed
            - Smaller to medium-sized datasets
            
            **Advantages**:
            - Generally faster to train than GAN-based models
            - Good at capturing the overall distribution of the data
            - More stable training process than GANs
            - Works well with continuous variables
            
            **Limitations**:
            - May not capture complex relationships as well as GAN-based models
            - Can struggle with highly categorical data
            - Sometimes produces less diverse samples than GAN-based approaches
            """
        )

    # CopulaGAN model explanation
    with st.expander("CopulaGAN"):
        st.write(
            """
            **CopulaGAN** combines copulas with GANs to better model statistical relationships in tabular data.
            
            **How it works**: CopulaGAN first transforms the data using a gaussian copula to capture the dependencies between columns, then uses a GAN to model the transformed data. This two-step approach helps preserve both the marginal distributions of individual columns and the dependencies between them.
            
            **Best suited for**:
            - Datasets with strong correlations between variables
            - Financial or insurance data where preserving dependencies is critical
            - Data with skewed distributions
            - Datasets where maintaining the exact shape of distributions is important
            
            **Advantages**:
            - Excellent at preserving correlations between variables
            - Better handles skewed distributions than standard GANs
            - Combines the strengths of statistical and deep learning approaches
            - Often produces more realistic synthetic data for financial applications
            
            **Limitations**:
            - More complex model with additional hyperparameters
            - May require more fine-tuning than simpler approaches
            - Training process can be slower due to the two-step approach
            """
        )

    st.subheader("How to Use")

    st.write(
        """
        1. **Upload Data**: Start by uploading a CSV file containing your data.
        2. **Configure Model**: Select a model type and configure parameters.
        3. **Generate Data**: Click the "Generate Synthetic Data" button to create synthetic data.
        4. **Visualize & Evaluate**: Use the Visualization and Evaluation tabs to analyze the quality of the synthetic data.
        
        ### Tips for Better Results
        
        - Clean your data before uploading (handle missing values, outliers, etc.)
        - For larger datasets, consider using more epochs and larger batch sizes
        - The optimization option can help find better parameters automatically
        - Check the quality metrics to ensure the synthetic data accurately represents your original data
        - Different models may perform better on different datasets, so try multiple models
        - For highly sensitive data, prioritize models and settings that show lower privacy risk
        """
    )

    st.subheader("Understanding the Optimization Process")

    st.write(
        """
        The optimization feature uses Optuna to find the best parameters for the selected model. It evaluates different parameter combinations and selects the one that produces the highest quality synthetic data.
        
        The number of trials determines how many different parameter combinations will be tested. More trials may find better parameters but will take longer to complete.
        
        **Parameters optimized include**:
        - Number of training epochs
        - Batch size
        - Embedding dimension
        - Network architecture (number and size of layers)
        
        The optimization process uses a quality score that measures how well the synthetic data preserves the statistical properties of the original data.
        """
    )

    st.subheader("Privacy Considerations")

    st.write(
        """
        While synthetic data is designed to protect privacy by not containing actual records from the original data, there are still privacy considerations to keep in mind:
        
        - **Memorization**: Models might occasionally memorize and reproduce exact records from the training data
        - **Attribute disclosure**: It may be possible to infer sensitive attributes about individuals in the original data
        - **Membership inference**: An attacker might be able to determine if a specific record was part of the training data
        
        The Privacy Metrics in the Evaluation tab help assess these risks. Lower exact match percentages and higher minimum distances generally indicate better privacy protection.
        
        For highly sensitive data, consider:
        - Using more epochs to help the model generalize better
        - Adding differential privacy techniques (not currently implemented in this app)
        - Carefully reviewing the synthetic data before sharing it
        """
    )

# Add a sidebar with download options
if st.session_state.synthetic_data is not None:
    st.sidebar.header("Download Options")

    # Create a download button for the synthetic data
    csv = st.session_state.synthetic_data.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Synthetic Data (CSV)",
        data=csv,
        file_name="synthetic_data.csv",
        mime="text/csv",
    )

    # Add JSON download option
    json_str = st.session_state.synthetic_data.to_json(orient="records")
    st.sidebar.download_button(
        label="Download Synthetic Data (JSON)",
        data=json_str,
        file_name="synthetic_data.json",
        mime="application/json",
    )

    # Add quality report download option if available
    if st.session_state.quality_report is not None:
        try:
            # Get the overall score
            overall_score = st.session_state.quality_report.get_score()

            # Get column shapes and convert DataFrame to dict if needed
            column_shapes = st.session_state.quality_report.get_details("Column Shapes")
            if isinstance(column_shapes, pd.DataFrame):
                column_shapes = column_shapes.to_dict()

            # Get column pairs and convert DataFrame to dict if needed
            column_pairs = st.session_state.quality_report.get_details(
                "Column Pair Trends"
            )
            if isinstance(column_pairs, pd.DataFrame):
                column_pairs = column_pairs.to_dict()

            # Create the report dictionary with serializable values
            report_dict = {
                "overall_score": overall_score,
                "column_shapes": column_shapes,
                "column_pairs": column_pairs,
            }

            # Convert to JSON
            report_json = json.dumps(report_dict, indent=2)

            st.sidebar.download_button(
                label="Download Quality Report (JSON)",
                data=report_json,
                file_name="quality_report.json",
                mime="application/json",
            )
        except Exception as e:
            st.sidebar.warning(
                f"Could not prepare quality report for download: {str(e)}"
            )
            st.sidebar.error(f"Error details: {type(e).__name__}")

    # Add privacy metrics download option if available
    if st.session_state.privacy_metrics:
        try:
            metrics_json = json.dumps(st.session_state.privacy_metrics, indent=2)
            st.sidebar.download_button(
                label="Download Privacy Metrics (JSON)",
                data=metrics_json,
                file_name="privacy_metrics.json",
                mime="application/json",
            )
        except Exception as e:
            st.sidebar.warning(f"Could not prepare privacy metrics for download: {e}")

# Add a sidebar section for additional information
st.sidebar.header("About")
st.sidebar.info(
    """
This application generates synthetic data using the SDV library with PyTorch backend. 
The synthetic data preserves the statistical properties of the original data 
while ensuring privacy.

**Developed by CDAO Intelligence Team:**
"""
)


# Run the app with: streamlit run sdv_streamlit_app.py
