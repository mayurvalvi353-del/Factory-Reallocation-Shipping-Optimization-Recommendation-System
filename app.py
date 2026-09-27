# ============================================================
# NASSAU CANDY DISTRIBUTOR
# FACTORY REALLOCATION & SHIPPING OPTIMIZATION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Nassau Candy Optimization",
    page_icon="🍫",
    layout="wide"
)


# ============================================================
# PRODUCT -> FACTORY MAPPING
# ============================================================

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",

    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",

    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",

    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory"
}


# ============================================================
# FACTORY LOCATIONS
# ============================================================

FACTORY_LOCATIONS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}


# ============================================================
# SHIPPING MODE BASE LEAD TIMES
#
# IMPORTANT:
# These are project simulation values because the original
# Ship Date column contains unrealistic date values.
# ============================================================

SHIP_MODE_DAYS = {
    "Same Day": 1,
    "First Class": 2,
    "Second Class": 3,
    "Standard Class": 5
}


# ============================================================
# FIND CSV
# ============================================================

def find_csv_file():

    project_folder = Path(__file__).resolve().parent

    exact_file = project_folder / "Nassau Candy Distributor(1).csv"

    if exact_file.exists():
        return exact_file

    data_folder = project_folder / "data"

    if data_folder.exists():

        data_file = data_folder / "Nassau Candy Distributor(1).csv"

        if data_file.exists():
            return data_file

    csv_files = list(project_folder.rglob("*.csv"))

    if len(csv_files) > 0:
        return csv_files[0]

    return None


# ============================================================
# DATA LOADING + CLEANING
# ============================================================

@st.cache_data
def load_dataset(file_path):

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    df = df.drop_duplicates().copy()

    # --------------------------------------------------------
    # Clean text columns
    # --------------------------------------------------------

    text_columns = df.select_dtypes(include=["object"]).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # REGION FIX
    # --------------------------------------------------------

    if "Region" in df.columns:

        df["Customer Region"] = (
            df["Region"]
            .astype(str)
            .str.strip()
        )

    elif "Customer Region" in df.columns:

        df["Customer Region"] = (
            df["Customer Region"]
            .astype(str)
            .str.strip()
        )

    else:

        df["Customer Region"] = "Unknown"

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "Sales",
        "Cost",
        "Gross Profit",
        "Units"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = df[column].fillna(
                df[column].median()
            )

    # --------------------------------------------------------
    # DATE CONVERSION
    # --------------------------------------------------------

    if "Order Date" in df.columns:

        df["Order Date"] = pd.to_datetime(
            df["Order Date"],
            dayfirst=True,
            errors="coerce"
        )

    if "Ship Date" in df.columns:

        df["Ship Date"] = pd.to_datetime(
            df["Ship Date"],
            dayfirst=True,
            errors="coerce"
        )

    # ========================================================
    # ORIGINAL LEAD TIME
    # ========================================================

    if (
        "Order Date" in df.columns
        and "Ship Date" in df.columns
    ):

        df["Original Lead Time"] = (
            df["Ship Date"] - df["Order Date"]
        ).dt.days

    else:

        df["Original Lead Time"] = np.nan

    # --------------------------------------------------------
    # Identify unrealistic dates
    # --------------------------------------------------------

    df["Date Anomaly"] = (
        (df["Original Lead Time"] < 0)
        |
        (df["Original Lead Time"] > 30)
    )

    # ========================================================
    # PROJECT LEAD TIME
    # ========================================================
    #
    # Because the supplied Ship Date values are unrealistic,
    # we create an operational lead-time field using the
    # shipping method.
    #
    # This is clearly identified as a PROJECT SIMULATION.
    # ========================================================

    if "Ship Mode" in df.columns:

        df["Operational Lead Time"] = (
            df["Ship Mode"]
            .map(SHIP_MODE_DAYS)
        )

    else:

        df["Operational Lead Time"] = 5

    # Handle unknown shipping modes

    df["Operational Lead Time"] = (
        df["Operational Lead Time"]
        .fillna(5)
        .astype(float)
    )

    # ========================================================
    # FACTORY MAPPING
    # ========================================================

    if "Product Name" in df.columns:

        df["Factory"] = (
            df["Product Name"]
            .map(PRODUCT_FACTORY)
            .fillna("Unknown")
        )

    else:

        df["Factory"] = "Unknown"

    # ========================================================
    # PROFIT FEATURES
    # ========================================================

    if (
        "Sales" in df.columns
        and "Gross Profit" in df.columns
    ):

        df["Profit Margin (%)"] = np.where(
            df["Sales"] != 0,
            (
                df["Gross Profit"]
                / df["Sales"]
            ) * 100,
            0
        )

    else:

        df["Profit Margin (%)"] = 0

    # --------------------------------------------------------

    if (
        "Sales" in df.columns
        and "Units" in df.columns
    ):

        df["Sales Per Unit"] = np.where(
            df["Units"] != 0,
            df["Sales"] / df["Units"],
            0
        )

    else:

        df["Sales Per Unit"] = 0

    # --------------------------------------------------------

    if (
        "Gross Profit" in df.columns
        and "Units" in df.columns
    ):

        df["Profit Per Unit"] = np.where(
            df["Units"] != 0,
            df["Gross Profit"] / df["Units"],
            0
        )

    else:

        df["Profit Per Unit"] = 0

    # ========================================================
    # DATE FEATURES
    # ========================================================

    if "Order Date" in df.columns:

        df["Order Year"] = (
            df["Order Date"].dt.year
        )

        df["Order Month"] = (
            df["Order Date"].dt.month
        )

        df["Order Day"] = (
            df["Order Date"].dt.day
        )

        df["Order Quarter"] = (
            df["Order Date"].dt.quarter
        )

    else:

        df["Order Year"] = 2025
        df["Order Month"] = 1
        df["Order Day"] = 1
        df["Order Quarter"] = 1

    return df


# ============================================================
# FACTORY ANALYSIS
# ============================================================

def factory_analysis(df):

    result = (
        df.groupby("Factory")
        .agg(
            Orders=("Sales", "count"),
            Sales=("Sales", "sum"),
            Profit=("Gross Profit", "sum"),
            Average_Lead_Time=(
                "Operational Lead Time",
                "mean"
            )
        )
        .reset_index()
    )

    result["Profit Margin (%)"] = np.where(
        result["Sales"] != 0,
        (
            result["Profit"]
            / result["Sales"]
        ) * 100,
        0
    )

    return result


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

def train_model(df):

    model_df = df.copy()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = [
        "Product Name",
        "Customer Region",
        "Ship Mode",
        "Factory",
        "Units",
        "Sales",
        "Cost",
        "Order Month",
        "Order Quarter"
    ]

    target = "Operational Lead Time"

    # --------------------------------------------------------
    # Check missing columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in features + [target]
        if column not in model_df.columns
    ]

    if len(missing_columns) > 0:

        return (
            None,
            None,
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Remove incomplete rows
    # --------------------------------------------------------

    model_df = model_df.dropna(
        subset=features + [target]
    )

    if len(model_df) < 20:

        return (
            None,
            None,
            "Not enough records for ML training."
        )

    X = model_df[features]

    y = model_df[target]

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_features = [
        "Product Name",
        "Customer Region",
        "Ship Mode",
        "Factory"
    ]

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    numeric_features = [
        "Units",
        "Sales",
        "Cost",
        "Order Month",
        "Order Quarter"
    ]

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                categorical_features
            ),

            (
                "numeric",

                "passthrough",

                numeric_features
            )
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "regressor",

                RandomForestRegressor(

                    n_estimators=150,

                    random_state=42,

                    n_jobs=-1
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    }

    return (
        model,
        metrics,
        None
    )


# ============================================================
# LOAD DATASET
# ============================================================

st.title(
    "🍫 Nassau Candy Factory Reallocation "
    "& Shipping Optimization"
)

st.write(
    "Machine Learning and Data Analytics "
    "system for shipping analysis and factory "
    "allocation."
)

csv_file = find_csv_file()


if csv_file is None:

    st.error(
        "❌ CSV dataset was not found."
    )

    st.info(
        "Put 'Nassau Candy Distributor(1).csv' "
        "in the same folder as app.py."
    )

    st.code(
        str(
            Path(__file__).resolve().parent
        )
    )

    st.stop()


# ============================================================
# READ DATA
# ============================================================

try:

    df = load_dataset(
        str(csv_file)
    )

except Exception as error:

    st.error(
        "❌ Error while loading dataset."
    )

    st.exception(error)

    st.stop()


st.success(
    "✅ Dataset loaded successfully"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📌 Navigation"
)

page = st.sidebar.radio(

    "Select Section",

    [
        "Dashboard",
        "Data Cleaning",
        "Factory Analysis",
        "Shipping Analysis",
        "ML Prediction",
        "Factory Recommendation"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header(
        "📊 Project Dashboard"
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    if "Order ID" in df.columns:

        total_orders = (
            df["Order ID"].nunique()
        )

    else:

        total_orders = len(df)

    total_sales = df["Sales"].sum()

    total_profit = (
        df["Gross Profit"].sum()
    )

    total_products = (
        df["Product Name"].nunique()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col2.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    col3.metric(
        "Gross Profit",
        f"${total_profit:,.2f}"
    )

    col4.metric(
        "Products",
        total_products
    )

    st.divider()

    # --------------------------------------------------------
    # SALES BY REGION
    # --------------------------------------------------------

    st.subheader(
        "Sales by Region"
    )

    region_sales = (
        df.groupby(
            "Customer Region"
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(

        region_sales,

        x="Customer Region",

        y="Sales",

        title="Sales by Customer Region"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------------
    # SALES BY PRODUCT
    # --------------------------------------------------------

    st.subheader(
        "Sales by Product"
    )

    product_sales = (
        df.groupby(
            "Product Name"
        )["Sales"]
        .sum()
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
    )

    fig2 = px.bar(

        product_sales,

        x="Sales",

        y="Product Name",

        orientation="h",

        title="Sales by Product"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )


# ============================================================
# DATA CLEANING
# ============================================================

elif page == "Data Cleaning":

    st.header(
        "🧹 Data Cleaning & Validation"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        len(df.columns)
    )

    col3.metric(
        "Duplicate Rows",
        df.duplicated().sum()
    )

    col4.metric(
        "Date Anomalies",
        int(df["Date Anomaly"].sum())
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    st.subheader(
        "Missing Values"
    )

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        st.success(
            "✅ No missing values found."
        )

    else:

        st.dataframe(
            missing.to_frame(
                "Missing Values"
            ),
            width="stretch"
        )

    # --------------------------------------------------------
    # Date issue
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Shipping Date Validation"
    )

    st.write(
        "The supplied Ship Date values produce "
        "very large lead times. They are kept "
        "unchanged for transparency."
    )

    date_col1, date_col2, date_col3 = (
        st.columns(3)
    )

    original_min = (
        df["Original Lead Time"].min()
    )

    original_avg = (
        df["Original Lead Time"].mean()
    )

    original_max = (
        df["Original Lead Time"].max()
    )

    date_col1.metric(
        "Minimum Original Lead Time",
        f"{original_min:.0f} days"
    )

    date_col2.metric(
        "Average Original Lead Time",
        f"{original_avg:.2f} days"
    )

    date_col3.metric(
        "Maximum Original Lead Time",
        f"{original_max:.0f} days"
    )

    st.warning(
        "The original Ship Date field contains "
        "unrealistic date differences. The application "
        "therefore uses Operational Lead Time for "
        "shipping simulation and ML."
    )

    # --------------------------------------------------------
    # Operational lead time
    # --------------------------------------------------------

    st.subheader(
        "✅ Operational Lead Time"
    )

    st.write(
        "Operational Lead Time is a project simulation "
        "based on the selected shipping mode."
    )

    mode_summary = (
        df.groupby(
            "Ship Mode"
        )["Operational Lead Time"]
        .first()
        .reset_index()
    )

    mode_summary.columns = [
        "Ship Mode",
        "Operational Lead Time (Days)"
    ]

    st.dataframe(
        mode_summary,
        width="stretch"
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        width="stretch"
    )


# ============================================================
# FACTORY ANALYSIS
# ============================================================

elif page == "Factory Analysis":

    st.header(
        "🏭 Factory Performance Analysis"
    )

    factory_data = (
        factory_analysis(df)
    )

    st.dataframe(
        factory_data.round(2),
        width="stretch"
    )

    # --------------------------------------------------------
    # Sales
    # --------------------------------------------------------

    fig = px.bar(

        factory_data,

        x="Factory",

        y="Sales",

        title="Sales by Factory"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------------
    # Profit
    # --------------------------------------------------------

    fig2 = px.bar(

        factory_data,

        x="Factory",

        y="Profit",

        title="Gross Profit by Factory"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

    # --------------------------------------------------------
    # Lead Time
    # --------------------------------------------------------

    fig3 = px.bar(

        factory_data,

        x="Factory",

        y="Average_Lead_Time",

        title="Average Operational Lead Time by Factory"
    )

    st.plotly_chart(
        fig3,
        width="stretch"
    )


# ============================================================
# SHIPPING ANALYSIS
# ============================================================

elif page == "Shipping Analysis":

    st.header(
        "🚚 Shipping Analysis"
    )

    # --------------------------------------------------------
    # Average
    # --------------------------------------------------------

    average_lead_time = (
        df["Operational Lead Time"].mean()
    )

    st.metric(
        "Average Operational Lead Time",
        f"{average_lead_time:.2f} days"
    )

    st.info(
        "Shipping analysis uses Operational Lead Time "
        "because the original Ship Date values contain "
        "data-quality anomalies."
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    fig = px.histogram(

        df,

        x="Operational Lead Time",

        nbins=10,

        title="Operational Lead-Time Distribution"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------------
    # Shipping Mode
    # --------------------------------------------------------

    mode_data = (
        df.groupby(
            "Ship Mode"
        )["Operational Lead Time"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(

        mode_data,

        x="Ship Mode",

        y="Operational Lead Time",

        title="Average Lead Time by Shipping Mode"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

    # --------------------------------------------------------
    # Shipping mode table
    # --------------------------------------------------------

    st.subheader(
        "Shipping Mode Summary"
    )

    st.dataframe(
        mode_data.round(2),
        width="stretch"
    )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "ML Prediction":

    st.header(
        "🤖 Shipping Lead-Time Prediction"
    )

    st.write(
        "Random Forest Regression is used to "
        "predict Operational Lead Time."
    )

    st.info(
        "Because the supplied Ship Date values are "
        "unrealistic, the ML target is the project-defined "
        "Operational Lead Time."
    )

    if st.button(
        "🚀 Train Random Forest Model"
    ):

        with st.spinner(
            "Training model..."
        ):

            model, metrics, error_message = (
                train_model(df)
            )

        if error_message:

            st.error(
                error_message
            )

        else:

            st.success(
                "✅ Machine Learning model trained successfully!"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "MAE",
                f"{metrics['MAE']:.2f} days"
            )

            col2.metric(
                "RMSE",
                f"{metrics['RMSE']:.2f}"
            )

            col3.metric(
                "R² Score",
                f"{metrics['R2']:.3f}"
            )

            st.write(
                "### Model Interpretation"
            )

            st.write(
                f"""
                **MAE:** {metrics['MAE']:.2f} days

                **RMSE:** {metrics['RMSE']:.2f}

                **R²:** {metrics['R2']:.3f}
                """
            )


# ============================================================
# FACTORY RECOMMENDATION
# ============================================================

elif page == "Factory Recommendation":

    st.header(
        "🏭 Factory Reallocation Simulator"
    )

    st.write(
        "Compare predicted operational lead time "
        "for different factory assignments."
    )

    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    products = sorted(
        df["Product Name"]
        .dropna()
        .unique()
        .tolist()
    )

    product = st.selectbox(
        "Select Product",
        products
    )

    # --------------------------------------------------------
    # Region
    # --------------------------------------------------------

    regions = sorted(
        df["Customer Region"]
        .dropna()
        .unique()
        .tolist()
    )

    region = st.selectbox(
        "Select Customer Region",
        regions
    )

    # --------------------------------------------------------
    # Ship Mode
    # --------------------------------------------------------

    ship_modes = sorted(
        df["Ship Mode"]
        .dropna()
        .unique()
        .tolist()
    )

    ship_mode = st.selectbox(
        "Select Shipping Mode",
        ship_modes
    )

    # --------------------------------------------------------
    # Units
    # --------------------------------------------------------

    units = st.number_input(
        "Units",
        min_value=1,
        value=10
    )

    # --------------------------------------------------------
    # Sales
    # --------------------------------------------------------

    sales = st.number_input(
        "Sales",
        min_value=0.0,
        value=100.0
    )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    cost = st.number_input(
        "Cost",
        min_value=0.0,
        value=50.0
    )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    month = st.slider(
        "Order Month",
        min_value=1,
        max_value=12,
        value=6
    )

    # --------------------------------------------------------
    # Current Factory
    # --------------------------------------------------------

    current_factory = PRODUCT_FACTORY.get(
        product,
        "Unknown"
    )

    st.info(
        f"Current product factory: **{current_factory}**"
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    if st.button(
        "🔍 Compare Factory Options"
    ):

        model, metrics, error_message = (
            train_model(df)
        )

        if model is None:

            st.error(
                "Factory prediction cannot be performed."
            )

            st.info(
                error_message
            )

        else:

            results = []

            # ------------------------------------------------
            # Test every factory
            # ------------------------------------------------

            for factory in FACTORY_LOCATIONS.keys():

                input_data = pd.DataFrame(
                    [
                        {

                            "Product Name":
                                product,

                            "Customer Region":
                                region,

                            "Ship Mode":
                                ship_mode,

                            "Factory":
                                factory,

                            "Units":
                                units,

                            "Sales":
                                sales,

                            "Cost":
                                cost,

                            "Order Month":
                                month,

                            "Order Quarter":
                                ((month - 1) // 3) + 1
                        }
                    ]
                )

                prediction = model.predict(
                    input_data
                )[0]

                results.append(
                    {
                        "Factory":
                            factory,

                        "Predicted Lead Time":
                            round(
                                float(
                                    prediction
                                ),
                                2
                            )
                    }
                )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            results_df = pd.DataFrame(
                results
            )

            results_df = (
                results_df
                .sort_values(
                    "Predicted Lead Time"
                )
                .reset_index(
                    drop=True
                )
            )

            st.subheader(
                "Factory Comparison"
            )

            st.dataframe(
                results_df,
                width="stretch"
            )

            # ------------------------------------------------
            # Chart
            # ------------------------------------------------

            fig = px.bar(

                results_df,

                x="Factory",

                y="Predicted Lead Time",

                title="Predicted Lead Time by Factory"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # ------------------------------------------------
            # Recommendation
            # ------------------------------------------------

            recommended_factory = (
                results_df.iloc[0]["Factory"]
            )

            predicted_days = (
                results_df.iloc[0]
                ["Predicted Lead Time"]
            )

            st.success(

                f"🏭 Factory with the lowest "
                f"predicted operational lead time: "
                f"**{recommended_factory}** "
                f"({predicted_days:.2f} days)"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Nassau Candy Distributor | "
    "Factory Reallocation & Shipping Optimization | "
    "Python + Pandas + Scikit-learn + Streamlit"
)