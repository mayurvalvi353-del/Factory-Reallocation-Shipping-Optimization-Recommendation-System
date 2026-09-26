# ============================================================
# NASSAU CANDY FACTORY REALLOCATION & SHIPPING OPTIMIZATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
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

    "Wonka Bar - Nutty Crunch Surprise":
        "Lot's O' Nuts",

    "Wonka Bar - Fudge Mallows":
        "Lot's O' Nuts",

    "Wonka Bar -Scrumdiddlyumptious":
        "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate":
        "Wicked Choccy's",

    "Wonka Bar - Triple Dazzle Caramel":
        "Wicked Choccy's",

    "Laffy Taffy":
        "Sugar Shack",

    "SweeTARTS":
        "Sugar Shack",

    "Nerds":
        "Sugar Shack",

    "Fun Dip":
        "Sugar Shack",

    "Fizzy Lifting Drinks":
        "Sugar Shack",

    "Everlasting Gobstopper":
        "Secret Factory",

    "Lickable Wallpaper":
        "Secret Factory",

    "Wonka Gum":
        "Secret Factory",

    "Hair Toffee":
        "The Other Factory",

    "Kazookles":
        "The Other Factory"
}


# ============================================================
# FIND CSV FILE AUTOMATICALLY
# ============================================================

def find_dataset():

    # Folder where app.py is located
    project_folder = Path(__file__).resolve().parent

    # Possible locations
    possible_paths = [

        project_folder / "Nassau Candy Distributor.csv",

        project_folder / "data" /
        "Nassau Candy Distributor(1).csv",

        project_folder / "Nassau_Candy_Distributor.csv",

        project_folder / "data" /
        "Nassau_Candy_Distributor.csv"
    ]

    # Check exact paths
    for path in possible_paths:

        if path.exists():
            return path

    # If exact filename is not found,
    # search for ANY CSV file
    csv_files = list(
        project_folder.rglob("*.csv")
    )

    if len(csv_files) > 0:

        return csv_files[0]

    return None


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

@st.cache_data
def load_data(file_path):

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # CLEAN TEXT COLUMNS
    # --------------------------------------------------------

    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # CONVERT NUMERIC COLUMNS
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

    # --------------------------------------------------------
    # HANDLE MISSING NUMERIC VALUES
    # --------------------------------------------------------

    for column in numeric_columns:

        if column in df.columns:

            if df[column].isnull().sum() > 0:

                df[column] = df[column].fillna(
                    df[column].median()
                )

    # --------------------------------------------------------
    # HANDLE MISSING TEXT VALUES
    # --------------------------------------------------------

    for column in text_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                "Unknown"
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

    # --------------------------------------------------------
    # LEAD TIME
    # --------------------------------------------------------

    if (
        "Order Date" in df.columns
        and
        "Ship Date" in df.columns
    ):

        df["Lead Time"] = (
            df["Ship Date"] -
            df["Order Date"]
        ).dt.days

    else:

        df["Lead Time"] = np.nan

    # --------------------------------------------------------
    # FLAG SUSPICIOUS LEAD TIMES
    # --------------------------------------------------------

    df["Lead Time Anomaly"] = (

        (df["Lead Time"] < 0)
        |
        (df["Lead Time"] > 365)

    )

    # --------------------------------------------------------
    # FACTORY MAPPING
    # --------------------------------------------------------

    if "Product Name" in df.columns:

        df["Factory"] = (
            df["Product Name"]
            .map(PRODUCT_FACTORY)
        )

    else:

        df["Factory"] = "Unknown"

    # --------------------------------------------------------
    # PROFIT MARGIN
    # --------------------------------------------------------

    if "Sales" in df.columns:

        df["Profit Margin (%)"] = np.where(

            df["Sales"] != 0,

            (
                df["Gross Profit"] /
                df["Sales"]
            ) * 100,

            0
        )

    # --------------------------------------------------------
    # SALES PER UNIT
    # --------------------------------------------------------

    if "Units" in df.columns:

        df["Sales Per Unit"] = np.where(

            df["Units"] != 0,

            df["Sales"] /
            df["Units"],

            0
        )

        df["Profit Per Unit"] = np.where(

            df["Units"] != 0,

            df["Gross Profit"] /
            df["Units"],

            0
        )

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    if "Order Date" in df.columns:

        df["Order Year"] = (
            df["Order Date"].dt.year
        )

        df["Order Month"] = (
            df["Order Date"].dt.month
        )

        df["Order Quarter"] = (
            df["Order Date"].dt.quarter
        )

        df["Order Day"] = (
            df["Order Date"].dt.day
        )

        df["Order Day Name"] = (
            df["Order Date"].dt.day_name()
        )

    return df


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

def train_ml_model(df):

    # --------------------------------------------------------
    # IMPORTANT:
    # Only use realistic lead times for ML
    # --------------------------------------------------------

    model_df = df[
        (df["Lead Time"] >= 0)
        &
        (df["Lead Time"] <= 365)
    ].copy()

    # Check whether enough valid records exist
    if len(model_df) < 20:

        return None, None, (
            "Not enough valid lead-time records "
            "for Machine Learning."
        )

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

    target = "Lead Time"

    # Make sure all required columns exist
    missing_features = [
        col for col in features
        if col not in model_df.columns
    ]

    if len(missing_features) > 0:

        return None, None, (
            "Missing columns: "
            + ", ".join(missing_features)
        )

    model_df = model_df.dropna(
        subset=features + [target]
    )

    if len(model_df) < 20:

        return None, None, (
            "Not enough clean records "
            "after preprocessing."
        )

    X = model_df[features]

    y = model_df[target]

    categorical_features = [
        "Product Name",
        "Customer Region",
        "Ship Mode",
        "Factory"
    ]

    numeric_features = [
        "Units",
        "Sales",
        "Cost",
        "Order Month",
        "Order Quarter"
    ]

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

    model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "regressor",

                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

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

    return model, metrics, None


# ============================================================
# FACTORY ANALYSIS
# ============================================================

def get_factory_analysis(df):

    result = (
        df
        .groupby("Factory")
        .agg(

            Orders=("Order ID", "count"),

            Sales=("Sales", "sum"),

            Cost=("Cost", "sum"),

            Profit=("Gross Profit", "sum"),

            Average_Lead_Time=(
                "Lead Time",
                "mean"
            )

        )
        .reset_index()
    )

    result["Profit Margin (%)"] = np.where(

        result["Sales"] != 0,

        (
            result["Profit"] /
            result["Sales"]
        ) * 100,

        0
    )

    return result


# ============================================================
# APPLICATION START
# ============================================================

st.title(
    "🍫 Nassau Candy Factory Reallocation & Shipping Optimization"
)

st.write(
    "A data-driven Machine Learning system for "
    "shipping analysis, factory allocation and "
    "logistics optimization."
)


# ============================================================
# FIND DATASET
# ============================================================

dataset_path = find_dataset()


if dataset_path is None:

    st.error(
        "❌ Dataset not found!"
    )

    st.warning(
        "Please place your CSV file in the same "
        "folder as app.py."
    )

    st.write(
        "Expected file:"
    )

    st.code(
        "Nassau Candy Distributor(1).csv"
    )

    st.write(
        "Current project folder:"
    )

    st.code(
        str(
            Path(__file__)
            .resolve()
            .parent
        )
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = load_data(
        str(dataset_path)
    )

except Exception as error:

    st.error(
        "❌ Error while reading dataset."
    )

    st.exception(error)

    st.stop()


st.success(
    f"✅ Dataset loaded successfully: "
    f"{dataset_path.name}"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📌 Project Menu"
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

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    # Total orders
    if "Order ID" in df.columns:

        total_orders = (
            df["Order ID"]
            .nunique()
        )

    else:

        total_orders = len(df)

    col1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col2.metric(
        "Total Sales",
        f"${df['Sales'].sum():,.2f}"
    )

    col3.metric(
        "Gross Profit",
        f"${df['Gross Profit'].sum():,.2f}"
    )

    col4.metric(
        "Products",
        df["Product Name"]
        .nunique()
    )

    st.divider()

    # --------------------------------------------------------
    # SALES BY REGION
    # --------------------------------------------------------

    st.subheader(
        "Sales by Customer Region"
    )

    region_sales = (

        df
        .groupby("Customer Region")
        ["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(

        region_sales,

        x="Customer Region",

        y="Sales",

        text_auto=".2s",

        title="Sales by Region"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PRODUCT SALES
    # --------------------------------------------------------

    st.subheader(
        "Sales by Product"
    )

    product_sales = (

        df
        .groupby("Product Name")
        ["Sales"]
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

        title="Product Sales"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# DATA CLEANING
# ============================================================

elif page == "Data Cleaning":

    st.header(
        "🧹 Data Cleaning & Preparation"
    )

    col1, col2, col3 = (
        st.columns(3)
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

    st.subheader(
        "Missing Values"
    )

    missing = (
        df.isnull()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        st.success(
            "✅ No missing values found."
        )

    else:

        st.dataframe(
            missing
        )

    st.subheader(
        "Data Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader(
        "Lead-Time Data Quality"
    )

    anomaly_count = (
        df["Lead Time Anomaly"]
        .sum()
    )

    valid_count = (
        (~df["Lead Time Anomaly"])
        .sum()
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Valid Lead-Time Records",
        f"{valid_count:,}"
    )

    c2.metric(
        "Suspicious Lead-Time Records",
        f"{anomaly_count:,}"
    )

    if anomaly_count > 0:

        st.warning(
            "Some Order Date and Ship Date combinations "
            "produce unusually large lead times. These "
            "records are flagged rather than silently "
            "changed or deleted."
        )


# ============================================================
# FACTORY ANALYSIS
# ============================================================

elif page == "Factory Analysis":

    st.header(
        "🏭 Factory Performance Analysis"
    )

    factory_data = (
        get_factory_analysis(df)
    )

    st.dataframe(

        factory_data.round(2),

        use_container_width=True
    )

    st.subheader(
        "Sales by Factory"
    )

    fig = px.bar(

        factory_data,

        x="Factory",

        y="Sales",

        title="Factory Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Gross Profit by Factory"
    )

    fig2 = px.bar(

        factory_data,

        x="Factory",

        y="Profit",

        title="Factory Gross Profit"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# SHIPPING ANALYSIS
# ============================================================

elif page == "Shipping Analysis":

    st.header(
        "🚚 Shipping Analysis"
    )

    valid_shipping = df[
        (df["Lead Time"] >= 0)
        &
        (df["Lead Time"] <= 365)
    ].copy()

    if len(valid_shipping) == 0:

        st.warning(
            "No realistic lead-time records "
            "were found in the current dataset."
        )

        st.info(
            "The dataset contains unusually large "
            "differences between Order Date and Ship Date. "
            "These records have been flagged instead of "
            "being modified automatically."
        )

    else:

        st.metric(
            "Average Lead Time",
            f"{valid_shipping['Lead Time'].mean():.2f} days"
        )

        fig = px.histogram(

            valid_shipping,

            x="Lead Time",

            nbins=30,

            title="Shipping Lead-Time Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        mode_data = (

            valid_shipping
            .groupby("Ship Mode")
            ["Lead Time"]
            .mean()
            .reset_index()
        )

        fig2 = px.bar(

            mode_data,

            x="Ship Mode",

            y="Lead Time",

            title="Average Lead Time by Shipping Mode"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "ML Prediction":

    st.header(
        "🤖 Shipping Lead-Time Prediction"
    )

    st.write(
        "Random Forest Regressor is used to predict "
        "shipping lead time from product, region, "
        "shipping mode, factory and order information."
    )

    if st.button(
        "🚀 Train Machine Learning Model"
    ):

        with st.spinner(
            "Training Random Forest model..."
        ):

            model, metrics, error_message = (
                train_ml_model(df)
            )

        if error_message:

            st.warning(
                "⚠️ " + error_message
            )

            st.info(
                "The current dataset does not contain "
                "enough realistic lead-time values "
                "(0–365 days) for reliable ML training."
            )

        else:

            st.success(
                "✅ Model trained successfully!"
            )

            c1, c2, c3 = (
                st.columns(3)
            )

            c1.metric(
                "MAE",
                f"{metrics['MAE']:.2f} days"
            )

            c2.metric(
                "RMSE",
                f"{metrics['RMSE']:.2f} days"
            )

            c3.metric(
                "R² Score",
                f"{metrics['R2']:.3f}"
            )

            st.write(
                "Lower MAE and RMSE indicate smaller "
                "prediction errors, while a higher R² "
                "indicates better explanatory performance."
            )


# ============================================================
# FACTORY RECOMMENDATION
# ============================================================

elif page == "Factory Recommendation":

    st.header(
        "🏭 Factory Reallocation Simulator"
    )

    st.write(
        "Compare possible factory assignments "
        "for a selected product and customer scenario."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    product = st.selectbox(

        "Select Product",

        sorted(
            df["Product Name"]
            .dropna()
            .unique()
        )
    )

    region = st.selectbox(

        "Customer Region",

        sorted(
            df["Customer Region"]
            .dropna()
            .unique()
        )
    )

    ship_mode = st.selectbox(

        "Shipping Mode",

        sorted(
            df["Ship Mode"]
            .dropna()
            .unique()
        )
    )

    units = st.number_input(

        "Number of Units",

        min_value=1,

        value=10
    )

    sales = st.number_input(

        "Sales Amount",

        min_value=0.0,

        value=100.0
    )

    cost = st.number_input(

        "Cost",

        min_value=0.0,

        value=50.0
    )

    month = st.slider(

        "Order Month",

        min_value=1,

        max_value=12,

        value=6
    )

    # --------------------------------------------------------
    # CURRENT FACTORY
    # --------------------------------------------------------

    current_factory = PRODUCT_FACTORY.get(
        product,
        "Unknown"
    )

    st.info(
        f"Current mapped factory: "
        f"**{current_factory}**"
    )

    # --------------------------------------------------------
    # RECOMMENDATION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Factory Options"
    ):

        # Try ML first
        model, metrics, error_message = (
            train_ml_model(df)
        )

        factories = sorted(
            df["Factory"]
            .dropna()
            .unique()
        )

        results = []

        # ----------------------------------------------------
        # ML-BASED COMPARISON
        # ----------------------------------------------------

        if model is not None:

            for factory in factories:

                input_data = pd.DataFrame([{

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
                }])

                predicted_time = (
                    model.predict(
                        input_data
                    )[0]
                )

                results.append({

                    "Factory":
                        factory,

                    "Predicted Lead Time":
                        round(
                            predicted_time,
                            2
                        )
                })

            results_df = pd.DataFrame(
                results
            )

            st.subheader(
                "ML-Based Factory Comparison"
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            fig = px.bar(

                results_df,

                x="Factory",

                y="Predicted Lead Time",

                title=
                "Predicted Lead Time by Factory"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            best_row = results_df.loc[
                results_df[
                    "Predicted Lead Time"
                ].idxmin()
            ]

            st.success(

                f"Lowest predicted lead time: "
                f"**{best_row['Factory']}** "
                f"({best_row['Predicted Lead Time']:.2f} days)"
            )

        # ----------------------------------------------------
        # HISTORICAL FALLBACK
        # ----------------------------------------------------

        else:

            st.warning(
                "ML prediction is unavailable because "
                "the dataset does not contain enough "
                "valid lead-time records."
            )

            st.info(
                "Showing historical factory performance "
                "instead."
            )

            factory_data = (
                get_factory_analysis(df)
            )

            st.dataframe(

                factory_data.round(2),

                use_container_width=True
            )

            # Use profit as additional business information
            st.write(
                "You can use factory sales, profit and "
                "historical performance to evaluate "
                "allocation scenarios."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Nassau Candy Factory Reallocation & Shipping "
    "Optimization System | Python + Pandas + "
    "Scikit-learn + Streamlit"
)