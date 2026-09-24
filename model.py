import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# PRODUCT -> FACTORY MAPPING
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FACTORY LOCATIONS
# ---------------------------------------------------------

FACTORY_LOCATIONS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data(file_path):

    df = pd.read_csv(file_path)

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Clean text columns
    text_columns = df.select_dtypes(include="object").columns

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert dates
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

# -----------------------------------------------------
    # LEAD TIME
    # -----------------------------------------------------

    df["Lead Time"] = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days

    # Flag suspicious dates
    df["Lead Time Anomaly"] = (
        (df["Lead Time"] < 0) |
        (df["Lead Time"] > 365)
    )

    # -----------------------------------------------------
    # FACTORY
    # -----------------------------------------------------

    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)

    # -----------------------------------------------------
    # PROFIT FEATURES
    # -----------------------------------------------------

    df["Profit Margin (%)"] = np.where(
        df["Sales"] != 0,
        (df["Gross Profit"] / df["Sales"]) * 100,
        0
    )

    df["Sales Per Unit"] = np.where(
        df["Units"] != 0,
        df["Sales"] / df["Units"],
        0
    )

    df["Profit Per Unit"] = np.where(
        df["Units"] != 0,
        df["Gross Profit"] / df["Units"],
        0
    )

    # -----------------------------------------------------
    # DATE FEATURES
    # -----------------------------------------------------

    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Day"] = df["Order Date"].dt.day
    df["Order Quarter"] = df["Order Date"].dt.quarter

    return df


# ---------------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# ---------------------------------------------------------

