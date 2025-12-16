import base64
from pathlib import Path
from PIL import Image
import streamlit as st
import joblib
import pandas as pd
import json
import plotly.express as px
import unicodedata
import re


# -----------------------------
# Name normalization
# -----------------------------
def normalize_name(x: str) -> str:
    if pd.isna(x):
        return x
    x = str(x)

    x = x.replace("\xa0", " ")
    x = re.sub(r"[\u200B-\u200D\uFEFF]", "", x)

    x = " ".join(x.split()).strip()

    x = unicodedata.normalize("NFC", x)
    return x


# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).parent
IMAGES = BASE / "images"
ARTIFACTS = BASE / "artifacts"
DATA = BASE / "data"

LISTINGS_PATH = DATA / "listings.csv"  


# -----------------------------
# Load GeoJSON
# -----------------------------
with open(DATA / "berlin_districts.geojson", "r", encoding="utf-8") as f:
    berlin_geojson = json.load(f)


@st.cache_data
def normalized_geojson(geojson: dict) -> dict:
    """Return a copy of geojson where properties.name is normalized."""
    gj = json.loads(json.dumps(geojson))  # deep copy
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if "name" in props:
            props["name"] = normalize_name(props["name"])
    return gj


berlin_geojson_norm = normalized_geojson(berlin_geojson)


# -----------------------------
# Load listings and Heatmap
# -----------------------------
@st.cache_data
def load_listings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    required = {"neighbourhood", "Price"}
    if not required.issubset(df.columns):
        raise ValueError(f"listings.csv must contain columns: {required}")

    df = df.rename(columns={"neighbourhood": "neighbourhood", "Price": "price"})
    df["neighbourhood"] = df["neighbourhood"].map(normalize_name)
    df = df.dropna(subset=["neighbourhood", "price"])
    return df


@st.cache_data
def load_heatmap_from_listings(path: Path) -> pd.DataFrame:
    df = load_listings(path)

    agg = (
        df.groupby("neighbourhood")["price"]
          .agg(median_price="median", listings="count")
          .reset_index()
    )

    agg = agg[agg["listings"] >= 10].copy()

    lo, hi = agg["median_price"].quantile([0.05, 0.95])
    agg["median_price_clipped"] = agg["median_price"].clip(lo, hi)

    return agg


# -----------------------------
# GeoJSON centroids 
# -----------------------------
@st.cache_data
def build_geojson_centroids(geojson: dict) -> dict:
    """
    Returns: { neighbourhood_name: (lat, lon) }
    GeoJSON coords are (lon, lat)
    """
    centroids = {}

    def collect_points_from_polygon(poly_coords, points_out):
        if poly_coords and isinstance(poly_coords[0], list):
            outer = poly_coords[0]
            for lon, lat in outer:
                points_out.append((lat, lon))

    for feat in geojson.get("features", []):
        name = feat.get("properties", {}).get("name")
        name = normalize_name(name) if name is not None else None

        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type")

        points = []
        if gtype == "Polygon":
            collect_points_from_polygon(coords, points)
        elif gtype == "MultiPolygon":
            for poly in coords:
                collect_points_from_polygon(poly, points)

        if name and points:
            lat_mean = sum(p[0] for p in points) / len(points)
            lon_mean = sum(p[1] for p in points) / len(points)
            centroids[name] = (lat_mean, lon_mean)

    return centroids


geo_centroids = build_geojson_centroids(berlin_geojson_norm)


# -----------------------------
# Load Model Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS / "price_model.pkl")
    features = joblib.load(ARTIFACTS / "model_features.pkl")
    return model, features


# -----------------------------
# Favicon + Page Config
# -----------------------------
try:
    favicon = Image.open(IMAGES / "home.png")
except Exception:
    favicon = "🏠"

st.set_page_config(
    page_title="Smart Pricing & Investment",
    page_icon=favicon,
    layout="wide"
)


# -----------------------------
# Session state
# -----------------------------
if "role" not in st.session_state:
    st.session_state.role = None

if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = {}


# -----------------------------
# Helpers
# -----------------------------
def img_tag(p: Path, alt: str = "", size: int = 108) -> str:
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"""
    <div class="icon-circle" style="width:{size}px;height:{size}px;">
        <img src="data:image/png;base64,{b64}" alt="{alt}">
    </div>
    """


def render_price_heatmap(selected_neighbourhood: str):
    agg = load_heatmap_from_listings(LISTINGS_PATH)
    selected_neighbourhood = normalize_name(selected_neighbourhood)

    DARK_ELEGANT_SCALE = [
        [0.0, "#020617"],   
        [0.25, "#0f172a"],  
        [0.5, "#1e40af"],   
        [0.75, "#2563eb"],  
        [1.0, "#38bdf8"],   
    ]

    fig = px.choropleth(
        agg,
        geojson=berlin_geojson_norm,
        locations="neighbourhood",
        featureidkey="properties.name",
        color="median_price_clipped",
        hover_data={
            "median_price": True,
            "listings": True,
            "median_price_clipped": False
        },
        labels={"median_price_clipped": "Median nightly price (€)"},
        color_continuous_scale=DARK_ELEGANT_SCALE,
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)"
    )

    line_widths = [4.5 if n == selected_neighbourhood else 0.5 for n in agg["neighbourhood"]]
    line_colors = ["#14b8a6" if n == selected_neighbourhood else "rgba(255,255,255,0.35)" for n in agg["neighbourhood"]]

    fig.update_traces(marker_line_width=line_widths, marker_line_color=line_colors)

    fig.update_layout(
        height=520,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(226,232,240,0.9)"),
        coloraxis_colorbar=dict(
            title="Median price",
            tickprefix="€",
            thickness=14,
            outlinewidth=0,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# DARK PREMIUM THEME (your CSS)
# -----------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">

<style>
[data-testid="stHeader"] {display: none;}
[data-testid="stToolbar"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background: radial-gradient(1200px 700px at 20% 10%, rgba(20,184,166,0.18), rgba(0,0,0,0)),
                radial-gradient(900px 600px at 90% 20%, rgba(99,102,241,0.16), rgba(0,0,0,0)),
                #0b1220;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: rgba(2,6,23,0.75) !important;
    border-right: 1px solid rgba(148,163,184,0.18);
}

.block-container {
    padding-top: 3.2rem;
    padding-bottom: 1.5rem;
    max-width: 1200px;
    margin: 0 auto;
}

.hero-title {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 46px;
    line-height: 1.05;
    margin-top: 12px;
    margin-bottom: 10px;
    letter-spacing: 0.3px;
    color: #f8fafc;
}

.hero-subtitle {
    text-align: center;
    font-size: 16px;
    color: rgba(226,232,240,0.78);
    margin-bottom: 28px;
}

label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stSlider label {
    color: rgba(248,250,252,0.95) !important;
    font-weight: 600 !important;
}

.icon-circle {
    border-radius: 50%;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148,163,184,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 14px auto;
    box-shadow:
        inset 0 0 0 1px rgba(255,255,255,0.04),
        0 10px 30px rgba(0,0,0,0.35);
}

.icon-circle img {
    width: 56%;
    height: 56%;
    object-fit: contain;
    filter: invert(1);
    opacity: 0.92;
}

div[data-testid="stButton"] > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    border: 1px solid rgba(148,163,184,0.22);
    background: rgba(2, 6, 23, 0.35);
    color: #e5e7eb;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border 0.15s ease;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(20,184,166,0.55);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

/* =========================
   METRICS (Key Results)
   ========================= */
div[data-testid="metric-container"]{
    background: transparent !important;
    text-align: center !important;
    padding: 18px 10px !important;
    border-radius: 14px !important;
}

/* Label */
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] div{
    color: rgba(248,250,252,0.95) !important;
    font-weight: 650 !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Value */
div[data-testid="metric-container"] [data-testid="stMetricValue"] div,
div[data-testid="metric-container"] [data-testid="stMetricValue"] p{
    color: #ffffff !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Delta */
div[data-testid="metric-container"] [data-testid="stMetricDelta"] div,
div[data-testid="metric-container"] [data-testid="stMetricDelta"] p{
    color: rgba(226,232,240,0.9) !important;
    justify-content: center !important;
    text-align: center !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="hero-title">Berlin Airbnb Pricing Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Smart pricing, what-if scenarios, and listing assistance for <b>Hosts</b> and <b>Investors</b>.</div>',
    unsafe_allow_html=True
)


# -----------------------------
# PAGES
# -----------------------------
def landing():
    left_spacer, colHost, colInvestor, right_spacer = st.columns([1, 2, 2, 1], gap="large")

    with colHost:
        with st.container(border=True):
            st.markdown(img_tag(IMAGES / "rent.png", "HOST", size=110), unsafe_allow_html=True)
            st.markdown("### Host")
            st.caption("Predict an optimal nightly rate and generate a compelling listing description.")
            if st.button("Continue as Host →", key="btn_host"):
                st.session_state.role = "host"
                st.rerun()

    with colInvestor:
        with st.container(border=True):
            st.markdown(img_tag(IMAGES / "property.png", "INVESTOR", size=110), unsafe_allow_html=True)
            st.markdown("### Investor")
            st.caption("Estimate revenue, yield, and run occupancy scenarios before you invest.")
            if st.button("Continue as Investor →", key="btn_investor"):
                st.session_state.role = "investor"
                st.rerun()


def host_page_placeholder():
    model, features = load_artifacts()

    if st.button("← Back", key="back_from_host"):
        st.session_state.role = None
        st.rerun()

    st.subheader("🏠 Host Workspace")
    st.caption("Enter your property details to estimate the optimal nightly price.")

    def detect_prefix(prefix_candidates):
        for p in prefix_candidates:
            if any(isinstance(f, str) and f.startswith(p) for f in features):
                return p
        return None

    neigh_prefix = detect_prefix(["neighbourhood_", "neighbourhood_cleansed_", "Neighborhood_", "Neighborhood Group_"])
    room_prefix = detect_prefix(["Room Type_", "room_type_"])

    if neigh_prefix:
        neighbourhood_options = sorted({
            normalize_name(f.split(neigh_prefix, 1)[1])
            for f in features
            if isinstance(f, str) and f.startswith(neigh_prefix)
        })
    else:
        neighbourhood_options = sorted({
            normalize_name(feat.get("properties", {}).get("name"))
            for feat in berlin_geojson_norm.get("features", [])
            if feat.get("properties", {}).get("name") is not None
        })

    with st.form("host_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
            accommodates = st.number_input("Accommodates", 1, 20, 2)
            bedrooms = st.number_input("Bedrooms", 0, 10, 1)

        with col2:
            bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)
            minimum_nights = st.number_input("Minimum nights", 1, 365, 2)
            neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

        instant_bookable = st.checkbox("Instant bookable", value=True)
        submitted = st.form_submit_button("Predict price")

    if submitted:
        X_df = pd.DataFrame(0.0, index=[0], columns=features)

        numeric_map = {
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "minimum_nights": minimum_nights,
        }
        for col, val in numeric_map.items():
            if col in X_df.columns:
                X_df.loc[0, col] = val

        if room_prefix:
            rt_col = f"{room_prefix}{room_type}"
            if rt_col in X_df.columns:
                X_df.loc[0, rt_col] = 1

        neighbourhood_norm = normalize_name(neighbourhood)
        if neigh_prefix:
            n_col = f"{neigh_prefix}{neighbourhood_norm}"
            if n_col in X_df.columns:
                X_df.loc[0, n_col] = 1

        if "instant_bookable" in X_df.columns:
            X_df.loc[0, "instant_bookable"] = int(instant_bookable)

        lat_col = next((c for c in X_df.columns if "lat" in c.lower()), None)
        lon_col = next((c for c in X_df.columns if "lon" in c.lower()), None)
        if lat_col and lon_col:
            lat, lon = geo_centroids.get(neighbourhood_norm, (52.52, 13.405))
            X_df.loc[0, lat_col] = lat
            X_df.loc[0, lon_col] = lon

        price = float(model.predict(X_df)[0])

        st.session_state.last_inputs = {
            "neighbourhood": neighbourhood_norm,
            "room_type": room_type,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "minimum_nights": minimum_nights,
            "instant_bookable": instant_bookable,
            "predicted_price": price,
        }

        st.success(f"Estimated Nightly Price: **€{price:,.0f}**")

        st.markdown("### 🧠 Why this price?")

        bullets = []

        premium_neigh = {
            "Mitte", "Prenzlauer Berg", "Kreuzberg", "Friedrichshain",
            "Charlottenburg", "Wilmersdorf", "Tiergarten", "Schöneberg"
        }
        value_neigh = {
            "Spandau", "Marzahn", "Hellersdorf", "Wartenberg", "Biesdorf",
            "Staaken", "Falkenberg", "Bohnsdorf", "Blankenfelde"
        }

        if neighbourhood_norm in premium_neigh:
            bullets.append("Central / high-demand neighbourhoods typically command a price premium.")
        elif neighbourhood_norm in value_neigh:
            bullets.append("This neighbourhood is usually more price-competitive compared to central areas.")
        else:
            bullets.append("Neighbourhood demand is a key driver of nightly rates in Berlin.")

        if int(accommodates) >= 4:
            bullets.append("Higher guest capacity often increases the expected nightly rate.")
        else:
            bullets.append("Smaller listings are often priced lower but can be highly competitive.")

        if int(bedrooms) >= 2:
            bullets.append("More bedrooms generally increases price, especially for families and groups.")
        elif int(bedrooms) == 0:
            bullets.append("Studios can price well when location and amenities are strong.")
        else:
            bullets.append("1-bedroom apartments often balance demand and price very well in Berlin.")

        if instant_bookable:
            bullets.append("Instant Book can improve conversion and supports slightly stronger pricing.")
        else:
            bullets.append("Non–Instant Book listings may need slightly more competitive pricing to convert.")

        if int(minimum_nights) >= 7:
            bullets.append("Longer minimum stays can reduce demand; pricing often reflects that trade-off.")
        elif int(minimum_nights) == 1:
            bullets.append("Short minimum stays can boost demand—often supporting higher nightly rates.")
        else:
            bullets.append("Minimum stay requirements influence demand and pricing flexibility.")

        bullets = bullets[:5]
        st.markdown("\n".join([f"- {b}" for b in bullets]))

        st.markdown("### 🗺️ Berlin Price Heatmap")
        st.caption("Full-market view. The selected neighbourhood is highlighted.")
        render_price_heatmap(neighbourhood_norm)

    if "last_inputs" in st.session_state:
        st.markdown("---")
        st.markdown("### ✍️ Not sure what to write in your Airbnb description?")
        st.caption("Upload photos and generate a professional, high-converting listing description.")

        if st.button("✨ Generate my Airbnb description", use_container_width=True, key="go_desc"):
            st.session_state.role = "description"
            st.rerun()


def description_page():
    if st.button("← Back to pricing", key="back_from_desc"):
        st.session_state.role = "host"
        st.rerun()

    data = st.session_state.get("last_inputs", {})

    st.subheader("📝 Airbnb Listing Description Generator")

    st.markdown("### 🏠 Listing Summary")
    if data:
        st.write(f"Location: {data.get('neighbourhood', '—')}")
        st.write(f"Room type: {data.get('room_type', '—')}")
        st.write(f"Guests: {data.get('accommodates', '—')}")
        st.write(f"Bedrooms: {data.get('bedrooms', '—')}")
        st.write(f"Bathrooms: {data.get('bathrooms', '—')}")
        st.write(f"Suggested price: €{float(data.get('predicted_price', 0) or 0):,.0f}")
    else:
        st.info("No pricing context found. Please go back and run a price prediction first.")
        return

    st.markdown("---")

    st.markdown("### 📸 Upload Photos")
    uploaded = st.file_uploader(
        "Upload up to 5 photos (JPG/PNG).",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded:
        st.caption("Preview")
        cols = st.columns(min(5, len(uploaded)))
        for i, img in enumerate(uploaded[:5]):
            cols[i].image(img, use_container_width=True)

    st.markdown("---")

    st.markdown("### ✍️ Description Preferences")

    tone = st.selectbox(
        "Preferred writing style",
        ["Warm & Welcoming", "Modern & Professional", "Luxury & Premium", "Cozy & Home-like"],
        index=0
    )

    highlights = st.text_area(
        "Key highlights to include (optional)",
        placeholder="e.g. quiet street, fast Wi-Fi, close to U-Bahn/S-Bahn, family-friendly, late check-in…"
    ).strip()

    st.markdown("### 🖼️ Photo Notes")
    st.caption("Write 1–2 sentences about what stands out in your photos (light, decor, kitchen, balcony, view, etc.).")

    photo_notes = st.text_area(
        "What do your photos show?",
        placeholder="e.g. bright living room with large windows, modern kitchen, cozy bedroom, balcony with a view…"
    ).strip()

    st.markdown("---")

    if st.button("✨ Create my description", use_container_width=True, key="btn_create_desc"):
        room_type = data.get("room_type", "Apartment")
        neighbourhood = data.get("neighbourhood", "Berlin")
        accommodates = int(data.get("accommodates", 2) or 2)
        bedrooms = int(data.get("bedrooms", 1) or 1)

        bathrooms_raw = data.get("bathrooms", 1)
        try:
            bathrooms_val = float(bathrooms_raw)
            bathrooms = int(bathrooms_val) if bathrooms_val.is_integer() else bathrooms_val
        except Exception:
            bathrooms = bathrooms_raw

        tone_map = {
            "Warm & Welcoming": ("warm and welcoming", "settle in and feel at home"),
            "Modern & Professional": ("modern and professional", "work, recharge, and explore with ease"),
            "Luxury & Premium": ("elevated and premium", "enjoy a refined Berlin stay"),
            "Cozy & Home-like": ("cozy and home-like", "unwind after a day out"),
        }
        tone_adj, closing_vibe = tone_map.get(tone, ("warm and welcoming", "enjoy Berlin"))

        if photo_notes:
            photo_sentence = (
                f"From the photos, you’ll notice {photo_notes.rstrip('.')}. "
                f"It really sets the tone for a comfortable and memorable stay."
            )
        else:
            photo_sentence = (
                "The space is designed to be comfortable, practical, and easy to enjoy from the moment you arrive."
            )

        extra_sentence = ""
        if highlights:
            if not highlights.endswith((".", "!", "?")):
                highlights += "."
            extra_sentence = f" A few extra notes: {highlights}"

        description = (
            f"Welcome to this {tone_adj} {room_type.lower()} in {neighbourhood}. "
            f"It’s a great fit for up to {accommodates} guests, with {bedrooms} bedroom(s) and {bathrooms} bathroom(s), "
            f"whether you’re visiting for a weekend city break, a longer stay, or a work trip.\n\n"
            f"{photo_sentence} "
            f"You’ll have a well-balanced setup for both relaxing and getting things done, with the essentials you’d expect for a smooth stay. "
            f"{extra_sentence}\n\n"
            f"Location-wise, you’re in {neighbourhood} with easy access to public transport, local cafés, supermarkets, and key Berlin hotspots—"
            f"so it’s simple to explore the city during the day and come back to {closing_vibe} in the evening."
        ).strip()

        st.success("Description generated ✅")
        st.markdown("### 📄 Generated Description")
        st.text_area(label="", value=description, height=320)


def investor_page_placeholder():
    model, features = load_artifacts()

    if st.button("← Back", key="back_from_investor"):
        st.session_state.role = None
        st.rerun()

    st.subheader("💼 Investor Workspace")
    st.caption("Estimate revenue, profit, and yield under different demand assumptions.")
    st.info("Fill the inputs below and click **Run investor analysis** to see results.")

    def detect_prefix(prefix_candidates):
        for p in prefix_candidates:
            if any(isinstance(f, str) and f.startswith(p) for f in features):
                return p
        return None

    neigh_prefix = detect_prefix(["neighbourhood_", "neighbourhood_cleansed_", "Neighborhood_", "Neighborhood Group_"])
    room_prefix  = detect_prefix(["Room Type_", "room_type_"])

    if neigh_prefix:
        neighbourhood_options = sorted({
            normalize_name(f.split(neigh_prefix, 1)[1])
            for f in features if isinstance(f, str) and f.startswith(neigh_prefix)
        })
    else:
        neighbourhood_options = sorted({
            normalize_name(feat.get("properties", {}).get("name"))
            for feat in berlin_geojson_norm.get("features", [])
            if feat.get("properties", {}).get("name") is not None
        })

    if "inv_manual_price" not in st.session_state:
        st.session_state.inv_manual_price = 120.0
    if "inv_override_price" not in st.session_state:
        st.session_state.inv_override_price = False

    st.markdown("---")

    with st.form("investor_form", clear_on_submit=False):
        col_prop, col_inv = st.columns([1, 1], gap="large")

        with col_prop:
            with st.container(border=True):
                st.markdown("### 🏠 Property Inputs")
                st.caption("Used to estimate the expected nightly price.")

                st.selectbox(
                    "Room Type",
                    ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
                    index=0,
                    key="inv_room_type",
                )
                st.number_input("Accommodates", 1, 20, 2, key="inv_accommodates")
                st.number_input("Bedrooms", 0, 10, 1, key="inv_bedrooms")
                st.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5, key="inv_bathrooms")
                st.number_input("Minimum Nights", 1, 365, 2, key="inv_min_nights")
                st.selectbox("Neighbourhood", neighbourhood_options, index=0, key="inv_neigh")
                st.checkbox("Instant Bookable", value=True, key="inv_instant")

        with col_inv:
            with st.container(border=True):
                st.markdown("### 💶 Investment Inputs")
                st.caption("Financial assumptions for profit & yield.")

                st.number_input(
                    "Purchase Price (€)", min_value=0.0, value=350000.0, step=5000.0, key="inv_purchase"
                )
                st.number_input(
                    "Total Monthly Costs (€)", min_value=0.0, value=650.0, step=50.0, key="inv_costs"
                )
                st.slider(
                    "Airbnb Fee (%)", 0.0, 15.0, 3.0, 0.5, key="inv_fee"
                )

        st.markdown("")

        with st.container(border=True):
            st.markdown("### 📈 Demand Assumption")
            st.caption("Choose a realistic demand level. We map it to typical occupancy rates.")

            occ_map = {"Low demand": 35, "Base demand": 60, "High demand": 80}
            demand_options = [f"{k} ({v}%)" for k, v in occ_map.items()]
            default_ix = demand_options.index(f"Base demand ({occ_map['Base demand']}%)")

            st.radio(
                "Expected Demand Level",
                demand_options,
                index=default_ix,
                horizontal=True,
                key="inv_demand_choice",
            )

            def nights(pct: int) -> int:
                return int(round(30 * pct / 100))

            st.caption(
                f"Occupancy mapping → "
                f"Low: {occ_map['Low demand']}% (~{nights(occ_map['Low demand'])} nights) · "
                f"Base: {occ_map['Base demand']}% (~{nights(occ_map['Base demand'])} nights) · "
                f"High: {occ_map['High demand']}% (~{nights(occ_map['High demand'])} nights)"
            )

        submitted_form = st.form_submit_button("Update Inputs")

    st.markdown("")

    with st.container(border=True):
        st.markdown("### 💡 Pricing Override")
        st.caption("Optionally override the model’s predicted nightly price.")

        st.session_state.inv_override_price = st.checkbox(
            "Override Nightly Price Manually",
            value=st.session_state.inv_override_price,
            key="inv_override_checkbox_outside",
        )

        st.session_state.inv_manual_price = st.number_input(
            "Manual Nightly Price (€)",
            min_value=0.0,
            value=float(st.session_state.inv_manual_price),
            step=5.0,
            disabled=not st.session_state.inv_override_price,
            key="inv_manual_price_input_outside",
        )

        if not st.session_state.inv_override_price:
            st.caption("Enable override to edit and use the manual price.")

    run = st.button("Run investor analysis", key="inv_run_btn")
    if not run:
        return

    # =========================================================
    # RESULTS
    # =========================================================
    room_type = st.session_state.get("inv_room_type", "Entire home/apt")
    accommodates = st.session_state.get("inv_accommodates", 2)
    bedrooms = st.session_state.get("inv_bedrooms", 1)
    bathrooms = st.session_state.get("inv_bathrooms", 1.0)
    minimum_nights = st.session_state.get("inv_min_nights", 2)
    neighbourhood = st.session_state.get("inv_neigh", neighbourhood_options[0] if neighbourhood_options else "Berlin")
    instant_bookable = st.session_state.get("inv_instant", True)

    purchase_price = float(st.session_state.get("inv_purchase", 350000.0) or 0.0)
    monthly_costs = float(st.session_state.get("inv_costs", 650.0) or 0.0)
    airbnb_fee_pct = float(st.session_state.get("inv_fee", 3.0) or 0.0)

    occ_map = {"Low demand": 35, "Base demand": 60, "High demand": 80}
    demand_choice = st.session_state.get("inv_demand_choice", "Base demand (60%)")
    demand_level = demand_choice.split(" (", 1)[0]
    occ_selected = occ_map.get(demand_level, 60)

    X_df = pd.DataFrame(0.0, index=[0], columns=features)
    for col, val in {
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "minimum_nights": minimum_nights,
    }.items():
        if col in X_df.columns:
            X_df.loc[0, col] = float(val)

    if room_prefix:
        rt_col = f"{room_prefix}{room_type}"
        if rt_col in X_df.columns:
            X_df.loc[0, rt_col] = 1.0

    neighbourhood_norm = normalize_name(neighbourhood)
    if neigh_prefix:
        n_col = f"{neigh_prefix}{neighbourhood_norm}"
        if n_col in X_df.columns:
            X_df.loc[0, n_col] = 1.0

    if "instant_bookable" in X_df.columns:
        X_df.loc[0, "instant_bookable"] = int(bool(instant_bookable))

    lat_col = next((c for c in X_df.columns if isinstance(c, str) and "lat" in c.lower()), None)
    lon_col = next((c for c in X_df.columns if isinstance(c, str) and "lon" in c.lower()), None)
    if lat_col and lon_col:
        lat, lon = geo_centroids.get(neighbourhood_norm, (52.52, 13.405))
        X_df.loc[0, lat_col] = float(lat)
        X_df.loc[0, lon_col] = float(lon)

    predicted_price = float(model.predict(X_df)[0])
    nightly_price = float(st.session_state.inv_manual_price) if st.session_state.inv_override_price else predicted_price

    fee = airbnb_fee_pct / 100.0
    days = 30.0
    occ = float(occ_selected) / 100.0

    monthly_gross = nightly_price * days * occ
    monthly_net = monthly_gross * (1.0 - fee) - monthly_costs
    annual_net = monthly_net * 12.0

    gross_yield = ((monthly_gross * 12.0) / purchase_price) * 100.0 if purchase_price > 0 else 0.0
    net_yield = (annual_net / purchase_price) * 100.0 if purchase_price > 0 else 0.0

    # =========================================================
    # WHITE COLORED METRICS
    # =========================================================
    st.markdown(
        """
        <style>
        div[data-testid="metric-container"]{
            background: transparent !important;
            text-align: center !important;
        }

        div[data-testid="metric-container"] *{
            opacity: 1 !important;
            filter: none !important;
            text-shadow: none !important;
        }

        /* Label */
        div[data-testid="metric-container"] [data-testid="stMetricLabel"] *{
            color: rgba(248,250,252,0.95) !important;
            -webkit-text-fill-color: rgba(248,250,252,0.95) !important;
            font-weight: 650 !important;
            text-align: center !important;
        }

        /* Value */
        div[data-testid="metric-container"] [data-testid="stMetricValue"] *{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 900 !important;
            text-align: center !important;
            line-height: 1.05 !important;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] p{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        /* Delta */
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] *{
            color: rgba(226,232,240,0.9) !important;
            -webkit-text-fill-color: rgba(226,232,240,0.9) !important;
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Key Results
    # -----------------------------
    st.markdown("### 📊 Key Results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nightly Price", f"€{nightly_price:,.0f}")
    k2.metric("Monthly Net Profit", f"€{monthly_net:,.0f}")
    k3.metric("Gross Yield", f"{gross_yield:.2f}%")
    k4.metric("Net Yield", f"{net_yield:.2f}%")

    summary_text = (
        f"Based on the selected demand scenario ({demand_level.lower()} / {occ_selected}% occupancy), "
        f"the property is estimated to earn approximately €{monthly_net:,.0f} net profit per month "
        f"using a nightly price of €{nightly_price:,.0f}. This implies a gross yield of {gross_yield:.2f}% "
        f"and a net yield of {net_yield:.2f}% per year, after platform fees and recurring monthly costs. "
        f"These figures are indicative estimates and may vary with seasonality, reviews, and operational efficiency."
    )

    st.markdown(
        f"""
        <div style="
            margin-top:18px;
            padding:16px 20px;
            border-radius:14px;
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(148,163,184,0.25);
            color: rgba(226,232,240,0.95);
            font-size: 14px;
            line-height: 1.65;
        ">
            {summary_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🗺️ Berlin Price Heatmap")
    render_price_heatmap(neighbourhood_norm)


# -----------------------------
# Main
# -----------------------------
if st.session_state.role is None:
    landing()
elif st.session_state.role == "host":
    host_page_placeholder()
elif st.session_state.role == "investor":
    investor_page_placeholder()
elif st.session_state.role == "description":
    description_page()
