import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="BookVerse | Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* ================================
   MAIN APP BACKGROUND
================================ */
.stApp {
    background-color: #0B1F33;
}


/* ================================
   SIDEBAR
================================ */
[data-testid="stSidebar"] {
    background-color: #061525;
}

[data-testid="stSidebar"] * {
    color: white !important;
}


/* ================================
   MAIN TITLE
================================ */
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #FFFFFF;
}

.subtitle {
    color: #B8C7D9;
    font-size: 15px;
}


/* ================================
   METRIC CARDS
================================ */
.metric-card {
    background: linear-gradient(135deg, #163A5F, #245A87);
    padding: 22px;
    border-radius: 15px;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.30);
    border: 1px solid #2F638D;
}

.metric-card h4 {
    color: #C9D9E8;
    margin: 0;
    font-size: 14px;
}

.metric-card h2 {
    color: white;
    font-size: 28px;
    margin: 8px 0;
}


/* ================================
   SECTION TITLES
================================ */
.section-title {
    color: #FFFFFF;
    font-size: 23px;
    font-weight: 700;
}


/* ================================
   BOOK CARDS
================================ */
.book-card {
    background: #132F4C;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.30);
    min-height: 190px;
    border: 1px solid #285273;
    color: white;
}


/* ================================
   STREAMLIT METRICS
================================ */
div[data-testid="stMetric"] {
    background: #132F4C;
    padding: 15px;
    border-radius: 12px;
    border-left: 4px solid #4D9ACC;
    box-shadow: 0 2px 8px rgba(0,0,0,0.30);
}

div[data-testid="stMetric"] label {
    color: #B8C7D9 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}


/* ================================
   GENERAL TEXT
================================ */
.stApp p,
.stApp span,
.stApp label {
    color: #E6EEF5;
}


/* ================================
   INPUT BOXES
================================ */
.stTextInput input,
.stNumberInput input,
.stSelectbox div,
.stMultiSelect div {
    background-color: #132F4C !important;
    color: white !important;
    border-color: #315D7D !important;
}


/* ================================
   DATAFRAME
================================ */
[data-testid="stDataFrame"] {
    background-color: #132F4C;
}


/* ================================
   BUTTONS
================================ */
.stButton > button {
    background-color: #245A87;
    color: white;
    border: 1px solid #4D9ACC;
    border-radius: 8px;
}

.stButton > button:hover {
    background-color: #3479A9;
    color: white;
}


/* ================================
   DIVIDERS
================================ */
hr {
    border-color: #315D7D;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data(show_spinner="Loading datasets...")
def load_data():

    ratings_path = BASE_DIR / "Ratings.csv"
    users_path = BASE_DIR / "Users.csv"
    books_path = BASE_DIR / "Books.csv"
    books = pd.read_csv(
    books_path,
    low_memory=False
)


    # Check files
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings.csv not found: {ratings_path}"
        )

    if not users_path.exists():
        raise FileNotFoundError(
            f"Users.csv not found: {users_path}"
        )

    if not books_path.exists():
        raise FileNotFoundError(
            f"Books_final.csv not found: {books_path}"
        )

    # Load datasets
    ratings = pd.read_csv(
        ratings_path,
        dtype={"ISBN": str},
        low_memory=False
    )

    users = pd.read_csv(
        users_path,
        low_memory=False
    )

    books = pd.read_csv(
        books_path,
        dtype={"ISBN": str},
        low_memory=False
    )

    # Clean column names
    ratings.columns = ratings.columns.str.strip()
    users.columns = users.columns.str.strip()
    books.columns = books.columns.str.strip()

    # Remove duplicates
    ratings = ratings.drop_duplicates(
        subset=["User-ID", "ISBN"]
    )

    books = books.drop_duplicates(
        subset=["ISBN"]
    )

    users = users.drop_duplicates(
        subset=["User-ID"]
    )

    # Clean ratings
    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"],
        errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["User-ID", "ISBN", "Book-Rating"]
    )

    # Explicit ratings
    explicit = ratings[
        ratings["Book-Rating"].between(1, 10)
    ].copy()

    # Clean publication year
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"],
        errors="coerce"
    )

    books.loc[
        ~books["Year-Of-Publication"].between(1000, 2026),
        "Year-Of-Publication"
    ] = np.nan

    # Clean age
    users["Age"] = pd.to_numeric(
        users["Age"],
        errors="coerce"
    )

    users.loc[
        ~users["Age"].between(5, 100),
        "Age"
    ] = np.nan

    # Fill missing text
    text_cols = [
        "Book-Title",
        "Book-Author",
        "Publisher"
    ]

    for col in text_cols:
        if col in books.columns:
            books[col] = (
                books[col]
                .fillna("Unknown")
                .astype(str)
            )

    return ratings, explicit, users, books


# --------------------------------------------------
# LOAD DATA SAFELY
# --------------------------------------------------

try:

    ratings, explicit, users, books = load_data()

except Exception as e:

    st.error(f"Error loading datasets: {e}")

    st.info(
        "Please make sure Ratings.csv, Users.csv and "
        "Books_final.csv are in the same GitHub folder as app.py."
    )

    st.stop()


# --------------------------------------------------
# PREPARE ANALYTICS DATA
# --------------------------------------------------

@st.cache_data
def prepare_data(ratings, explicit, users, books):

    # Book rating statistics
    book_stats = (
        explicit.groupby("ISBN")
        .agg(
            Average_Rating=("Book-Rating", "mean"),
            Rating_Count=("Book-Rating", "count"),
            Unique_Users=("User-ID", "nunique")
        )
        .reset_index()
    )

    # Merge book information
    book_stats = book_stats.merge(
        books,
        on="ISBN",
        how="left"
    )

    # Weighted rating
    if not book_stats.empty:

        C = book_stats["Average_Rating"].mean()
        m = book_stats["Rating_Count"].quantile(0.75)

        if m == 0 or pd.isna(m):
            m = 1

        book_stats["Weighted_Rating"] = (
            (
                book_stats["Rating_Count"]
                /
                (book_stats["Rating_Count"] + m)
            )
            * book_stats["Average_Rating"]
            +
            (
                m
                /
                (book_stats["Rating_Count"] + m)
            )
            * C
        )

    else:

        book_stats["Weighted_Rating"] = 0

    # Author statistics
    author_stats = (
        book_stats.groupby("Book-Author")
        .agg(
            Total_Ratings=("Rating_Count", "sum"),
            Total_Books=("ISBN", "nunique"),
            Average_Rating=("Average_Rating", "mean")
        )
        .reset_index()
    )

    # Publisher statistics
    publisher_stats = (
        book_stats.groupby("Publisher")
        .agg(
            Total_Books=("ISBN", "nunique"),
            Total_Ratings=("Rating_Count", "sum")
        )
        .reset_index()
    )

    # User statistics
    user_stats = (
        explicit.groupby("User-ID")
        .agg(
            Total_Ratings=("Book-Rating", "count"),
            Average_Rating=("Book-Rating", "mean"),
            Highest_Rating=("Book-Rating", "max")
        )
        .reset_index()
    )

    user_stats = user_stats.merge(
        users,
        on="User-ID",
        how="left"
    )

    # Age groups
    user_stats["Age_Group"] = pd.cut(
        user_stats["Age"],
        bins=[0, 18, 25, 35, 45, 60, 100],
        labels=[
            "Under 18",
            "18-25",
            "26-35",
            "36-45",
            "46-60",
            "60+"
        ]
    )

    return (
        book_stats,
        author_stats,
        publisher_stats,
        user_stats
    )


book_stats, author_stats, publisher_stats, user_stats = prepare_data(
    ratings,
    explicit,
    users,
    books
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("# 📚 BookVerse")
st.sidebar.caption("Book Recommendation & Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Overview",
        "📊 Book Analytics",
        "👥 User Analytics",
        "🔍 Book Explorer",
        "🎯 Recommendations"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔎 Global Filters")


# --------------------------------------------------
# YEAR FILTER
# --------------------------------------------------

years = sorted(
    book_stats["Year-Of-Publication"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

if years:

    selected_years = st.sidebar.slider(
        "Publication Year",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years))
    )

else:

    selected_years = (1900, 2026)


# --------------------------------------------------
# RATING FILTER
# --------------------------------------------------

min_ratings = st.sidebar.slider(
    "Minimum Ratings",
    min_value=1,
    max_value=100,
    value=5
)


# --------------------------------------------------
# FILTER BOOKS
# --------------------------------------------------

filtered_books = book_stats[
    book_stats["Year-Of-Publication"].between(
        selected_years[0],
        selected_years[1]
    )
    &
    (
        book_stats["Rating_Count"] >= min_ratings
    )
].copy()


st.sidebar.markdown("---")

st.sidebar.caption(
    "Developed using Python, Streamlit & Plotly"
)


# --------------------------------------------------
# HELPER FUNCTION - METRIC CARD
# --------------------------------------------------

def metric_card(title, value, icon):

    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{icon} {title}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# HELPER FUNCTION - BOOK CARDS
# --------------------------------------------------

def display_book_cards(data, limit=5):

    if data.empty:

        st.warning(
            "No books found for the selected filters."
        )

        return

    number_of_cards = min(
        limit,
        len(data)
    )

    cols = st.columns(number_of_cards)

    for i, (_, row) in enumerate(
        data.head(limit).iterrows()
    ):

        with cols[i]:

            st.markdown(
                "### 📚"
            )

            title = str(
                row.get(
                    "Book-Title",
                    "Unknown Book"
                )
            )

            author = str(
                row.get(
                    "Book-Author",
                    "Unknown Author"
                )
            )

            st.markdown(
                f"**{title[:65]}**"
            )

            st.caption(
                f"✍️ {author}"
            )

            if pd.notna(
                row.get("Average_Rating")
            ):

                st.write(
                    f"⭐ {row['Average_Rating']:.2f}/10"
                )

            rating_count = row.get(
                "Rating_Count",
                0
            )

            st.caption(
                f"Ratings: {int(rating_count):,}"
            )


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        '📚 BookVerse Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore books, readers, ratings and personalized recommendations.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Total Books",
            f"{books['ISBN'].nunique():,}",
            "📚"
        )

    with c2:

        metric_card(
            "Total Readers",
            f"{users['User-ID'].nunique():,}",
            "👥"
        )

    with c3:

        metric_card(
            "Total Ratings",
            f"{len(ratings):,}",
            "⭐"
        )

    with c4:

        avg_rating = explicit[
            "Book-Rating"
        ].mean()

        metric_card(
            "Average Rating",
            f"{avg_rating:.2f}/10",
            "🌟"
        )

    st.write("")

    st.markdown(
        '<div class="section-title">'
        '📈 Rating Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Rating distribution
    with col1:

        rating_dist = (
            explicit["Book-Rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_dist.columns = [
            "Rating",
            "Count"
        ]

        fig = px.bar(
            rating_dist,
            x="Rating",
            y="Count",
            text_auto=".2s",
            title="Book Rating Distribution",
            color="Count",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            template="plotly_dark",
            height=380,
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Rating category pie chart
    with col2:

        rating_category = explicit.copy()

        rating_category["Category"] = pd.cut(
            rating_category["Book-Rating"],
            bins=[0, 3, 6, 8, 10],
            labels=[
                "Low (1-3)",
                "Average (4-6)",
                "Good (7-8)",
                "Excellent (9-10)"
            ]
        )

        category_data = (
            rating_category["Category"]
            .value_counts()
            .reset_index()
        )

        category_data.columns = [
            "Category",
            "Count"
        ]

        fig = px.pie(
            category_data,
            names="Category",
            values="Count",
            hole=0.55,
            title="Rating Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            template="plotly_dark",
            height=380
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">'
        '🏆 Most Popular Books'
        '</div>',
        unsafe_allow_html=True
    )

    popular = (
        filtered_books
        .sort_values(
            "Weighted_Rating",
            ascending=False
        )
    )

    display_book_cards(
        popular,
        5
    )

    st.write("")

    st.markdown(
        '<div class="section-title">'
        '✍️ Top Authors'
        '</div>',
        unsafe_allow_html=True
    )

    top_authors = (
        author_stats
        .sort_values(
            "Total_Ratings",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top_authors,
        x="Total_Ratings",
        y="Book-Author",
        orientation="h",
        title="Top 10 Authors by Number of Ratings",
        color="Total_Ratings",
        color_continuous_scale="Teal"
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        yaxis={
            "categoryorder": "total ascending"
        },
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# PAGE 2: BOOK ANALYTICS
# ==================================================

elif page == "📊 Book Analytics":

    st.title(
        "📊 Book Analytics & EDA"
    )

    st.caption(
        "Understand book popularity, ratings, publication and authors."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Books in Filter",
        f"{len(filtered_books):,}"
    )

    col2.metric(
        "Average Rating",
        (
            f"{filtered_books['Average_Rating'].mean():.2f}"
            if not filtered_books.empty
            else "N/A"
        )
    )

    col3.metric(
        "Total Explicit Ratings",
        f"{filtered_books['Rating_Count'].sum():,}"
    )

    st.markdown(
        "### 📅 Books Published Per Year"
    )

    year_data = (
        filtered_books
        .groupby("Year-Of-Publication")
        .size()
        .reset_index(
            name="Number of Books"
        )
    )

    if not year_data.empty:

        year_data[
            "Year-Of-Publication"
        ] = (
            year_data[
                "Year-Of-Publication"
            ].astype(int)
        )

        fig = px.area(
            year_data,
            x="Year-Of-Publication",
            y="Number of Books",
            title="Publication Trend",
            markers=True
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### ⭐ Top Rated Books"
        )

        top_rated = (
            filtered_books
            .sort_values(
                "Weighted_Rating",
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(
            top_rated,
            x="Weighted_Rating",
            y="Book-Title",
            orientation="h",
            color="Average_Rating",
            title="Top 15 Books by Weighted Rating",
            hover_data=[
                "Rating_Count",
                "Book-Author"
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### 🔥 Most Rated Books"
        )

        most_rated = (
            filtered_books
            .nlargest(
                15,
                "Rating_Count"
            )
        )

        fig = px.bar(
            most_rated,
            x="Rating_Count",
            y="Book-Title",
            orientation="h",
            color="Rating_Count",
            title="Top 15 Most Rated Books",
            hover_data=[
                "Average_Rating",
                "Book-Author"
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        "### ✍️ Author Performance"
    )

    top_author = (
        author_stats
        .nlargest(
            15,
            "Total_Ratings"
        )
    )

    fig = px.scatter(
        top_author,
        x="Total_Books",
        y="Average_Rating",
        size="Total_Ratings",
        color="Average_Rating",
        hover_name="Book-Author",
        title="Author Performance: Books vs Average Rating",
        size_max=45,
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        template="plotly_dark",
        height=480
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### 🏢 Top Publishers"
    )

    top_publishers = (
        publisher_stats
        .nlargest(
            15,
            "Total_Books"
        )
    )

    fig = px.bar(
        top_publishers,
        x="Publisher",
        y="Total_Books",
        color="Total_Books",
        title="Top 15 Publishers by Book Count"
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### 📋 Book Analytics Table"
    )

    display_cols = [
        "Book-Title",
        "Book-Author",
        "Publisher",
        "Year-Of-Publication",
        "Average_Rating",
        "Rating_Count",
        "Weighted_Rating"
    ]

    st.dataframe(
        filtered_books[
            display_cols
        ].sort_values(
            "Weighted_Rating",
            ascending=False
        ),
        use_container_width=True,
        height=400
    )

    csv = (
        filtered_books
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Book Analytics CSV",
        csv,
        "book_analytics.csv",
        "text/csv"
    )


# ==================================================
# PAGE 3: USER ANALYTICS
# ==================================================

elif page == "👥 User Analytics":

    st.title(
        "👥 Reader & User Analytics"
    )

    st.caption(
        "Explore reader activity, age distribution and rating behaviour."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Active Readers",
        f"{len(user_stats):,}"
    )

    c2.metric(
        "Average Ratings per Reader",
        f"{user_stats['Total_Ratings'].mean():.2f}"
    )

    c3.metric(
        "Average Reader Age",
        f"{user_stats['Age'].mean():.1f}"
    )

    col1, col2 = st.columns(2)

    with col1:

        age_data = (
            user_stats["Age_Group"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        age_data.columns = [
            "Age Group",
            "Readers"
        ]

        fig = px.bar(
            age_data,
            x="Age Group",
            y="Readers",
            color="Readers",
            title="Readers by Age Group",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            user_stats.dropna(
                subset=["Age"]
            ),
            x="Age",
            nbins=30,
            title="Age Distribution of Readers",
            color_discrete_sequence=[
                "#456B8C"
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        "### 🌍 Reader Locations"
    )

    locations = (
        users["Location"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",")
        .str[-1]
        .str.strip()
        .str.title()
        .value_counts()
        .head(15)
        .reset_index()
    )

    locations.columns = [
        "Country",
        "Readers"
    ]

    fig = px.bar(
        locations,
        x="Readers",
        y="Country",
        orientation="h",
        title="Top 15 Reader Countries",
        color="Readers",
        color_continuous_scale="Teal"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        yaxis={
            "categoryorder": "total ascending"
        },
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### ⭐ Reader Activity"
    )

    activity = user_stats.copy()

    activity["Activity"] = pd.cut(
        activity["Total_Ratings"],
        bins=[
            0,
            5,
            20,
            50,
            100,
            np.inf
        ],
        labels=[
            "1-5",
            "6-20",
            "21-50",
            "51-100",
            "100+"
        ]
    )

    activity_data = (
        activity["Activity"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    activity_data.columns = [
        "Ratings Given",
        "Readers"
    ]

    fig = px.pie(
        activity_data,
        names="Ratings Given",
        values="Readers",
        hole=0.5,
        title="Reader Activity Segmentation"
    )

    fig.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### 🏆 Top Active Readers"
    )

    st.dataframe(
        user_stats
        .sort_values(
            "Total_Ratings",
            ascending=False
        )
        .head(100),
        use_container_width=True
    )


# ==================================================
# PAGE 4: BOOK EXPLORER
# ==================================================

elif page == "🔍 Book Explorer":

    st.title(
        "🔍 Book Explorer"
    )

    st.caption(
        "Search, filter and discover books from the complete dataset."
    )

    search = st.text_input(
        "🔎 Search Book Title or Author",
        placeholder="Enter book name or author..."
    )

    col1, col2 = st.columns(2)

    with col1:

        authors = sorted(
            books["Book-Author"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_author = st.selectbox(
            "Select Author",
            ["All Authors"] + authors
        )

    with col2:

        publishers = sorted(
            books["Publisher"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_publisher = st.selectbox(
            "Select Publisher",
            ["All Publishers"] + publishers
        )

    explorer = filtered_books.copy()

    # Search
    if search:

        explorer = explorer[
            explorer["Book-Title"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
            |
            explorer["Book-Author"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # Author filter
    if selected_author != "All Authors":

        explorer = explorer[
            explorer["Book-Author"]
            == selected_author
        ]

    # Publisher filter
    if selected_publisher != "All Publishers":

        explorer = explorer[
            explorer["Publisher"]
            == selected_publisher
        ]

    st.write(
        f"### 📚 Found {len(explorer):,} books"
    )

    sort_option = st.selectbox(
        "Sort By",
        [
            "Highest Rated",
            "Most Popular",
            "Newest Published",
            "Oldest Published"
        ]
    )

    # Sorting
    if sort_option == "Highest Rated":

        explorer = explorer.sort_values(
            "Weighted_Rating",
            ascending=False
        )

    elif sort_option == "Most Popular":

        explorer = explorer.sort_values(
            "Rating_Count",
            ascending=False
        )

    elif sort_option == "Newest Published":

        explorer = explorer.sort_values(
            "Year-Of-Publication",
            ascending=False
        )

    else:

        explorer = explorer.sort_values(
            "Year-Of-Publication",
            ascending=True
        )

    # --------------------------------------------------
    # IMPORTANT:
    # Image-URL-M REMOVED
    # because Books_final.csv doesn't contain it.
    # --------------------------------------------------

    explorer_display_cols = [
        "Book-Title",
        "Book-Author",
        "Publisher",
        "Year-Of-Publication",
        "Average_Rating",
        "Rating_Count"
    ]

    st.dataframe(
        explorer[
            explorer_display_cols
        ].head(500),
        use_container_width=True,
        height=500
    )

    st.caption(
        "Showing maximum 500 matching books in the table."
    )

    # --------------------------------------------------
    # BOOK PREVIEW
    # --------------------------------------------------

    st.markdown(
        "### 📖 Book Preview"
    )

    if not explorer.empty:

        selected_isbn = st.selectbox(
            "Choose a book",
            explorer["ISBN"]
            .astype(str)
            .tolist(),

            format_func=lambda x: str(
                explorer.loc[
                    explorer["ISBN"]
                    .astype(str)
                    == x,
                    "Book-Title"
                ].iloc[0]
            )[:90]
        )

        selected_book = explorer[
            explorer["ISBN"]
            .astype(str)
            == selected_isbn
        ].iloc[0]

        col1, col2 = st.columns(
            [1, 3]
        )

        with col1:

            st.markdown(
                """
                <div style="
                    background:#132F4C;
                    padding:35px;
                    border-radius:12px;
                    text-align:center;
                    font-size:60px;
                ">
                📚
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.subheader(
                selected_book["Book-Title"]
            )

            st.write(
                f"**Author:** "
                f"{selected_book['Book-Author']}"
            )

            st.write(
                f"**Publisher:** "
                f"{selected_book['Publisher']}"
            )

            st.write(
                f"**ISBN:** "
                f"{selected_book['ISBN']}"
            )

            publication_year = (
                selected_book[
                    "Year-Of-Publication"
                ]
            )

            if pd.notna(publication_year):

                st.write(
                    f"**Publication Year:** "
                    f"{int(publication_year)}"
                )

            else:

                st.write(
                    "**Publication Year:** Unknown"
                )

            st.write(
                f"**Average Rating:** "
                f"⭐ {selected_book['Average_Rating']:.2f}"
            )

            st.write(
                f"**Total Ratings:** "
                f"{selected_book['Rating_Count']:,}"
            )


# ==================================================
# PAGE 5: RECOMMENDATIONS
# ==================================================

elif page == "🎯 Recommendations":

    st.title(
        "🎯 Personalized Book Recommendation"
    )

    st.caption(
        "Discover books based on your favourite book, "
        "author and community popularity."
    )

    recommendation_method = st.radio(
        "Choose Recommendation Method",
        [
            "Similar Author",
            "Popular Books",
            "Top Rated Books",
            "Find Books for a User"
        ],
        horizontal=True
    )

    st.markdown("---")

    # --------------------------------------------------
    # SIMILAR AUTHOR
    # --------------------------------------------------

    if recommendation_method == "Similar Author":

        st.subheader(
            "📚 Find Books by Similar Author"
        )

        book_names = (
            books["Book-Title"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        chosen_book = st.selectbox(
            "Select Your Favourite Book",
            book_names
        )

        selected_rows = books[
            books["Book-Title"]
            == chosen_book
        ]

        if selected_rows.empty:

            st.warning(
                "Book information not found."
            )

        else:

            selected_row = selected_rows.iloc[0]

            author = selected_row[
                "Book-Author"
            ]

            st.success(
                f"Author of selected book: {author}"
            )

            recommendations = filtered_books[
                filtered_books["Book-Author"]
                == author
            ].copy()

            recommendations = recommendations[
                recommendations["Book-Title"]
                != chosen_book
            ]

            recommendations = recommendations.sort_values(
                "Weighted_Rating",
                ascending=False
            )

            st.markdown(
                "### ✨ Recommended Books"
            )

            display_book_cards(
                recommendations,
                5
            )

            st.dataframe(
                recommendations[
                    [
                        "Book-Title",
                        "Book-Author",
                        "Average_Rating",
                        "Rating_Count"
                    ]
                ].head(50),
                use_container_width=True
            )

    # --------------------------------------------------
    # POPULAR BOOKS
    # --------------------------------------------------

    elif recommendation_method == "Popular Books":

        st.subheader(
            "🔥 Community Favourite Books"
        )

        recommendations = (
            filtered_books
            .sort_values(
                "Rating_Count",
                ascending=False
            )
        )

        display_book_cards(
            recommendations,
            5
        )

        st.dataframe(
            recommendations[
                [
                    "Book-Title",
                    "Book-Author",
                    "Average_Rating",
                    "Rating_Count"
                ]
            ].head(50),
            use_container_width=True
        )

    # --------------------------------------------------
    # TOP RATED BOOKS
    # --------------------------------------------------

    elif recommendation_method == "Top Rated Books":

        st.subheader(
            "⭐ Highest Rated Books"
        )

        recommendations = (
            filtered_books
            .sort_values(
                "Weighted_Rating",
                ascending=False
            )
        )

        display_book_cards(
            recommendations,
            5
        )

        st.dataframe(
            recommendations[
                [
                    "Book-Title",
                    "Book-Author",
                    "Average_Rating",
                    "Rating_Count",
                    "Weighted_Rating"
                ]
            ].head(50),
            use_container_width=True
        )

    # --------------------------------------------------
    # USER BASED RECOMMENDATIONS
    # --------------------------------------------------

    else:

        st.subheader(
            "👤 User-Based Book Discovery"
        )

        available_users = (
            explicit["User-ID"]
            .unique()
        )

        if len(available_users) == 0:

            st.warning(
                "No users with explicit ratings found."
            )

        else:

            user_id = st.number_input(
                "Enter User ID",
                min_value=1,
                max_value=int(
                    explicit["User-ID"].max()
                ),
                value=int(
                    available_users[0]
                ),
                step=1
            )

            user_ratings = explicit[
                explicit["User-ID"]
                == user_id
            ]

            if user_ratings.empty:

                st.warning(
                    "No explicit ratings found for this user."
                )

            else:

                st.success(
                    f"Found {len(user_ratings)} "
                    f"ratings for User ID {user_id}"
                )

                history = user_ratings.merge(
                    books,
                    on="ISBN",
                    how="left"
                )

                history = history.sort_values(
                    "Book-Rating",
                    ascending=False
                )

                st.markdown(
                    "### ❤️ Your Highly Rated Books"
                )

                st.dataframe(
                    history[
                        [
                            "Book-Title",
                            "Book-Author",
                            "Book-Rating",
                            "ISBN"
                        ]
                    ].head(20),
                    use_container_width=True
                )

                # Favourite books
                favourite = history[
                    history["Book-Rating"] >= 7
                ]

                if favourite.empty:

                    favourite = history.head(5)

                favourite_authors = (
                    favourite["Book-Author"]
                    .dropna()
                    .value_counts()
                    .head(5)
                    .index
                    .tolist()
                )

                # Already rated books
                already_read = set(
                    user_ratings["ISBN"]
                )

                recommendations = filtered_books[
                    filtered_books["Book-Author"]
                    .isin(
                        favourite_authors
                    )
                    &
                    ~filtered_books["ISBN"]
                    .isin(already_read)
                ].copy()

                recommendations = (
                    recommendations
                    .sort_values(
                        "Weighted_Rating",
                        ascending=False
                    )
                )

                st.markdown(
                    "### 🎁 Recommended For You"
                )

                if recommendations.empty:

                    st.info(
                        "No matching recommendations found. "
                        "Try the Popular Books recommendation."
                    )

                else:

                    display_book_cards(
                        recommendations,
                        5
                    )

                    st.dataframe(
                        recommendations[
                            [
                                "Book-Title",
                                "Book-Author",
                                "Average_Rating",
                                "Rating_Count"
                            ]
                        ].head(50),
                        use_container_width=True
                    )

    st.markdown("---")

    st.info(
        "Note: Similar Author and User Discovery use "
        "author-based filtering. Popular and Top Rated "
        "recommendations use rating statistics."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#B8C7D9;
        padding:15px;
    ">
        📚 <b>BookVerse</b> | Book Recommendation System<br>
        Built with Python • Pandas • Plotly • Streamlit<br>
        Data Science Project
    </div>
    """,
    unsafe_allow_html=True
)
