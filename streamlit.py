import streamlit as st
import pandas as pd
import numpy as np
from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Store",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - No constraints on detail images
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    /* Product grid images - exact 200px */
    [data-testid="stImage"] img {
        max-height: 200px;
        width: auto;
        object-fit: contain;
    }

    div[data-testid="column"] > div {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        min-height: 420px;
    }

    div[data-testid="column"] > div:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    raw_data = pd.read_csv("clean_data.csv")
    return process_data(raw_data)

# Initialize session state
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'selected_product_name' not in st.session_state:
    st.session_state.selected_product_name = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = 0

# Load data
try:
    data = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file 'clean_data.csv' not found.")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.title("🛍️ User Settings")

    user_id = st.number_input(
        "Enter User ID",
        min_value=0,
        value=st.session_state.user_id,
        step=1,
        help="User ID 0 = New User"
    )
    st.session_state.user_id = user_id

    st.markdown("---")

    if user_id != 0:
        if user_id in data['ID'].values:
            user_products_count = len(data[data['ID'] == user_id]['Name'].unique())
            st.success(f"✅ User Found")
            st.info(f"🛒 {user_products_count} purchases")
        else:
            st.warning("⚠️ User not found")
            st.info("Showing trending products")
    else:
        st.info("👋 New User Mode")
        st.caption("Explore trending products!")

# Main Header
st.markdown('<div class="main-header">🛍️ E-Commerce Store</div>', unsafe_allow_html=True)

# Search Bar
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_query = st.text_input(
        "🔍 Search Products",
        placeholder="Search for products by name, brand, or category...",
        key="search_input",
        label_visibility="collapsed"
    )

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        if st.button("🔍 Search", use_container_width=True):
            st.session_state.search_query = search_query
    with search_col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.search_query = ""
            st.rerun()

st.markdown("---")

# Product Detail Modal - Get FULL product details from dataset
def show_product_detail(product_name):
    st.markdown("### 📦 Product Details")

    # Get COMPLETE product data from the dataset
    product_data = data[data['Name'] == product_name].iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        # Display FULL SIZE image without any constraints
        if pd.notna(product_data['ImageURL']) and product_data['ImageURL']:
            # No width parameter = full original size
            st.image(product_data['ImageURL'])
        else:
            st.image("https://via.placeholder.com/500x500?text=No+Image")

    with col2:
        st.markdown(f"### {product_data['Name']}")

        # Brand
        if 'Brand' in product_data.index and pd.notna(product_data['Brand']):
            st.markdown(f"**Brand:** {product_data['Brand']}")

        # Rating
        if 'Rating' in product_data.index and pd.notna(product_data['Rating']):
            st.markdown(f"**Rating:** ⭐ {product_data['Rating']:.1f}/5.0")

        # Reviews
        if 'ReviewCount' in product_data.index and pd.notna(product_data['ReviewCount']):
            st.markdown(f"**Reviews:** {int(product_data['ReviewCount']):,}")

        st.markdown("---")

        # Description - Check if exists and has content
        if 'Description' in product_data.index:
            description = str(product_data['Description'])
            if description and description.strip() and description.lower() not in ['nan', 'none', '']:
                st.markdown("**Description:**")
                st.write(description)
            else:
                st.info("No description available for this product.")
        else:
            st.info("No description available for this product.")

        # Category
        if 'Category' in product_data.index and pd.notna(product_data['Category']):
            category = str(product_data['Category'])
            if category and category.strip() and category.lower() not in ['nan', 'none', '']:
                st.markdown(f"**Category:** {category}")

    if st.button("← Back to Products", use_container_width=True):
        st.session_state.selected_product_name = None
        st.rerun()

# Display Product Card
def display_product_card(product, col, idx):
    with col:
        # Image
        if 'ImageURL' in product.index and pd.notna(product['ImageURL']) and product['ImageURL']:
            st.image(product['ImageURL'], use_container_width=True)
        else:
            st.image("https://via.placeholder.com/200x200?text=No+Image", use_container_width=True)

        # Product Name
        product_name = str(product['Name'])
        if len(product_name) > 50:
            product_name = product_name[:50] + "..."
        st.markdown(f"**{product_name}**")

        # Brand
        brand = product.get('Brand', 'Unknown')
        if pd.notna(brand) and brand:
            brand_text = str(brand)[:25] + "..." if len(str(brand)) > 25 else str(brand)
            st.caption(f"Brand: {brand_text}")

        # Rating
        rating = product.get('Rating', 0)
        review_count = int(product.get('ReviewCount', 0))
        st.markdown(f"⭐ **{rating:.1f}** ({review_count:,})")

        # View Details Button
        if st.button(f"View Details", key=f"btn_{idx}", use_container_width=True):
            # Store the product NAME to fetch full details later
            st.session_state.selected_product_name = product['Name']
            st.rerun()

# Search Results
def display_search_results(search_query):
    if search_query:
        mask = (
            data['Name'].str.contains(search_query, case=False, na=False) |
            data['Brand'].str.contains(search_query, case=False, na=False)
        )

        if 'Category' in data.columns:
            mask |= data['Category'].str.contains(search_query, case=False, na=False)

        search_results = data[mask].drop_duplicates(subset=['Name']).head(20)

        if not search_results.empty:
            st.markdown(f'<div class="section-header">🔍 Search Results for "{search_query}"</div>', unsafe_allow_html=True)
            st.caption(f"Found {len(search_results)} products")

            for i in range(0, len(search_results), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(search_results):
                        product = search_results.iloc[i + j]
                        display_product_card(product, col, f"search_{i+j}")
            return True
        else:
            st.warning(f"No products found matching '{search_query}'")
            return False
    return False

# Main Content
if st.session_state.selected_product_name is not None:
    show_product_detail(st.session_state.selected_product_name)
else:
    if st.session_state.search_query:
        display_search_results(st.session_state.search_query)
    else:
        # For New Users
        if user_id == 0:
            st.markdown('<div class="section-header">🔥 Trending Products</div>', unsafe_allow_html=True)

            try:
                trending_products = get_top_rated_items(data, top_n=12)

                if not trending_products.empty:
                    for i in range(0, min(12, len(trending_products)), 4):
                        cols = st.columns(4)
                        for j, col in enumerate(cols):
                            if i + j < len(trending_products):
                                product = trending_products.iloc[i + j]
                                display_product_card(product, col, f"trending_{i+j}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # For Existing Users
        if user_id != 0:
            if user_id in data['ID'].values:
                # Content-Based
                st.markdown('<div class="section-header">✨ Recommended For You</div>', unsafe_allow_html=True)
                st.caption("Products similar to your previous purchases")

                try:
                    user_purchases = data[data['ID'] == user_id]

                    if not user_purchases.empty:
                        purchased_products = user_purchases['Name'].unique()
                        last_purchase_name = purchased_products[-1]

                        if last_purchase_name in data['Name'].values:
                            content_recommendations = content_based_recommendation(
                                data, 
                                last_purchase_name, 
                                top_n=8
                            )

                            if not content_recommendations.empty:
                                st.success(f"📦 Based on: {last_purchase_name[:60]}...")

                                for i in range(0, min(8, len(content_recommendations)), 4):
                                    cols = st.columns(4)
                                    for j, col in enumerate(cols):
                                        if i + j < len(content_recommendations):
                                            product = content_recommendations.iloc[i + j]
                                            display_product_card(product, col, f"content_{i+j}")
                            else:
                                st.info("No similar products found.")
                        else:
                            st.warning("Product no longer available.")
                    else:
                        st.info("Start shopping to get recommendations!")

                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")

                # Collaborative Filtering
                st.markdown('<div class="section-header">👥 Customers Also Bought</div>', unsafe_allow_html=True)
                st.caption("Popular among users with similar taste")

                try:
                    collab_recommendations = collaborative_filtering_recommendations(data, user_id, top_n=8)

                    if not collab_recommendations.empty:
                        collab_recommendations = collab_recommendations.drop_duplicates(subset=['Name'])

                        for i in range(0, min(8, len(collab_recommendations)), 4):
                            cols = st.columns(4)
                            for j, col in enumerate(cols):
                                if i + j < len(collab_recommendations):
                                    product = collab_recommendations.iloc[i + j]
                                    display_product_card(product, col, f"collab_{i+j}")
                    else:
                        st.info("No collaborative recommendations available.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>🛍️ E-Commerce Recommendation System</p>
</div>
""", unsafe_allow_html=True)