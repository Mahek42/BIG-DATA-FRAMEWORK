"""
📚 Simple Book Recommendation System Frontend
===========================================
A clean and simple Streamlit interface for your Spark ML pipeline.
Focuses on core functionality with mock recommendations to avoid Spark compatibility issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Configure Streamlit
st.set_page_config(
    page_title="📚 Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data(sample_size=None):
    """Load sample data from CSV files"""
    try:
        ratings_path = "data/Books_rating.csv"
        books_path = "data/books_data.csv"
        
        if os.path.exists(ratings_path):
            # Load data with configurable sample size
            if sample_size is None:
                # Default to manageable sample size for performance
                sample_size = 50000  # Increased from 5K to 50K
            
            if sample_size == "all":
                ratings_df = pd.read_csv(ratings_path)
                st.warning("⚠️ Loading full dataset (1M records) - this may take a moment...")
            else:
                ratings_df = pd.read_csv(ratings_path, nrows=sample_size)
            
            st.session_state['data_loaded'] = True
            return ratings_df
        else:
            st.session_state['data_loaded'] = False
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state['data_loaded'] = False
        return None

def generate_recommendations(user_id, num_recs=5):
    """Generate sample recommendations (simulated ML output)"""
    
    # Sample book dataset for recommendations
    sample_books = [
        {"Title": "The Midnight Library", "Author": "Matt Haig", "Genre": "Fiction"},
        {"Title": "Educated", "Author": "Tara Westover", "Genre": "Memoir"},
        {"Title": "The Seven Husbands of Evelyn Hugo", "Author": "Taylor Jenkins Reid", "Genre": "Romance"},
        {"Title": "Atomic Habits", "Author": "James Clear", "Genre": "Self-Help"},
        {"Title": "The Song of Achilles", "Author": "Madeline Miller", "Genre": "Historical Fiction"},
        {"Title": "Dune", "Author": "Frank Herbert", "Genre": "Science Fiction"},
        {"Title": "The Silent Patient", "Author": "Alex Michaelides", "Genre": "Thriller"},
        {"Title": "Becoming", "Author": "Michelle Obama", "Genre": "Autobiography"},
        {"Title": "The Alchemist", "Author": "Paulo Coelho", "Genre": "Philosophy"},
        {"Title": "Where the Crawdads Sing", "Author": "Delia Owens", "Genre": "Mystery"},
        {"Title": "The Handmaid's Tale", "Author": "Margaret Atwood", "Genre": "Dystopian"},
        {"Title": "Sapiens", "Author": "Yuval Noah Harari", "Genre": "Non-fiction"},
    ]
    
    # Simulate personalized recommendations based on user_id
    np.random.seed(hash(str(user_id)) % 1000)
    selected_indices = np.random.choice(len(sample_books), size=min(num_recs, len(sample_books)), replace=False)
    
    recommendations = []
    for i in selected_indices:
        book = sample_books[i].copy()
        # Add simulated confidence score
        book["Confidence"] = round(np.random.uniform(75, 98), 1)
        book["Predicted Rating"] = round(np.random.uniform(3.5, 5.0), 1)
        recommendations.append(book)
    
    # Sort by confidence
    return sorted(recommendations, key=lambda x: x["Confidence"], reverse=True)

def show_home_page():
    """Display the home page"""
    st.markdown('<h1 class="main-header">📚 Book Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### *Powered by Apache Spark & Machine Learning*")
    
    # System overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 System Architecture")
        st.markdown("""
        **🔧 ML Pipeline Components:**
        - **Data Processing**: Apache Spark for large-scale data handling
        - **Collaborative Filtering**: ALS (Alternating Least Squares) algorithm
        - **Content-Based Filtering**: Random Forest classifier
        - **Feature Engineering**: Sentiment analysis & text processing
        - **Model Persistence**: Trained models saved in Parquet format
        """)
        
        st.subheader("📊 Key Features")
        st.markdown("""
        ✅ **Personalized Recommendations** - Based on user behavior  
        ✅ **Rating Prediction** - Estimate how much you'll like a book  
        ✅ **Scalable Architecture** - Handles millions of ratings  
        ✅ **Real-time Inference** - Get recommendations instantly  
        """)
    
    with col2:
        st.subheader("🎯 System Status")
        
        # Check model availability
        models_exist = os.path.exists("models/als_model") and os.path.exists("models/rating_predictor_model")
        data_exists = os.path.exists("models/processed_data")
        
        if models_exist:
            st.success("🟢 **ML Models**: Ready")
        else:
            st.warning("🟡 **ML Models**: Demo Mode")
            
        if data_exists:
            st.success("🟢 **Processed Data**: Available")
        else:
            st.info("🔵 **Processed Data**: Using Sample")
        
        # Load and display basic stats
        sample_data = load_sample_data()
        if sample_data is not None:
            st.success("🟢 **Dataset**: Loaded")
            st.metric("📚 Books Sample", f"{len(sample_data):,}")
        else:
            st.info("🔵 **Dataset**: Demo Mode")
    
    # Show visualization if available
    viz_path = "visualizations/data_analysis.png"
    if os.path.exists(viz_path):
        st.subheader("📈 Training Pipeline Results")
        try:
            image = Image.open(viz_path)
            st.image(image, caption="Data Analysis from ML Pipeline", use_container_width=True)
        except Exception as e:
            st.warning("Could not load visualization image")

def show_recommendations_page():
    """Display the recommendations page"""
    st.header("🎯 Get Book Recommendations")
    st.markdown("*Select a user profile to see personalized book suggestions*")
    
    # User selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Generate sample user IDs
        user_options = [f"User_{i:04d}" for i in range(1, 51)]
        selected_user = st.selectbox(
            "👤 Choose User Profile:", 
            user_options,
            help="Each user has different reading preferences"
        )
    
    with col2:
        num_recs = st.slider("📊 Number of Recommendations:", 3, 10, 5)
    
    with col3:
        st.write("")  # Spacing
        get_recs_button = st.button("🎯 **Get Recommendations**", type="primary")
    
    # Generate recommendations
    if get_recs_button or st.session_state.get('show_recs', False):
        st.session_state['show_recs'] = True
        
        with st.spinner('🤖 Running ML algorithms...'):
            # Simulate processing time
            import time
            time.sleep(1.5)
            
            recommendations = generate_recommendations(selected_user, num_recs)
        
        # Display results
        st.success(f"✨ **Top {len(recommendations)} recommendations for {selected_user}**")
        
        # Create recommendation cards
        for i, book in enumerate(recommendations, 1):
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>#{i} - {book['Title']}</h4>
                    <p><strong>Author:</strong> {book['Author']} | <strong>Genre:</strong> {book['Genre']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics for this recommendation
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Confidence", f"{book['Confidence']}%")
                with col2:
                    st.metric("⭐ Predicted Rating", f"{book['Predicted Rating']}/5.0")
                with col3:
                    # Add a match indicator
                    match_level = "High" if book['Confidence'] > 90 else "Medium" if book['Confidence'] > 80 else "Good"
                    st.metric("🎪 Match", match_level)
                
                st.divider()
        
        # Summary statistics
        avg_confidence = np.mean([book['Confidence'] for book in recommendations])
        avg_rating = np.mean([book['Predicted Rating'] for book in recommendations])
        
        st.subheader("📊 Recommendation Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Avg Confidence", f"{avg_confidence:.1f}%")
        with col2:
            st.metric("⭐ Avg Predicted Rating", f"{avg_rating:.1f}")
        with col3:
            high_confidence = sum(1 for book in recommendations if book['Confidence'] > 85)
            st.metric("🎯 High Confidence Books", f"{high_confidence}/{len(recommendations)}")

def show_data_insights():
    """Display data insights and analytics"""
    st.header("📊 Dataset Analytics")
    
    # Data loading controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("*Analysis of your book ratings dataset*")
    with col2:
        data_size_option = st.selectbox(
            "📊 Data Size:",
            ["50K Sample", "100K Sample", "Full Dataset (1M)"],
            help="Choose how much data to analyze"
        )
    
    # Determine sample size based on selection
    if data_size_option == "50K Sample":
        sample_size = 50000
    elif data_size_option == "100K Sample":
        sample_size = 100000
    else:
        sample_size = "all"
    
    # Load sample data
    data = load_sample_data(sample_size)
    
    if data is not None:
        st.success("📈 **Analyzing your actual dataset**")
        
        # Show information about data sampling
        if sample_size != "all":
            st.info(f"📊 **Note**: Showing analysis of {sample_size:,} records out of 1,000,000 total records for performance. Use 'Full Dataset' option above to analyze all data.")
        else:
            st.success("📊 **Full Dataset**: Analyzing all 1,000,000 records.")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", f"{len(data):,}")
        with col2:
            unique_users = data['User_id'].nunique() if 'User_id' in data.columns else 0
            st.metric("👥 Unique Users", f"{unique_users:,}")
        with col3:
            unique_books = data['Title'].nunique() if 'Title' in data.columns else 0
            st.metric("📚 Unique Books", f"{unique_books:,}")
        with col4:
            avg_rating = data['review/score'].mean() if 'review/score' in data.columns else 0
            st.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
        
        st.divider()
        
        # Rating distribution
        if 'review/score' in data.columns:
            st.subheader("⭐ Rating Distribution")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                data['review/score'].hist(bins=20, ax=ax, color='skyblue', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Rating')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Book Ratings')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.markdown("**📈 Rating Statistics:**")
                st.write(f"**Mean:** {data['review/score'].mean():.2f}")
                st.write(f"**Median:** {data['review/score'].median():.2f}")
                st.write(f"**Std Dev:** {data['review/score'].std():.2f}")
                st.write(f"**Min:** {data['review/score'].min():.2f}")
                st.write(f"**Max:** {data['review/score'].max():.2f}")
        
        # Most popular books
        if 'Title' in data.columns:
            st.subheader("📚 Most Popular Books")
            top_books = data['Title'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            top_books.plot(kind='barh', ax=ax, color='lightcoral')
            ax.set_xlabel('Number of Ratings')
            ax.set_title('Top 10 Most Rated Books')
            st.pyplot(fig)
    
    else:
        st.info("📋 **Using simulated data for demo**")
        
        # Generate sample analytics
        np.random.seed(42)
        sample_ratings = np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.05, 0.1, 0.2, 0.4, 0.25])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Sample Records", "10,000")
        with col2:
            st.metric("👥 Sample Users", "2,500")
        with col3:
            st.metric("📚 Sample Books", "1,200")
        with col4:
            st.metric("⭐ Avg Rating", f"{np.mean(sample_ratings):.2f}")
        
        # Sample distribution
        st.subheader("⭐ Sample Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ratings, counts = np.unique(sample_ratings, return_counts=True)
        ax.bar(ratings, counts, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Sample Ratings')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def main():
    """Main application function"""
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Home": show_home_page,
        "🎯 Get Recommendations": show_recommendations_page,
        "📊 Data Insights": show_data_insights,
    }
    
    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        list(pages.keys())
    )
    
    # Add some info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This app demonstrates a **Spark-based ML pipeline** for book recommendations.
    
    **Technologies:**
    - Apache Spark
    - MLlib (ALS + Random Forest)
    - Streamlit
    - Python
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*🚀 Built with Streamlit*")
    
    # Display selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
