# =============================================================================
# STREAMLIT RAG CHATBOT FOR PHARMA SALES DATA
# =============================================================================

import streamlit as st
import sqlite3
import pandas as pd
import openai
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import requests

# =============================================================================
# DATABASE DOWNLOAD FUNCTION
# =============================================================================

def download_database_from_gdrive():
    """Download database from Google Drive if not exists locally"""
    db_path = 'pharma_sales.db'
    
    if not os.path.exists(db_path):
        # Replace with your actual Google Drive file ID
        # Extract FILE_ID from: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        # Your folder: https://drive.google.com/drive/folders/1r1eKY_v1-9qdSNoEs5vZtB0l0wdLx0gV?usp=sharing
        # You need to get the specific file ID for pharma_sales.db
        
        file_id = "1myzQbj-IZmw8X7xK9BCoVEs3Idk43w7S"  # pharma_sales.db file ID
        gdrive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            with st.spinner("📥 Downloading database from Google Drive..."):
                # Handle large file downloads from Google Drive
                session = requests.Session()
                response = session.get(gdrive_url, stream=True)
                
                # Google Drive may redirect for large files
                if 'confirm=' in response.text:
                    # Extract confirmation token for large files
                    confirm_token = None
                    for line in response.text.splitlines():
                        if 'confirm=' in line:
                            confirm_token = line.split('confirm=')[1].split('&')[0]
                            break
                    
                    if confirm_token:
                        confirmed_url = f"{gdrive_url}&confirm={confirm_token}"
                        response = session.get(confirmed_url, stream=True)
                
                response.raise_for_status()
                
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                
                with open(db_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress_bar.progress(downloaded / total_size)
                
                progress_bar.empty()
                st.success("✅ Database downloaded successfully from Google Drive!")
                
        except Exception as e:
            st.error(f"❌ Error downloading database: {e}")
            st.info("Please check the Google Drive file ID and permissions.")
            return None
    
    return db_path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Pharma Sales Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATABASE MANAGER CLASS
# =============================================================================

class PharmaDataManager:
    def __init__(self, db_path):
        self.db_path = db_path
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query, params=None):
        try:
            conn = self.get_connection()
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            conn.close()
            return df, None
        except Exception as e:
            return None, str(e)
    
    def get_schema_info(self):
        schema_info = {
            "table_name": "pharma_sales",
            "columns": {
                "country": "TEXT - Country name (e.g., 'Mexico', 'Brazil', 'France')",
                "date": "DATE - Date of the record",
                "competitive_market": "TEXT - Market category (e.g., 'Gut Microbiota Care', 'Ear Drops')",
                "submarket_1": "TEXT - Submarket level 1 (e.g., 'Probiotic', 'Non Probiotic')",
                "corp_new": "TEXT - PRIORITY: Consolidated corporation name",
                "brands_new": "TEXT - PRIORITY: Consolidated brand name",
                "sales_euro": "REAL - Sales VALUE in EUR (PRIMARY SALES METRIC)",
                "sales_units": "INTEGER - Sales VOLUME in units/boxes",
                "year": "INTEGER - Year (2023, 2024, 2025)",
                "month": "INTEGER - Month (1-12)"
            }
        }
        return schema_info

# =============================================================================
# CHATGPT QUERY PROCESSOR CLASS
# =============================================================================

class ChatGPTQueryProcessor:
    def __init__(self, api_key, model, data_manager):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.data_manager = data_manager
        self.schema_info = data_manager.get_schema_info()
    
    def create_system_prompt(self):
        return f"""
You are a specialized SQL query generator for pharmaceutical sales data analysis with TREND ANALYSIS capabilities.

DATABASE SCHEMA:
Table: pharma_sales

EXACT COLUMN NAMES (case-sensitive):
- country: TEXT - Country name (e.g., 'Mexico', 'Brazil', 'France', 'Ukraine')
- date: DATE - Date of the record
- competitive_market: TEXT - Market category (e.g., 'Gut Microbiota Care', 'Ear Drops')
- submarket_1: TEXT - Submarket level 1 (e.g., 'Probiotic', 'Non Probiotic')
- submarket_2: TEXT - Submarket level 2
- corporation_original: TEXT - Original corporation name
- brand_original: TEXT - Original brand name
- product: TEXT - Product/SKU name
- sales_local_currency: REAL - Sales in local currency
- sales_units: INTEGER - Sales VOLUME in units/boxes
- currency: TEXT - Local currency code
- sales_euro: REAL - Sales VALUE converted to EUR (PRIMARY SALES METRIC)
- corp_new: TEXT - PRIORITY: Consolidated corporation name
- brands_new: TEXT - PRIORITY: Consolidated brand name
- year: INTEGER - Year (2023, 2024, 2025)
- month: INTEGER - Month (1-12)

KEY BUSINESS RULES:
1. ALWAYS use 'corp_new' instead of 'corporation_original' 
2. ALWAYS use 'brands_new' instead of 'brand_original'
3. 'sales_euro' is Sales VALUE (revenue in EUR)
4. 'sales_units' is Sales VOLUME (quantity in units/boxes)
5. For Biocodex queries: WHERE corp_new LIKE '%biocodex%' OR corp_new LIKE '%BIOCODEX%'
6. Brand 'Sb' is a major Biocodex brand

MANDATORY TREND ANALYSIS FOR ALL QUERIES:
- For YEARLY data: Include year-over-year comparison
- For MONTHLY data: Include month-over-month AND year-over-year
- For PERIOD data: Compare same periods

TREND ANALYSIS QUERY STRUCTURE:
```sql
WITH current_period AS (
    SELECT SUM(sales_euro) as current_value, SUM(sales_units) as current_volume
    FROM pharma_sales 
    WHERE [current period conditions]
),
previous_period AS (
    SELECT SUM(sales_euro) as previous_value, SUM(sales_units) as previous_volume
    FROM pharma_sales 
    WHERE [previous period conditions]
)
SELECT 
    cp.current_value,
    cp.current_volume,
    pp.previous_value,
    pp.previous_volume,
    ROUND(((cp.current_value - pp.previous_value) / pp.previous_value * 100), 2) as value_growth_percent,
    ROUND(((cp.current_volume - pp.previous_volume) / pp.previous_volume * 100), 2) as volume_growth_percent
FROM current_period cp, previous_period pp
```

SAMPLE COUNTRIES: Mexico, Brazil, France, Germany, Belgium, Poland, Ukraine, Russia, Turkey, US
SAMPLE BRANDS: Sb, OTIPAX, SAFORELLE, MUCOGYNE, HYDROMEGA, GALACTOGIL, SYMBIOSYS
SAMPLE MARKETS: Gut Microbiota Care, Ear Drops, Intimate Dryness, Immunity, Urinary

Return ONLY the SQL query with trend analysis, no explanations or markdown formatting.
"""
    
    def generate_sql_query(self, user_question):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": f"Convert this question to SQL: {user_question}"}
                ],
                temperature=0,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            return sql_query, None
        except Exception as e:
            return None, f"Error generating SQL query: {str(e)}"
    
    def generate_response(self, user_question, query_results, sql_query):
        try:
            if query_results is not None and not query_results.empty:
                results_text = query_results.to_string(index=False, max_rows=20)
                results_summary = f"Query returned {len(query_results)} rows"
            else:
                results_text = "No results found"
                results_summary = "Query returned no data"
            
            # Extract entities for focused analysis
            brands = ['Sb', 'OTIPAX', 'SAFORELLE', 'MUCOGYNE', 'HYDROMEGA', 'GALACTOGIL', 'SYMBIOSYS']
            countries = ['Mexico', 'Brazil', 'France', 'Germany', 'Belgium', 'Poland', 'Ukraine', 'Russia', 'Turkey', 'US']
            
            brand_mentioned = None
            country_mentioned = None
            
            for brand in brands:
                if brand.lower() in user_question.lower():
                    brand_mentioned = brand
                    break
            
            for country in countries:
                if country.lower() in user_question.lower():
                    country_mentioned = country
                    break
            
            response_prompt = f"""
You are a pharmaceutical data analyst providing FACTUAL analysis based only on database results.

USER QUESTION: {user_question}
QUERY RESULTS: {results_text}

CRITICAL TERMINOLOGY:
- sales_euro = SALES VALUE (in EUR) - this is revenue/money
- sales_units = SALES VOLUME (in units/boxes) - this is quantity sold
- Always specify VALUE (euros) vs VOLUME (units/boxes) in responses

RESPONSE STRUCTURE:
```
📊 {brand_mentioned or 'DATA'} ANALYSIS{' - ' + country_mentioned.upper() if country_mentioned else ''}
═══════════════════════════════════════
Current Performance: [State exact VALUE (€) and VOLUME (units) from database]
Growth Comparison: [Mathematical comparison - specify value vs volume growth]

🔍 KEY INSIGHTS:
• Data Observations: [What the VALUE and VOLUME numbers show]
• Comparative Analysis: [How VALUE and VOLUME compare to previous periods]
• Quantitative Trends: [Mathematical trends for both value and volume]
• Measurable Changes: [Specific % changes for VALUE and VOLUME separately]
```

PRECISE LANGUAGE:
- "Sales VALUE of €X million" (not just "sales of €X")
- "Sales VOLUME of X units" (not just "X units sold")
- "VALUE increased by X% while VOLUME changed by Y%"

FACTUAL ONLY - NO SPECULATION about marketing, customer behavior, or external factors.
Provide only data-driven insights with precise value/volume terminology.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical data analyst. Provide ONLY factual analysis based on database results. Be precise about VALUE (euros) vs VOLUME (units). Use temperature=0 for factual responses."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    # Title and description
    st.title("💊 Pharma Sales Intelligence Chatbot")
    st.markdown("Ask questions about Biocodex and competitor sales data in natural language!")
    
    # Ensure database is available
    db_path = download_database_from_gdrive()
    if not db_path:
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # OpenAI API Configuration
        st.subheader("OpenAI Settings")
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        
        model_options = [
            "gpt-4",
            "gpt-4-turbo-preview", 
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ]
        
        selected_model = st.selectbox(
            "Select ChatGPT Model",
            model_options,
            index=0,
            help="Choose the ChatGPT model for analysis"
        )
        
        # Database file check
        st.subheader("📊 Database Status")
        db_path = 'pharma_sales.db'
        
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT COUNT(*) FROM pharma_sales")
                total_records = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM pharma_sales WHERE corp_new LIKE '%biocodex%'")
                biocodex_records = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT MIN(year), MAX(year) FROM pharma_sales")
                year_range = cursor.fetchone()
                
                conn.close()
                
                st.success("✅ Database Connected")
                st.info(f"📈 Total Records: {total_records:,}")
                st.info(f"🏢 Biocodex Records: {biocodex_records:,}")
                st.info(f"📅 Data Range: {year_range[0]}-{year_range[1]}")
                
            except Exception as e:
                st.error(f"❌ Database Error: {str(e)}")
                return
        else:
            st.error("❌ Database file 'pharma_sales.db' not found!")
            st.info("Please ensure pharma_sales.db is in the same directory as this app.")
            return
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are Sb sales in Mexico in 2024?",
            "Show me Biocodex top brands by sales value",
            "Compare Sb sales across countries with growth",
            "Analyze Sb sales in Ukraine March 2025",
            "What's the trend of Sb sales over time?",
            "Show monthly trends for Sb in key markets",
            "Biocodex performance in Gut Microbiota Care",
            "Top performing brands in Europe 2024"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}"):
                st.session_state.selected_question = question
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the chatbot.")
        st.info("Get your API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = PharmaDataManager(db_path)
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = ChatGPTQueryProcessor(
            api_key, selected_model, st.session_state.data_manager
        )
    
    # Update query processor if model changes
    if hasattr(st.session_state.query_processor, 'model'):
        if st.session_state.query_processor.model != selected_model:
            st.session_state.query_processor = ChatGPTQueryProcessor(
                api_key, selected_model, st.session_state.data_manager
            )
    
    # Chat interface
    st.subheader("💬 Chat Interface")
    
    # Handle sample question selection
    user_question = ""
    if hasattr(st.session_state, 'selected_question'):
        user_question = st.session_state.selected_question
        del st.session_state.selected_question
    
    # Chat input
    user_input = st.text_area(
        "Ask a question about the pharma sales data:",
        value=user_question,
        height=100,
        placeholder="e.g., What are Sb sales in Ukraine in 2025?"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process user question
    if ask_button and user_input.strip():
        user_question = user_input.strip()
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.spinner("🤔 Analyzing your question..."):
            # Generate SQL query
            sql_query, sql_error = st.session_state.query_processor.generate_sql_query(user_question)
            
            if sql_error:
                st.error(f"Error generating query: {sql_error}")
                return
            
            # Execute SQL query
            results_df, db_error = st.session_state.data_manager.execute_query(sql_query)
            
            if db_error:
                st.error(f"Database error: {db_error}")
                st.code(sql_query, language="sql")
                return
            
            # Generate response
            response = st.session_state.query_processor.generate_response(
                user_question, results_df, sql_query
            )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "sql_query": sql_query,
                "results_df": results_df
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📋 Conversation History")
        
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Expandable sections for SQL and data
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("🔍 View SQL Query"):
                            st.code(message["sql_query"], language="sql")
                    
                    with col2:
                        if message["results_df"] is not None and not message["results_df"].empty:
                            with st.expander("📊 View Raw Data"):
                                st.dataframe(message["results_df"])
                    
                    # Auto-generate simple visualization
                    if message["results_df"] is not None and not message["results_df"].empty:
                        df = message["results_df"]
                        if len(df) > 1 and len(df.columns) >= 2:
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                try:
                                    x_col = df.columns[0]
                                    y_col = numeric_cols[0]
                                    
                                    if len(df) <= 20:  # Simple bar chart for smaller datasets
                                        fig = px.bar(df, x=x_col, y=y_col, 
                                                   title=f"{y_col} by {x_col}")
                                        fig.update_layout(xaxis_tickangle=-45)
                                        st.plotly_chart(fig, use_container_width=True)
                                except:
                                    pass  # Skip visualization if error

if __name__ == "__main__":
    main()
