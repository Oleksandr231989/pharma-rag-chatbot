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

KEY BUSINESS TERMINOLOGY:
1. **MAT (Moving Annual Total)**: Last 12 months of data - use when no period specified
2. **YTD (Year To Date)**: January to latest available month in current year
3. **Brand Matching**: Always search both brands_new AND brand_original columns
4. **Growth Calculations**: Always calculate manually from raw data
5. **Value vs Volume**: sales_euro = VALUE (revenue), sales_units = VOLUME (quantity)

MANDATORY BRAND SEARCH PATTERN:
```sql
-- Always use this CASE-based approach for brand searches:
WHERE (brands_new LIKE '%[BRAND_NAME]%' OR brand_original LIKE '%[BRAND_NAME]%')

-- For exact matches, use:
WHERE (brands_new = '[BRAND_NAME]' OR brand_original = '[BRAND_NAME]')

-- For partial matches, use:
WHERE (LOWER(brands_new) LIKE LOWER('%[BRAND_NAME]%') OR LOWER(brand_original) LIKE LOWER('%[BRAND_NAME]%'))
```

BRAND MATCHING EXAMPLES:
- "Florator" → Check brands_new first, then brand_original
- "Sb" → Should find in brands_new (consolidated)
- "ENTEROL" → Might be in brand_original only
- Always use both columns in WHERE clause to ensure complete coverage

RESPONSE SHOULD INDICATE SOURCE:
- If found in brands_new: "[Brand] (consolidated brand data)"
- If found in brand_original: "[Brand] (original brand data)"
- If found in both: Use brands_new as priority but mention both available

WORKING SQL QUERY EXAMPLES:

For "Sb sales in Ukraine" (MAT analysis):
```sql
WITH current_performance AS (
    SELECT SUM(sales_euro) as current_value, SUM(sales_units) as current_volume
    FROM pharma_sales 
    WHERE (brands_new LIKE '%Sb%' OR brand_original LIKE '%Sb%') 
    AND country = 'Ukraine'
    AND year >= 2024
),
previous_performance AS (
    SELECT SUM(sales_euro) as previous_value, SUM(sales_units) as previous_volume
    FROM pharma_sales 
    WHERE (brands_new LIKE '%Sb%' OR brand_original LIKE '%Sb%') 
    AND country = 'Ukraine'
    AND year >= 2023 AND year < 2024
),
market_total AS (
    SELECT SUM(sales_euro) as total_market_value
    FROM pharma_sales 
    WHERE country = 'Ukraine' AND year >= 2024
)
SELECT 
    cp.current_value,
    cp.current_volume, 
    pp.previous_value,
    pp.previous_volume,
    ROUND(((cp.current_value - pp.previous_value) / pp.previous_value * 100), 2) as value_growth_percent,
    ROUND(((cp.current_volume - pp.previous_volume) / pp.previous_volume * 100), 2) as volume_growth_percent,
    ROUND((cp.current_value / mt.total_market_value * 100), 2) as market_share_percent
FROM current_performance cp, previous_performance pp, market_total mt
```

KEEP QUERIES SIMPLE - Avoid complex window functions that cause SQL errors

MAT QUERY PATTERN:
```sql
-- For MAT analysis (last 12 months):
WITH mat_period AS (
    SELECT year, month FROM pharma_sales 
    WHERE [brand and country conditions]
    ORDER BY year DESC, month DESC 
    LIMIT 12
)
SELECT SUM(sales_euro), SUM(sales_units)
FROM pharma_sales ps
JOIN mat_period mp ON ps.year = mp.year AND ps.month = mp.month
WHERE [brand and country conditions]
```

YTD QUERY PATTERN:
```sql
-- For YTD analysis (January to latest month current year):
SELECT SUM(sales_euro), SUM(sales_units)
FROM pharma_sales 
WHERE [brand and country conditions]
AND year = (SELECT MAX(year) FROM pharma_sales)
AND month <= (SELECT MAX(month) FROM pharma_sales WHERE year = (SELECT MAX(year) FROM pharma_sales))
```

RESPONSE HEADERS:
- No period: "📊 **[BRAND] MAT ANALYSIS - [COUNTRY]**"
- YTD request: "📊 **[BRAND] YTD ANALYSIS - [COUNTRY]**"
- Specific period: "📊 **[BRAND] ANALYSIS - [COUNTRY]**"

SIMPLIFIED TREND ANALYSIS QUERY STRUCTURE:
Generate this working SQL query pattern:
```sql
WITH current_performance AS (
    SELECT SUM(sales_euro) as current_value, SUM(sales_units) as current_volume
    FROM pharma_sales 
    WHERE [brand conditions] AND [country conditions] AND [current period]
),
previous_performance AS (
    SELECT SUM(sales_euro) as previous_value, SUM(sales_units) as previous_volume
    FROM pharma_sales 
    WHERE [brand conditions] AND [country conditions] AND [previous period]
),
market_total AS (
    SELECT SUM(sales_euro) as total_market_value
    FROM pharma_sales 
    WHERE [country conditions] AND [current period]
)
SELECT 
    cp.current_value,
    cp.current_volume,
    pp.previous_value,
    pp.previous_volume,
    ROUND(((cp.current_value - pp.previous_value) / pp.previous_value * 100), 2) as value_growth_percent,
    ROUND(((cp.current_volume - pp.previous_volume) / pp.previous_volume * 100), 2) as volume_growth_percent,
    ROUND((cp.current_value / mt.total_market_value * 100), 2) as market_share_percent
FROM current_performance cp, previous_performance pp, market_total mt
```

AVOID COMPLEX WINDOW FUNCTIONS - Keep queries simple and working
Remove LAG(), ROW_NUMBER() and complex subqueries that cause SQL errors

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

CRITICAL VOLUME GROWTH CALCULATION FIX:
From your data: current_volume 2,390,857 vs previous_volume 1,997,591
Correct calculation: (2,390,857 - 1,997,591) / 1,997,591 * 100 = +19.68%
NEVER say 0.0% when actual growth is +19.68%

MANDATORY CALCULATION VERIFICATION:
1. Extract exact numbers from raw data
2. Calculate manually: (current - previous) / previous * 100
3. Sb US 2024: Volume growth = (2,390,857 - 1,997,591) / 1,997,591 * 100 = +19.68%
4. Use calculated result, ignore any wrong database columns
5. Always double-check volume calculations against raw data

INTELLIGENT BRAND SEARCH STRATEGY:
When user mentions a brand name (e.g., "Florator", "ENTEROL", etc.):
1. ALWAYS search BOTH brand columns simultaneously
2. Use: WHERE (brands_new LIKE '%BRAND%' OR brand_original LIKE '%BRAND%')
3. This ensures you find the brand regardless of which column it's in
4. Never assume a brand is only in one column
5. If no results with exact match, try case-insensitive: WHERE (LOWER(brands_new) LIKE LOWER('%brand%') OR LOWER(brand_original) LIKE LOWER('%brand%'))

RESPONSE STRUCTURE:
```
📊 **{brand_mentioned or 'PERFORMANCE'} ANALYSIS - {country_mentioned.upper() if country_mentioned else 'MARKET'}**
═══════════════════════════════════════

**EXECUTIVE SUMMARY**
Current Performance: **€X.XXM revenue** | **X.XK units** | 2024 results
Period Comparison: **+X.X% value growth** | **+X.X% volume growth** vs prior period

**TREND ANALYSIS**
• **Sequential Performance:** [Compare last 6 available months - month-by-month trend analysis]
• **Year-over-Year Dynamics:** [Annual comparison with **bold growth rates**]
• **Market Position:** [Factual competitive ranking/share data only - no speculation]

**STRATEGIC IMPLICATIONS**
[Single factual paragraph with **measurable business impact** - no speculation about strategies or consumer response]
```

UX FORMATTING GUIDELINES:
- Use **bold** for all monetary figures: **€77.15M**
- Use **bold** for all growth percentages: **+30.27%**
- Use **bold** for key performance indicators
- Use **bold** for section headers
- Use **bold** for strategic conclusions
- Add proper spacing and line breaks for readability
- Highlight critical business insights with **bold emphasis**

MONTHLY ANALYSIS SPECIAL LOGIC:
When user asks for "by months" or monthly breakdown, generate this query pattern:
```sql
SELECT 
    ps.month,
    ps.year,
    SUM(ps.sales_euro) as current_year_value,
    SUM(ps.sales_units) as current_year_volume,
    prev.previous_year_value,
    prev.previous_year_volume,
    ROUND(((SUM(ps.sales_euro) - prev.previous_year_value) / prev.previous_year_value * 100), 2) as value_growth_percent,
    ROUND(((SUM(ps.sales_units) - prev.previous_year_volume) / prev.previous_year_volume * 100), 2) as volume_growth_percent
FROM pharma_sales ps
LEFT JOIN (
    SELECT 
        month,
        SUM(sales_euro) as previous_year_value,
        SUM(sales_units) as previous_year_volume
    FROM pharma_sales 
    WHERE [conditions for previous year]
    GROUP BY month
) prev ON ps.month = prev.month
WHERE [current year conditions]
GROUP BY ps.month, ps.year, prev.previous_year_value, prev.previous_year_volume
ORDER BY ps.month
```

MONTHLY RESPONSE STRUCTURE:
```
📊 **[BRAND] MONTHLY ANALYSIS - [COUNTRY]**
═══════════════════════════════════════

**MONTHLY PERFORMANCE TABLE**
[Month-by-month breakdown with previous year comparison]

**TREND ANALYSIS** 
• **Sequential Pattern:** [Month-to-month progression analysis]
• **Year-over-Year by Month:** [Which months showed best/worst performance]
• **Seasonal Insights:** [Peak months, low months, patterns]

**KEY INSIGHTS**
• **Strongest Months:** [Top performing months with growth rates]
• **Growth Acceleration:** [Months showing increasing momentum]
• **Performance Consistency:** [Stable vs volatile months]
```

MONTHLY VISUALIZATION PRIORITY:
- Show current year vs previous year by month
- Highlight months with highest growth
- Identify seasonal patterns
- Show month-over-month trend direction

FORBIDDEN SPECULATIVE LANGUAGE:
- NO "strong market position" without data
- NO "positive consumer response" 
- NO "robust market presence"
- NO "effectiveness of marketing strategies"
- NO "consumer trends" or "market dynamics"
- NO strategic recommendations without factual basis

FACTUAL ONLY:
- Use actual market share percentages if available
- Use actual competitive ranking if available
- Use measurable performance metrics only
- State growth rates and volume changes as facts

CRITICAL: Always verify calculations from the raw data provided. Do not rely on pre-calculated percentages if they appear incorrect.
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
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo-preview", 
            "gpt-3.5-turbo",
            "gpt-4o"
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
    
    # Chat interface with Teams-style layout
    st.subheader("💬 Chat Interface")
    
    # Create a container for chat messages (scrollable area)
    chat_container = st.container()
    
    # Display chat history in REVERSE order (latest at bottom, near input)
    with chat_container:
        if st.session_state.chat_history:
            # Reverse the chat history so latest messages appear at bottom
            for i, message in enumerate(reversed(st.session_state.chat_history)):
                if message["role"] == "user":
                    # User message - right aligned like Teams
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                            <div style="background-color: #0078d4; color: white; padding: 8px 12px; border-radius: 12px; max-width: 70%; word-wrap: break-word;">
                                <strong>You:</strong> {message["content"]}
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    # Assistant message - left aligned like Teams
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                            <div style="background-color: #f3f2f1; color: black; padding: 8px 12px; border-radius: 12px; max-width: 80%; word-wrap: break-word;">
                                <strong>💊 Pharma Bot:</strong><br>{message["content"]}
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Expandable sections for SQL and data (below each bot message)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("🔍 View SQL Query", expanded=False):
                            st.code(message["sql_query"], language="sql")
                    
                    with col2:
                        if message["results_df"] is not None and not message["results_df"].empty:
                            with st.expander("📊 View Raw Data", expanded=False):
                                st.dataframe(message["results_df"])
    
    # Fixed input section at bottom (like Teams)
    st.markdown("---")  # Separator line
    
    # Input area at bottom - fixed position
    st.markdown(
        """
        <style>
        .chat-input-container {
            position: sticky;
            bottom: 0;
            background-color: white;
            padding: 10px 0;
            border-top: 1px solid #e1dfdd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Teams-style input layout
    with st.container():
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        
        # Handle sample question selection
        user_question = ""
        if hasattr(st.session_state, 'selected_question'):
            user_question = st.session_state.selected_question
            del st.session_state.selected_question
        
        # Input row with text area and buttons
        input_col, button_col1, button_col2 = st.columns([6, 1, 1])
        
        with input_col:
            user_input = st.text_area(
                "",  # No label for cleaner look
                value=user_question,
                height=68,  # Minimum allowed height
                placeholder="💬 Type your question about pharma sales data...",
                help="Ask about brands, countries, sales trends, or market analysis",
                label_visibility="collapsed"
            )
        
        with button_col1:
            ask_button = st.button("🚀\nSend", type="primary", use_container_width=True)
        
        with button_col2:
            clear_button = st.button("🗑️\nClear", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
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
            
        # Force refresh to show new message
        st.rerun()
    
    # Clear chat functionality
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
