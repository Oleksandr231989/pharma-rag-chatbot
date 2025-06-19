# =============================================================================
# GOOGLE COLAB RAG CHATBOT FOR PHARMA SALES DATA
# Complete Step 3: RAG System with ChatGPT Integration
# =============================================================================

# STEP 1: Install required packages
!pip install openai plotly ipywidgets

# STEP 2: Upload your database file
from google.colab import files
import sqlite3
import pandas as pd
import openai
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, VBox, HBox

print("📁 Please upload your pharma_sales.db file:")
uploaded = files.upload()

# Get the uploaded database filename
db_filename = list(uploaded.keys())[0]
print(f"✅ Database uploaded: {db_filename}")

# =============================================================================
# RAG CHATBOT SYSTEM CLASSES
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
                "corp_new": "TEXT - PRIORITY: Consolidated corporation name (use this over corporation_original)",
                "brands_new": "TEXT - PRIORITY: Consolidated brand name (use this over brand_original)",
                "sales_euro": "REAL - Sales converted to EUR (PRIMARY SALES METRIC)",
                "sales_units": "INTEGER - Sales units/volume",
                "year": "INTEGER - Year (2023, 2024, 2025)",
                "month": "INTEGER - Month (1-12)"
            },
            "key_entities": {
                "biocodex_corporation": "Use WHERE corp_new LIKE '%biocodex%' OR corp_new LIKE '%BIOCODEX%'",
                "sb_brand": "Use WHERE brands_new = 'Sb' (this is a major Biocodex brand)",
                "market_share_calculation": "Brand Sales ÷ Total Market Sales × 100"
            }
        }
        return schema_info

class ChatGPTQueryProcessor:
    def __init__(self, api_key, data_manager):
        self.client = openai.OpenAI(api_key=api_key)
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
- sales_units: INTEGER - Sales units/volume
- currency: TEXT - Local currency code
- sales_euro: REAL - Sales converted to EUR (PRIMARY SALES METRIC)
- corp_new: TEXT - PRIORITY: Consolidated corporation name (use this over corporation_original)
- brands_new: TEXT - PRIORITY: Consolidated brand name (use this over brand_original)
- year: INTEGER - Year (2023, 2024, 2025)
- month: INTEGER - Month (1-12)

KEY BUSINESS RULES:
1. ALWAYS use 'corp_new' instead of 'corporation_original' 
2. ALWAYS use 'brands_new' instead of 'brand_original'
3. 'sales_euro' is the PRIMARY metric for sales comparisons
4. For Biocodex queries: WHERE corp_new LIKE '%biocodex%' OR corp_new LIKE '%BIOCODEX%'
5. Brand 'Sb' is a major Biocodex brand
6. Market share = (Brand Sales / Total Market Sales) * 100

CRITICAL TREND ANALYSIS REQUIREMENTS:
7. For YEARLY data: ALWAYS include year-over-year growth comparison
8. For MONTHLY data: ALWAYS include month-over-month AND year-over-year comparison
9. For PERIOD data: Compare same periods (e.g., Jan-May 2025 vs Jan-May 2024)
10. NEVER provide absolute figures without context - always show growth rates

MANDATORY QUERY PATTERNS:

FOR YEARLY ANALYSIS:
- Current year sales + Previous year same months comparison
- Calculate growth rate: ((Current - Previous) / Previous) * 100
- Example: 2025 sales vs same months in 2024

FOR MONTHLY ANALYSIS:
- Current month vs previous month (sequential)
- Current month vs same month previous year
- Show both month-over-month and year-over-year trends

FOR PERIOD ANALYSIS (e.g., Q1, Jan-Mar, etc.):
- Current period vs same period previous year
- Calculate percentage growth

TREND ANALYSIS QUERY STRUCTURE:
```sql
WITH current_period AS (
    SELECT SUM(sales_euro) as current_sales, SUM(sales_units) as current_units
    FROM pharma_sales 
    WHERE [current period conditions]
),
previous_period AS (
    SELECT SUM(sales_euro) as previous_sales, SUM(sales_units) as previous_units
    FROM pharma_sales 
    WHERE [previous period conditions]
)
SELECT 
    cp.current_sales,
    cp.current_units,
    pp.previous_sales,
    pp.previous_units,
    ROUND(((cp.current_sales - pp.previous_sales) / pp.previous_sales * 100), 2) as sales_growth_percent,
    ROUND(((cp.current_units - pp.previous_units) / pp.previous_units * 100), 2) as units_growth_percent
FROM current_period cp, previous_period pp
```

SPECIFIC EXAMPLES:

For "Sb sales in Ukraine 2025":
- Query 2025 data AND same months of 2024
- Calculate 2025 vs 2024 growth rate

For "Sb sales in Mexico March 2025":
- Query March 2025 data
- Compare to February 2025 (month-over-month)
- Compare to March 2024 (year-over-year)

For "Sb sales Q1 2025":
- Query Jan-Mar 2025 data
- Compare to Jan-Mar 2024 data
- Calculate Q1 growth rate

CRITICAL COLUMN USAGE:
- Use 'sales_euro' NOT 'Sales euro' or ' Sales euro '
- Use 'sales_units' NOT 'Sales, units' or ' Sales, units '
- Use 'brands_new' NOT 'Brand' or 'Brands new'
- Use 'corp_new' NOT 'Corporation' or 'Corp new'
- Use 'country' NOT 'Country'
- Use 'competitive_market' NOT 'Competitive market'

SAMPLE COUNTRIES: Mexico, Brazil, France, Germany, Belgium, Poland, Ukraine, Russia, Turkey, US
SAMPLE BRANDS: Sb, OTIPAX, SAFORELLE, MUCOGYNE, HYDROMEGA, GALACTOGIL, SYMBIOSYS
SAMPLE MARKETS: Gut Microbiota Care, Ear Drops, Intimate Dryness, Immunity, Urinary
DATA YEARS: 2023, 2024, 2025 (use for trend comparisons)

QUERY GENERATION RULES:
1. Generate ONLY valid SQLite SQL queries with CTE structure for trend analysis
2. ALWAYS include growth rate calculations when possible
3. Use exact column names as specified above
4. Include appropriate GROUP BY and ORDER BY clauses
5. Use LIMIT for top/bottom queries
6. Handle case-insensitive searches with LIKE and wildcards
7. For market share calculations, use subqueries or CTEs
8. MANDATORY: Include previous period comparison for context

Return ONLY the SQL query with trend analysis, no explanations or markdown formatting.
"""
    
    def generate_sql_query(self, user_question):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": f"Convert this question to SQL: {user_question}"}
                ],
                temperature=0.1,
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
            
            # Extract specific entities from the question for focused analysis
            brand_mentioned = None
            country_mentioned = None
            market_mentioned = None
            company_mentioned = None
            
            # Common entities
            brands = ['Sb', 'OTIPAX', 'SAFORELLE', 'MUCOGYNE', 'HYDROMEGA', 'GALACTOGIL', 'SYMBIOSYS', 'MEDIKINET', 'STERIMAR', 'GESTARELLE']
            countries = ['Mexico', 'Brazil', 'France', 'Germany', 'Belgium', 'Poland', 'Ukraine', 'Russia', 'Turkey', 'US', 'India', 'Italy', 'Romania', 'Bulgaria', 'Estonia', 'Finland', 'Greece', 'Hungary', 'Latvia', 'Lithuania', 'Morocco', 'Portugal']
            markets = ['Gut Microbiota Care', 'Ear Drops', 'Intimate Dryness', 'Immunity', 'Urinary', 'Intimate Care', 'Pregnancy', 'Weight Control']
            
            for brand in brands:
                if brand.lower() in user_question.lower():
                    brand_mentioned = brand
                    break
            
            for country in countries:
                if country.lower() in user_question.lower():
                    country_mentioned = country
                    break
                    
            for market in markets:
                if market.lower() in user_question.lower():
                    market_mentioned = market
                    break
            
            if 'biocodex' in user_question.lower():
                company_mentioned = 'Biocodex'
            
            # Determine response type
            if brand_mentioned and country_mentioned:
                response_type = "BRAND_COUNTRY_SPECIFIC"
            elif brand_mentioned:
                response_type = "BRAND_SPECIFIC"
            elif company_mentioned:
                response_type = "COMPANY_SPECIFIC"
            elif market_mentioned or country_mentioned:
                response_type = "MARKET_SPECIFIC"
            else:
                response_type = "GENERIC"
            
            response_prompt = f"""
You are a pharmaceutical data analyst providing FACTUAL analysis based only on database results.

ANALYSIS TYPE: {response_type}
USER QUESTION: {user_question}
ENTITIES IDENTIFIED: Brand: {brand_mentioned}, Country: {country_mentioned}, Market: {market_mentioned}, Company: {company_mentioned}
QUERY RESULTS: {results_text}

CRITICAL TERMINOLOGY:
- sales_euro = SALES VALUE (in EUR) - this is revenue/money
- sales_units = SALES VOLUME (in units/boxes) - this is quantity sold
- sales_local_currency = only mention when user specifically asks about local currency
- Always specify VALUE vs VOLUME in responses

CRITICAL REQUIREMENTS:
1. Use ONLY factual data from the database results
2. BE PRECISE: Specify VALUE (euros) vs VOLUME (units/boxes)
3. DO NOT speculate about marketing campaigns, customer acquisition, or external factors
4. DO NOT mention causes you cannot verify from the data
5. Stick to mathematical comparisons and observable data trends
6. Use ONLY the data provided - no assumptions or industry knowledge

RESPONSE STRUCTURE:
```
📊 {brand_mentioned or 'DATA'} ANALYSIS{' - ' + country_mentioned.upper() if country_mentioned else ''}
═══════════════════════════════════════
Current Performance: [State exact VALUE (€) and VOLUME (units) from database]
Growth Comparison: [Mathematical comparison with previous periods - specify value vs volume]

🔍 KEY INSIGHTS:
• Data Observations: [What the numbers show - be precise about value vs volume]
• Comparative Analysis: [How VALUE and VOLUME metrics compare separately to previous periods]
• Quantitative Trends: [Mathematical trends for both value and volume]
• Measurable Changes: [Specific percentage changes for VALUE and VOLUME separately]
```

PRECISE LANGUAGE EXAMPLES:
- "Sales VALUE of €9.98 million" (not just "sales of €9.98 million")
- "Sales VOLUME of 749,923 units" (not just "749,923 units sold")
- "VALUE increased by X% while VOLUME remained unchanged"
- "VOLUME growth of X% with VALUE growth of Y%"

FACTUAL LANGUAGE TO USE:
- "The data shows..."
- "According to database records..."
- "Mathematical comparison reveals..."
- "VALUE analysis indicates..."
- "VOLUME data shows..."
- "Database records show..."

AVOID COMPLETELY:
- Mixing up value and volume terminology
- Speculation about marketing campaigns
- Assumptions about customer behavior  
- External market factors not in database
- Business strategy recommendations
- Causal explanations without data support

FOCUS ON:
- Exact VALUE (euros) and VOLUME (units) figures
- Mathematical comparisons for both metrics
- Percentage calculations for value vs volume separately
- Data trends visible in numbers
- Factual observations only

Provide only data-driven insights with precise value/volume terminology.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a pharmaceutical data analyst. Provide ONLY factual analysis based on the database results. Do not speculate about marketing campaigns, customer acquisition, or external factors not present in the data. Use temperature=0 for factual responses."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0,  # Factual, no creativity
                max_tokens=1200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

# =============================================================================
# COLAB CHATBOT INTERFACE
# =============================================================================

class ColabChatbot:
    def __init__(self, db_path):
        self.data_manager = PharmaDataManager(db_path)
        self.query_processor = None
        self.chat_history = []
        
        # Test database connection
        self.test_database()
        
    def test_database(self):
        """Test database connection and show stats"""
        try:
            conn = sqlite3.connect(self.data_manager.db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM pharma_sales")
            total_records = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM pharma_sales WHERE corp_new LIKE '%biocodex%'")
            biocodex_records = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT country) FROM pharma_sales")
            countries = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT brands_new) FROM pharma_sales")
            brands = cursor.fetchone()[0]
            
            # Test the specific Ukraine Sb query that was failing
            cursor = conn.execute("""
                SELECT SUM(sales_euro) as total_sales, SUM(sales_units) as total_units, COUNT(*) as records
                FROM pharma_sales 
                WHERE brands_new = 'Sb' AND country = 'Ukraine' AND year = 2025
            """)
            ukraine_test = cursor.fetchone()
            
            conn.close()
            
            print("=" * 60)
            print("🎉 DATABASE SUCCESSFULLY CONNECTED!")
            print("=" * 60)
            print(f"📊 Total Records: {total_records:,}")
            print(f"🏢 Biocodex Records: {biocodex_records:,}")
            print(f"🌍 Countries: {countries}")
            print(f"🏷️ Brands: {brands:,}")
            print("=" * 60)
            print("🔍 TEST QUERY RESULTS:")
            print(f"💰 Sb sales in Ukraine 2025: €{ukraine_test[0]:,.2f}" if ukraine_test[0] else "No data")
            print(f"📦 Units sold: {ukraine_test[1]:,}" if ukraine_test[1] else "0")
            print(f"📋 Records: {ukraine_test[2]}")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            return False
    
    def setup_api_key(self):
        """Setup OpenAI API key"""
        from getpass import getpass
        
        print("🔑 Please enter your OpenAI API key:")
        print("   (Get it from: https://platform.openai.com/api-keys)")
        
        api_key = getpass("API Key: ")
        
        if api_key and api_key.startswith('sk-'):
            self.query_processor = ChatGPTQueryProcessor(api_key, self.data_manager)
            print("✅ API key configured successfully!")
            return True
        else:
            print("❌ Invalid API key format. Should start with 'sk-'")
            return False
    
    def create_sample_questions_interface(self):
        """Create sample questions interface"""
        sample_questions = [
            "What are Sb sales in Mexico in 2024?",
            "What is the market share of Sb in 2023?", 
            "Show me top 5 Biocodex brands by sales with growth trends",
            "Compare Sb sales across countries with year-over-year growth",
            "What's the trend of Sb sales over time?",
            "Analyze Sb sales in March 2025 vs previous months",
            "Which competitive market has highest Biocodex growth?",
            "Show Biocodex Q1 2025 performance vs Q1 2024",
            "Analyze Sb sales in Ukraine 2025 with growth analysis",
            "What are monthly trends for Sb in key markets?"
        ]
        
        print("\n💡 SAMPLE QUESTIONS - Click to try:")
        print("=" * 50)
        
        for i, question in enumerate(sample_questions, 1):
            print(f"{i:2d}. {question}")
        
        print("=" * 50)
        
        # Create selection widget
        question_dropdown = widgets.Dropdown(
            options=[f"{i}: {q}" for i, q in enumerate(sample_questions, 1)],
            description='Sample:',
            style={'description_width': 'initial'}
        )
        
        ask_button = widgets.Button(
            description="Ask This Question",
            button_style='info',
            icon='question'
        )
        
        # Add a test button for the specific Ukraine query
        test_ukraine_button = widgets.Button(
            description="🧪 Test Ukraine Query",
            button_style='warning',
            icon='flask'
        )
        
        def on_ask_sample(b):
            selected = question_dropdown.value
            question = selected.split(': ', 1)[1]
            self.process_question(question)
        
        def on_test_ukraine(b):
            self.process_question("What are the sales of Sb in Ukraine in 2025?")
        
        ask_button.on_click(on_ask_sample)
        test_ukraine_button.on_click(on_test_ukraine)
        
        display(HBox([question_dropdown, ask_button, test_ukraine_button]))
    
    def create_chat_interface(self):
        """Create interactive chat interface"""
        
        # Question input
        question_input = widgets.Textarea(
            placeholder="Ask a question about the pharma sales data...",
            description="Question:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', height='80px')
        )
        
        # Ask button
        ask_button = widgets.Button(
            description="Ask Question",
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='20%')
        )
        
        # Output area
        output = widgets.Output()
        
        def on_ask_button_clicked(b):
            question = question_input.value.strip()
            if question:
                question_input.value = ""  # Clear input
                with output:
                    self.process_question(question)
        
        ask_button.on_click(on_ask_button_clicked)
        
        # Layout
        input_box = HBox([question_input, ask_button])
        chat_interface = VBox([input_box, output])
        
        display(chat_interface)
        
        return output
    
    def process_question(self, question):
        """Process a user question"""
        if not self.query_processor:
            print("❌ Please set up your OpenAI API key first!")
            return
        
        print(f"\n🤔 User Question: {question}")
        print("=" * 60)
        
        # Generate SQL query
        print("🔍 Generating SQL query...")
        sql_query, sql_error = self.query_processor.generate_sql_query(question)
        
        if sql_error:
            print(f"❌ Error generating query: {sql_error}")
            return
        
        print(f"📝 Generated SQL:")
        print("```sql")
        print(sql_query)
        print("```")
        
        # Execute query
        print("\n⚡ Executing query...")
        results_df, db_error = self.data_manager.execute_query(sql_query)
        
        if db_error:
            print(f"❌ Database error: {db_error}")
            return
        
        # Generate response
        print("🤖 Generating response...")
        response = self.query_processor.generate_response(question, results_df, sql_query)
        
        print(f"\n💬 Bot Response:")
        print("=" * 60)
        print(response)
        
        # Show results table if available
        if results_df is not None and not results_df.empty:
            print(f"\n📊 Query Results ({len(results_df)} rows):")
            print("=" * 60)
            display(results_df)
            
            # Auto-generate visualization
            self.create_visualization(results_df, question)
        
        print("\n" + "=" * 60)
    
    def create_visualization(self, df, question):
        """Create automatic visualizations"""
        try:
            if len(df) < 2:
                return
            
            # Simple heuristic for chart type
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return
            
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = numeric_cols[0]
                
                # Determine chart type
                if 'year' in x_col.lower() or 'month' in x_col.lower() or 'time' in question.lower():
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                elif len(df) <= 20:  # Bar chart for smaller datasets
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    fig.update_layout(xaxis_tickangle=-45)
                else:
                    return  # Skip visualization for large datasets
                
                print(f"\n📈 Auto-generated Visualization:")
                fig.show()
                
        except Exception as e:
            print(f"ℹ️ Could not generate visualization: {str(e)}")
    
    def run(self):
        """Run the complete chatbot system"""
        
        print("🚀 PHARMA SALES RAG CHATBOT")
        print("=" * 60)
        print("Welcome to your intelligent pharma sales assistant!")
        print("Ask questions in natural language about Biocodex and competitor data.")
        print("=" * 60)
        
        # Setup API key
        if not self.setup_api_key():
            return
        
        # Show sample questions
        self.create_sample_questions_interface()
        
        # Create chat interface
        print("\n💬 CHAT INTERFACE:")
        print("Type your questions below or use the sample questions above.")
        print("=" * 60)
        
        self.create_chat_interface()

# =============================================================================
# RUN THE CHATBOT
# =============================================================================

# Initialize and run the chatbot
chatbot = ColabChatbot(db_filename)
chatbot.run()
