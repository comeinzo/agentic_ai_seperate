# AI seperate
import json
import os
from datetime import timezone
from datetime import timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
import traceback
from config import DB_NAME, USER_NAME, PASSWORD, HOST, PORT
# ==============================
# Third-Party Imports
# ==============================
import pandas as pd
# import numpy as np  # Add this if not already imported at the top
import psycopg2

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
# from openai import OpenAI
import google.generativeai as genai



load_dotenv()
load_dotenv(override=True)
print(f"DEBUG: HF Key loaded: {bool(os.getenv('HUGGINGFACE_API_KEY'))}")

# ============================================================================
# GOOGLE GEMINI CONFIGURATION & ROBUST MODEL SELECTOR
# ============================================================================

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


ACTIVE_MODEL_NAME = "gemini-2.5-flash-lite"


# ============================================================================



def list_available_tables():
    """List all available tables in the database."""
    try:
        conn = psycopg2.connect(
            host=HOST, port=PORT, database=DB_NAME, user=USER_NAME, password=PASSWORD
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [table[0] for table in tables]
    except Exception as e:
        print(f"Error listing tables: {e}")
        return []


def get_table_schema(table_name):
    """Fetch the schema for a specific table."""
    try:
        conn = psycopg2.connect(
            host=HOST, port=PORT, database=DB_NAME, user=USER_NAME, password=PASSWORD
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = cursor.fetchall()
        
        if not columns:
            cursor.close()
            conn.close()
            return None
        
        # Format schema
        schema_text = f"Table: {table_name}\nColumns:\n"
        for col_name, data_type, is_nullable in columns:
            nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
            schema_text += f"  - {col_name} ({data_type}) {nullable}\n"
        
        cursor.close()
        conn.close()
        
        return schema_text
    except Exception as e:
        print(f"Error getting schema: {e}")
        return None



def execute_sql(sql_query):
    """Execute SQL query and return results."""
    try:
        conn = psycopg2.connect(
            host=HOST, port=PORT, database=DB_NAME, user=USER_NAME, password=PASSWORD
        )
        
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        return df, None
    except Exception as e:
        return None, str(e)




def generate_comprehensive_data_profile(df, table_name):
    """Generate a comprehensive profile of the data."""
    profile = {
        'table_name': table_name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': {},
        'relationships': []
    }
    
    for col in df.columns:
        col_info = {
            'name': col,
            'type': str(df[col].dtype),
            'unique_count': int(df[col].nunique()),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': float(df[col].isnull().sum() / len(df) * 100)
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['data_type'] = 'numeric'
            col_info['min'] = float(df[col].min()) if pd.notna(df[col].min()) else None
            col_info['max'] = float(df[col].max()) if pd.notna(df[col].max()) else None
            col_info['mean'] = float(df[col].mean()) if pd.notna(df[col].mean()) else None
            col_info['median'] = float(df[col].median()) if pd.notna(df[col].median()) else None
            col_info['std'] = float(df[col].std()) if pd.notna(df[col].std()) else None
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
            col_info['data_type'] = 'date'
        else:
            col_info['data_type'] = 'categorical'
            top_values = df[col].value_counts().head(5)
            col_info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
        
        profile['columns'][col] = col_info
    
    return profile


def generate_default_kpi_structure(data_profile):
    """Generate a default KPI structure if AI fails."""
    primary_kpis = []
    charts = []
    
    # Find numeric columns for KPIs
    numeric_cols = [col for col, info in data_profile['columns'].items() 
                    if info['data_type'] == 'numeric']
    
    categorical_cols = [col for col, info in data_profile['columns'].items() 
                        if info['data_type'] == 'categorical']
    
    # Create default KPIs from numeric columns
    for col in numeric_cols[:4]:
        primary_kpis.append({
            'name': col.replace('_', ' ').title(),
            'description': f'Total {col}',
            'calculation': col,
            'format': 'currency' if any(word in col.lower() for word in ['price', 'amount', 'cost', 'revenue']) else 'number',
            'icon': 'TrendingUp',
            'category': 'operational'
        })
    
    # Create default charts
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        charts.append({
            'type': 'bar',
            'title': f'{categorical_cols[0]} Analysis',
            'x_axis': categorical_cols[0],
            'y_axis': numeric_cols[0],
            'aggregation': 'sum',
            'description': f'Distribution by {categorical_cols[0]}',
            'limit': 10
        })
    
    return {
        'primary_kpis': primary_kpis,
        'charts': charts,
        'trends': [],
        'segments': []
    }


def calculate_kpis(df, kpi_structure):
    """Calculate actual KPI values based on the AI-generated structure."""
    kpi_values = []
    
    for kpi in kpi_structure.get('primary_kpis', []):
        try:
            calculation = kpi['calculation']
            # Default to sum if not specified, but check for specific aggregation instructions
            aggregation = kpi.get('aggregation', 'sum').lower()
            
            # Check if it's a simple column reference (or if calculation IS the column name)
            col_name = calculation
            if col_name in df.columns:
                
                # Pre-processing for numeric operations
                if aggregation in ['sum', 'avg', 'average', 'mean', 'min', 'minimum', 'max', 'maximum']:
                    numeric_series = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                
                if aggregation == 'sum':
                    value = float(numeric_series.sum())
                elif aggregation in ['avg', 'average', 'mean']:
                    value = float(numeric_series.mean())
                elif aggregation in ['min', 'minimum']:
                    value = float(numeric_series.min())
                elif aggregation in ['max', 'maximum']:
                    value = float(numeric_series.max())
                elif aggregation == 'count':
                    value = float(df[col_name].count())
                elif aggregation in ['distinct count', 'distinct_count', 'nunique']:
                    value = float(df[col_name].nunique())
                else: 
                     # Fallback logic based on format/name if aggregation is unknown or implicit
                    if kpi.get('format') == 'number' and 'id' in col_name.lower():
                         value = float(df[col_name].nunique())
                    elif kpi.get('format') in ['number', 'currency', 'percentage']:
                         numeric_series = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                         if kpi.get('format') == 'percentage':
                             value = float(numeric_series.mean() * 100)
                         else:
                             value = float(numeric_series.sum())
                    else:
                        # numeric default
                        numeric_series = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                        value = float(numeric_series.sum())

                # Calculate change (mock for now)
                change = 0.0
                
                kpi_values.append({
                    'name': kpi['name'],
                    'value': value,
                    'format': kpi['format'],
                    'icon': kpi.get('icon', 'TrendingUp'),
                    'category': kpi.get('category', 'operational'),
                    'description': kpi['description'],
                    'change': change,
                    'change_direction': 'up' if change > 0 else 'down' if change < 0 else 'neutral'
                })
            else:
                # Handle generic counts if column not found but "count" is requested
                if 'count' in calculation.lower() or aggregation == 'count':
                    value = len(df)
                else:
                    value = 0
                
                kpi_values.append({
                    'name': kpi['name'],
                    'value': value,
                    'format': kpi['format'],
                    'icon': kpi.get('icon', 'TrendingUp'),
                    'category': kpi.get('category', 'operational'),
                    'description': kpi['description'],
                    'change': 0.0,
                    'change_direction': 'neutral'
                })
                
        except Exception as e:
            print(f"Error calculating KPI {kpi.get('name', 'Unknown')}: {e}")
            continue
    
    return kpi_values


def generate_chart_data(df, kpi_structure):
    """Generate data for all charts based on KPI structure."""
    chart_data_list = []
    
    for chart_config in kpi_structure.get('charts', []):
        try:
            x_axis = chart_config['x_axis']
            y_axis = chart_config['y_axis']
            aggregation = chart_config.get('aggregation', 'sum').lower()
            limit = chart_config.get('limit', 10)
            
            if x_axis not in df.columns or y_axis not in df.columns:
                continue
            
            # --- Ensure Y-Axis is Numeric for Aggregation ---
            if aggregation in ['sum', 'avg', 'average', 'mean', 'min', 'minimum', 'max', 'maximum']:
                 df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce').fillna(0)

            # Group and aggregate data
            if aggregation == 'sum':
                grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            elif aggregation == 'count':
                grouped = df.groupby(x_axis)[y_axis].count().reset_index()
            elif aggregation in ['distinct count', 'distinct_count', 'nunique']:
                 grouped = df.groupby(x_axis)[y_axis].nunique().reset_index()
            elif aggregation in ['avg', 'average', 'mean']:
                grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
            elif aggregation in ['min', 'minimum']:
                grouped = df.groupby(x_axis)[y_axis].min().reset_index()
            elif aggregation in ['max', 'maximum']:
                grouped = df.groupby(x_axis)[y_axis].max().reset_index()
            else:
                grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            
            # Sort and limit
            if limit is not None:
                 grouped = grouped.nlargest(int(limit), y_axis) 
            else:
                 grouped = grouped.nlargest(10, y_axis)
            
            # Convert to list of dicts
            chart_data = grouped.to_dict('records')
            
            chart_data_list.append({
                'type': chart_config['type'],
                'title': chart_config['title'],
                'description': chart_config.get('description', ''),
                'x_axis': x_axis,
                'y_axis': y_axis,
                'data': chart_data
            })
            
        except Exception as e:
            print(f"Error generating chart data: {e}")
            continue
    
    return chart_data_list


def generate_kpi_structure_with_ai(table_name, schema, data_profile, sample_data):
    """Use AI to determine the most relevant KPIs (with HF Fallback)."""
    
    prompt = f"""You are a data analytics expert. Analyze this database table and determine the most relevant KPIs.

Table Name: {table_name}
Schema: {schema}
Data Profile: {json.dumps(data_profile, indent=2)}
Sample Data: {sample_data}

Respond ONLY with valid JSON in this exact structure:
{{
  "primary_kpis": [
    {{
      "name": "KPI Name",
      "description": "What it measures",
      "calculation": "column_name or aggregation",
      "format": "number|currency|percentage|date",
      "icon": "icon_name",
      "category": "category",
      "aggregation": "sum|avg|max|min|count|distinct count"
    }}
  ],
  "charts": [
    {{
      "type": "bar|line|pie|area",
      "title": "Chart Title",
      "x_axis": "column_name",
      "y_axis": "column_name",
      "aggregation": "sum|count|avg|min|max|distinct count",
      "description": "Insight provided",
      "limit": 10
    }}
  ],
  "trends": [],
  "segments": []
}}"""
    
    try:
        kpi_text = generate_content_with_fallback(prompt, genai.GenerationConfig(temperature=0.3), json_mode=True)
        kpi_text = kpi_text.replace('```json', '').replace('```', '').strip()
        return json.loads(kpi_text)
        
    except Exception as e:
        print(f"Error generating KPI structure: {e}")
        return generate_default_kpi_structure(data_profile)




def generate_kpi_insights_with_ai(table_name, kpi_values, chart_data):
    """Generate business insights based on KPI values (with HF Fallback)."""
    
    prompt = f"""You are a business intelligence expert. Analyze these KPI metrics and provide actionable insights.

Table: {table_name}
KPI Values: {json.dumps(kpi_values, indent=2)}
Chart Data Summary: {json.dumps([{'title': c['title'], 'type': c['type']} for c in chart_data], indent=2)}

Respond with JSON:
{{
  "observations": ["observation 1", ...],
  "action_items": [
    {{"title": "Action", "description": "Details", "priority": "high|medium|low"}}
  ],
  "opportunities": ["opportunity 1", ...],
  "risks": ["risk 1", ...]
}}"""
    
    try:
        insights_text = generate_content_with_fallback(prompt, genai.GenerationConfig(temperature=0.7), json_mode=True)
        insights_text = insights_text.replace('```json', '').replace('```', '').strip()
        return json.loads(insights_text)
        
    except Exception as e:
        print(f"Error generating insights: {e}")
        return {
            "observations": [], "action_items": [], "opportunities": [], "risks": []
        }




import requests
import time

# Hugging Face Configuration (OpenAI Compatible)
# Hugging Face Configuration (OpenAI Compatible)
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

def query_huggingface(prompt, max_new_tokens=1024, system_prompt=None):
    """Fallback generation using Hugging Face (OpenAI Compatible Endpoint)."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("⚠️ HUGGINGFACE_API_KEY missing. Cannot failover.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # OpenAI Format
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct", 
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.3 # Lower temp for more deterministic/focused output
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        # DEBUG: Print status and raw text if not 200
        if response.status_code != 200:
            print(f"⚠️ HF API Status: {response.status_code}")
            print(f"⚠️ HF API Response: {response.text}")
        
        try:
            output = response.json()
        except ValueError:
            print(f"❌ HF JSON Decode Error. Raw text: {response.text}")
            return None
            
        # Parse OpenAI-style response
        if "choices" in output and len(output["choices"]) > 0:
            content = output["choices"][0]["message"]["content"]
            return content.strip()
            
        # Fallback for error messages in JSON
        if "error" in output:
             print(f"HF Error: {output['error']}")
        
        print(f"Unexpected HF response format: {output}")
        return None
        
    except Exception as e:
        print(f"Error querying Hugging Face: {e}")
        return None


def generate_content_with_fallback(prompt, config=None, json_mode=False):
    """
    Wrapper to try Gemini first, then fallback to Hugging Face on error.
    """
    # 0. Force Hugging Face (Testing/User Preference)
    FORCE_HUGGINGFACE_ONLY = False  # <--- CHANGED BACK TO FALSE (Gemini First)
    
    if FORCE_HUGGINGFACE_ONLY:
        print("🟢 FORCE MODE: Using Hugging Face Only...")
        hf_prompt = prompt
        sys_prompt = "You are a helpful AI assistant."
        
        if json_mode:
             hf_prompt += "\n\nProvide the response strictly in valid JSON format. Do not add markdown backticks."
             sys_prompt = "You are an expert data analyst. Your task is to analyze data and output valid, strict JSON ONLY. No preamble, no explanation."
             
        return query_huggingface(hf_prompt, system_prompt=sys_prompt)

    # 1. Try Gemini
    try:
        print("🔵 Attempting to generate with Google Gemini...")
        model = genai.GenerativeModel(ACTIVE_MODEL_NAME)
        # Adjust config for JSON mode if requested (Gemini specific)
        generation_config = config if config else genai.GenerationConfig(temperature=0.7)
        if json_mode:
            generation_config.response_mime_type = "application/json"
            
        response = model.generate_content(prompt, generation_config=generation_config)
        print("✅ Google Gemini Success")
        return response.text.strip() # Success
        
    except Exception as e:
        error_str = str(e).lower()
        # Check for Rate Limit (429) or Quota issues (ResourceExhausted)
        if "429" in error_str or "quota" in error_str or "resource" in error_str:
            print(f"⚠️ Gemini Quota Exceeded. Failing over to Hugging Face... ({e})")
            print("🟢 Attempting to generate with Hugging Face...")
            
            # 2. Fallback to Hugging Face
            hf_prompt = prompt
            sys_prompt = "You are a helpful AI assistant."
            
            if json_mode:
                 hf_prompt += "\n\nProvide the response strictly in valid JSON format. Do not add markdown backticks."
                 sys_prompt = "You are an expert data analyst. Your task is to analyze data and output valid, strict JSON ONLY. No preamble, no explanation."
            
            hf_result = query_huggingface(hf_prompt, system_prompt=sys_prompt)
            if hf_result:
                print("✅ Hugging Face Fallback Success")
                return hf_result
            else:
                print("❌ Hugging Face fallback failed.")
                raise e # Re-raise original if fallback fails
        else:
            print(f"❌ Gemini Error (Not Quota): {e}")
            raise e


def generate_sql_with_gemini(question, schema, table_name):
    """Generate SQL using Google Gemini API (with HF Fallback)."""
    prompt = f"""You are an expert SQL agent who generates optimized PostgreSQL queries.
    
Database Schema:
{schema}

Question: {question}

Instructions:
1. Analyze the question carefully
2. Identify the relevant columns from the schema
3. Generate an efficient PostgreSQL query
4. The table name is '{table_name}'
5. Return ONLY the SQL query without any explanation or markdown formatting.

SQL Query:"""
    
    try:
        # Use fallback wrapper
        response_text = generate_content_with_fallback(prompt, genai.GenerationConfig(temperature=0.6))
        
        if not response_text:
             return None
             
        sql = response_text
        # Clean up the SQL if AI adds markdown
        sql = sql.replace('```sql', '').replace('```', '').strip()
        return sql
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return None





def generate_natural_answer(question, sql_query, results_df):
    """Generate a natural language answer (with HF Fallback)."""
    
    if results_df is None or len(results_df) == 0:
        return "No results found for your query."
    
    row_count = len(results_df)
    preview = results_df.head(3).to_string(index=False) if row_count > 0 else "No data"
    
    prompt = f"""You are a helpful data analyst assistant.
    
User's Question: {question}
SQL Query Executed: {sql_query}
Number of Rows: {row_count}
Data Preview:
{preview}

Provide a brief, friendly answer that:
1. Directly answers the question
2. Mentions key findings
3. Is conversational
4. Avoids technical jargon

Answer:"""
    
    try:
        return generate_content_with_fallback(prompt, genai.GenerationConfig(temperature=0.7))
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I found {row_count} result{'s' if row_count != 1 else ''} for your query."

def generate_business_insights(table_name, schema, data_summary, focus_area, sample_data):
    """Generate AI-powered business insights (with HF Fallback)."""
    
    prompt = f"""You are an expert business analyst and data scientist. Analyze the following database table and provide actionable insights.

                Table Name: {table_name}

                Schema:
                {schema}

                Data Summary:
                - Total Records: {data_summary['total_rows']}
                - Numeric Columns Stats: {

                            json.dumps(data_summary.get('numeric_stats',{}), indent=2)
                }
                - Categorical Columns Stats: {
                            json.dumps(data_summary.get('categorical_stats',{}), indent=2)
                }

                Sample Data Preview:
                {sample_data}

                Focus Area: {focus_area}

                Please provide:
                1. **Key Findings**
                2. **Actionable Recommendations**
                3. **Data Quality Observations**
                4. **Suggested Metrics to Track**
                5. **Quick Wins**

                Format your response as JSON with the following structure:
                {{
                "key_findings": ["finding 1", ...],
                "recommendations": [
                    {{"title": "Title", "description": "Desc", "priority": "high", "impact": "Impact"}}
                ],
                "data_quality": ["obs 1", ...],
                "suggested_metrics": ["metric 1", ...],
                "quick_wins": [
                    {{"action": "Action", "expected_outcome": "Outcome"}}
                ]
                }}

            Provide ONLY the JSON response, no additional text or markdown."""
    
    try:
        insights_text = generate_content_with_fallback(prompt, genai.GenerationConfig(temperature=0.7), json_mode=True)
        
        if not insights_text:
             return {
                "key_findings": ["Error generating insights (AI returned empty)"],
                "recommendations": [], "data_quality": [], "suggested_metrics": [], "quick_wins": []
            }
            
         # Clean up markdown if explicitly added by HF (Gemini JSON mode usually handles it, but HF might not)
        insights_text = insights_text.replace('```json', '').replace('```', '').strip()
        return json.loads(insights_text)
        
    except Exception as e:
        print(f"Error generating insights: {e}")
        return {
            "key_findings": ["Error generating insights"],
            "recommendations": [],
            "data_quality": [],
            "suggested_metrics": [],
            "quick_wins": []
        }


def generate_contextual_chat_response(user_message, table_name, schema, context, sample_data):
    """Generate contextual AI responses (with HF Fallback)."""
    
    context_str = json.dumps(context, indent=2) if context else "No additional context"
    
    prompt = f"""You are an AI business advisor integrated into a data analytics dashboard. 

Current Table: {table_name}

Schema:
{schema}

Sample Data:
{sample_data}

Dashboard Context:
{context_str}

User Message: {user_message}

Provide a helpful, actionable response that:
1. Directly addresses the user's question or concern
2. Provides specific, data-driven recommendations when appropriate
3. Offers tips to increase profit, market share, or operational efficiency
4. Is conversational and encouraging

Keep your response concise (2-4 paragraphs)."""
    
    try:
        return generate_content_with_fallback(prompt, genai.GenerationConfig(temperature=0.8))
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

app = Flask(
    __name__,
    static_url_path='/static',
    static_folder='uploaded_logos'
)
CORS(app, resources={r"/*": {"origins": "*"}})
# Enable WebSockets
# socketio.init_app(app, cors_allowed_origins="*")

@app.route('/api/dashboard/kpi/insights/<table_name>', methods=['POST'])
def get_kpi_insights(table_name):
    """Generate AI insights specifically for KPI dashboard."""
    try:
        data = request.json
        kpi_values = data.get('kpi_values', [])
        chart_data = data.get('chart_data', [])
        
        # Generate contextual insights using GEMINI
        insights = generate_kpi_insights_with_ai(table_name, kpi_values, chart_data)
        
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        print(f"Error generating KPI insights: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500





@app.route('/api/tables', methods=['GET'])
def get_tables_for_ai():
    """Get list of available tables."""
    try:
        tables = list_available_tables()
        return jsonify({
            'success': True,
            'tables': tables,
            'count': len(tables)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dashboard/kpi/<table_name>', methods=['GET'])
def generate_kpi_dashboard(table_name):
    """Generate intelligent KPI dashboard based on AI analysis of table structure."""
    try:
        # Get table schema
        schema = get_table_schema(table_name)
        if not schema:
            return jsonify({
                'success': False,
                'error': f'Table {table_name} not found'
            }), 404
        
        # Get all data from table
        full_data_query = f"SELECT * FROM {table_name}"
        full_df, error = execute_sql(full_data_query)
        
        if error or full_df is None or len(full_df) == 0:
            return jsonify({
                'success': False,
                'error': 'Unable to fetch data for KPI generation'
            }), 500
        
        # Convert datetime columns to string for processing
        for col in full_df.columns:
            if pd.api.types.is_datetime64_any_dtype(full_df[col]):
                full_df[col] = full_df[col].astype(str)
        
        # Generate comprehensive data profile
        data_profile = generate_comprehensive_data_profile(full_df, table_name)
        
        # Use AI to determine appropriate KPIs using GEMINI
        kpi_structure = generate_kpi_structure_with_ai(
            table_name=table_name,
            schema=schema,
            data_profile=data_profile,
            sample_data=full_df.head(50).to_string()
        )
        
        # Calculate actual KPI values based on AI recommendations
        kpi_values = calculate_kpis(full_df, kpi_structure)
        
        # Generate visualizations data
        chart_data = generate_chart_data(full_df, kpi_structure)
        
        return jsonify({
            'success': True,
            'table_name': table_name,
            'total_records': len(full_df),
            'kpi_structure': kpi_structure,
            'kpi_values': kpi_values,
            'chart_data': chart_data,
            'data_profile': data_profile
        })
        
    except Exception as e:
        print(f"Error generating KPI dashboard: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=True)
