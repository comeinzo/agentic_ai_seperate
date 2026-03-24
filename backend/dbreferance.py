import psycopg2
import json
import os
from collections import defaultdict
from dotenv import load_dotenv
import google.generativeai as genai
import dash
from dash import dcc, html
import plotly.express as px

# -------------------------------
# 1. Connect to Database
# -------------------------------
def get_schema_metadata(connection_string):
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()

    # Get tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = [row[0] for row in cursor.fetchall()]

    # Get columns
    cursor.execute("""
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public'
    """)
    columns = cursor.fetchall()

    # Get relationships
    cursor.execute("""
        SELECT
            tc.constraint_name AS FK_name,
            tc.table_name AS parent_table,
            ccu.table_name AS referenced_table
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
    """)
    relationships = cursor.fetchall()

    return {
        "tables": tables,
        "columns": [(c[0], c[1], c[2]) for c in columns],
        "relationships": [(r[1], r[2]) for r in relationships]
    }

# -------------------------------
# 2. Group Tables by Relationships
# -------------------------------
def group_tables_by_relationships(schema):
    """Group tables into connected components based on foreign keys."""
    adj = defaultdict(set)
    for parent, referenced in schema["relationships"]:
        adj[parent].add(referenced)
        adj[referenced].add(parent)
        
    visited = set()
    components = []
    
    for table in schema["tables"]:
        if table not in visited:
            component = []
            queue = [table]
            visited.add(table)
            
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Keep groups that have some reference (>= 2 tables interconnected)
            if len(component) > 1:
                components.append(component)
                
    grouped_schemas = []
    for i, component in enumerate(components):
        group_cols = [c for c in schema["columns"] if c[0] in component]
        group_rels = [r for r in schema["relationships"] if r[0] in component and r[1] in component]
        
        grouped_schemas.append({
            "group_name": f"Group {i+1} ({', '.join(component)})",
            "schema": {
                "tables": component,
                "columns": group_cols,
                "relationships": group_rels
            }
        })
        
    return grouped_schemas

# -------------------------------
# 3. Ask Gemini for KPIs
# -------------------------------
def generate_kpis(schema_metadata, group_name):
    prompt = f"""
    You are a data analyst. Given this database schema for {group_name}:
    {json.dumps(schema_metadata, indent=2)}

    Suggest 3-5 useful business KPIs and provide clearly formatted SQL queries to calculate them.
    Return ONLY a raw JSON array of objects. Do not include markdown blocks like ```json or any other text.
    Each object must have exactly two keys: "name" (a string for the KPI name) and "sql" (a string for the SQL query).
    """

    # Load environment variables from .env
    load_dotenv()
    
    # If you have an API key, set it as an environment variable or paste it here instead of the default value.
    api_key = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)

    # Clean up the output in case Gemini wraps it in standard markdown block
    response_text = response.text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
        
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return response_text.strip()

# -------------------------------
# 4. Build Dashboard (Dash + Plotly)
# -------------------------------
def build_dashboard(groups_kpi_data):
    app = dash.Dash(__name__)

    dashboard_layout = [html.H1("AI-Generated Group-wise KPI Dashboard")]

    for group_data in groups_kpi_data:
        group_name = group_data["group_name"]
        try:
            kpis = json.loads(group_data["kpis_json"])
        except json.JSONDecodeError:
            kpis = [{"name": "Error Parse", "sql": "Could not parse KPIs JSON for this group."}]

        # Create simple KPI cards for this group
        cards = [html.Div([
            html.H4(kpi.get("name", "Unknown KPI")),
            html.P(f"Query: {kpi.get('sql', 'N/A')}")
        ], style={"border": "1px solid #ccc", "padding": "10px", "margin": "10px", "backgroundColor": "#f9f9f9", "flex": "1", "minWidth": "300px"}) for kpi in kpis]

        # Create a section for the group
        group_section = html.Div([
            html.H2(group_name, style={"marginTop": "30px", "borderBottom": "2px solid #ccc", "paddingBottom": "10px"}),
            html.Div(cards, style={"display": "flex", "flexWrap": "wrap"})
        ], style={"marginBottom": "40px"})
        
        dashboard_layout.append(group_section)

    app.layout = html.Div(dashboard_layout, style={"fontFamily": "Arial, sans-serif", "padding": "20px"})

    app.run(debug=True)

# -------------------------------
# 5. Main Flow
# -------------------------------
if __name__ == "__main__":
    # Update with your actual PostgreSQL connection details
    connection_string = "dbname=comienzonew user=postgres password=jaTHU@12 host=localhost port=5432"

    print("Fetching schema metadata...")
    schema = get_schema_metadata(connection_string)
    
    # 1. Group tables by FK relationships
    print("Grouping tables by foreign key relationships...")
    grouped_schemas = group_tables_by_relationships(schema)
    
    if not grouped_schemas:
        print("No interconnected table groups found based on foreign keys. Dashboard will be empty.")
        build_dashboard([])
    else:
        # 2. Ask Gemini for KPIs per group
        groups_kpi_data = []
        for g in grouped_schemas:
            print(f"Generating KPIs for {g['group_name']}...")
            kpis_json = generate_kpis(g["schema"], g["group_name"])
            groups_kpi_data.append({
                "group_name": g["group_name"],
                "kpis_json": kpis_json
            })
            
        print("Starting Dashboard...")
        # 3. Build Dashboard
        build_dashboard(groups_kpi_data)