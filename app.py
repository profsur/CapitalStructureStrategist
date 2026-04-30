import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
import numpy.linalg as la
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
import warnings
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Capital Structure Strategist", layout="wide")
st.title("📊 Corporate Life-Cycle & Capital Structure Strategist")
st.markdown("Analyze how cash-flow life stages dictate leverage capacity and structural risk over a 25-year horizon.")

# --- 2. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    # Loading the 25-year panel data
    df = pd.read_stata('nf400withMktRet25yrs.dta')
    
    # Sort data to accurately calculate lagged variables
    df = df.sort_values(by=['companyname', 'year'])
    
    # Calculate Lagged Leverage (t-1) for dynamic impact analysis
    if 'leverage' in df.columns:
        df['leverage_lag1'] = df.groupby('companyname')['leverage'].shift(1)
    
    # Clean up categories for UI
    if 'corplifestage' in df.columns:
        df['corplifestage'] = df['corplifestage'].astype(str)
        
    return df

df = load_data()

# --- HAUSMAN TEST FUNCTION ---
def hausman_test(fe_model, re_model):
    """Calculates the Hausman test statistic to choose between FE and RE."""
    b = fe_model.params
    B = re_model.params
    v_b = fe_model.cov
    v_B = re_model.cov
    df_haus = b.size
    try:
        chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
        pval = stats.chi2.sf(chi2, df_haus)
        return chi2, pval
    except la.LinAlgError:
        return None, None

# --- 3. SIDEBAR: CONTROL PANEL ---
st.sidebar.header("Control Panel")

# Filter by Industry
if 'industrygroup' in df.columns:
    industries = ["All"] + list(df['industrygroup'].dropna().unique())
    selected_industry = st.sidebar.selectbox("Select Industry Sector:", industries)
    if selected_industry != "All":
        filtered_df = df[df['industrygroup'] == selected_industry]
    else:
        filtered_df = df.copy()
else:
    filtered_df = df.copy()
    selected_industry = "All Sectors"

# Select Target Company
if 'companyname' in filtered_df.columns:
    companies = list(filtered_df['companyname'].dropna().unique())
    selected_company = st.sidebar.selectbox("Select Target Company:", companies)

    # Isolate Target Company Data
    company_df = filtered_df[filtered_df['companyname'] == selected_company]
    latest_year = company_df['year'].max() if 'year' in company_df.columns else None
    latest_data = company_df[company_df['year'] == latest_year].iloc[0] if (latest_year and not company_df.empty) else None
else:
    latest_data = None
    st.sidebar.warning("Company Name column missing.")

# --- MAIN DASHBOARD ---
if latest_data is not None and 'leverage' in df.columns:
    
    # --- SECTION 1: THE DIAGNOSTIC ---
    st.header(f"1. Life-Stage Diagnostic: {selected_company} ({latest_year})")
    cols = st.columns(4)
    cols[0].metric("Current Life Stage", latest_data['corplifestage'])
    cols[1].metric("Operating CF (NCFO)", f"{latest_data['ncfo']:.2f}" if 'ncfo' in latest_data and pd.notnull(latest_data['ncfo']) else "N/A")
    cols[2].metric("Investing CF (NCFI)", f"{latest_data['ncfi']:.2f}" if 'ncfi' in latest_data and pd.notnull(latest_data['ncfi']) else "N/A")
    cols[3].metric("Financing CF (NCFF)", f"{latest_data['ncff']:.2f}" if 'ncff' in latest_data and pd.notnull(latest_data['ncff']) else "N/A")
    st.info("**Strategic Insight:** A firm's life stage is dictated by its cash flow footprint. The rules of capital structure optimization shift entirely based on the stage shown above.")
    st.divider()

    # --- SECTION 2: THE LEVERAGE BENCHMARK ---
    st.header("2. Leverage Benchmarking by Life Stage")
    fig_bench = px.box(filtered_df, x="corplifestage", y="leverage", 
                       title=f"Leverage Distribution ({selected_industry})",
                       color="corplifestage")
    if pd.notnull(latest_data['leverage']):
        fig_bench.add_scatter(x=[latest_data['corplifestage']], y=[latest_data['leverage']], 
                              mode='markers', marker=dict(color='black', size=15, symbol='star'),
                              name=f"{selected_company}")
    st.plotly_chart(fig_bench, use_container_width=True)
    st.divider()

    # --- SECTION 3: AGGREGATIVE TIME TRENDS ---
    st.header("3. Macro Aggregative Trends: Leverage Over Time")
    trend_view = st.radio("Select Trend View:", ["Aggregate Market Average", "Average by Corporate Life Stage"], horizontal=True)
    
    if trend_view == "Aggregate Market Average":
        trend_df = filtered_df.groupby('year')['leverage'].mean().reset_index()
        fig_trend = px.line(trend_df, x="year", y="leverage", title=f"Overall Average Leverage ({selected_industry})", markers=True)
        fig_trend.update_traces(line_color='black', line_width=3)
    else:
        trend_df = filtered_df.groupby(['year', 'corplifestage'])['leverage'].mean().reset_index()
        fig_trend = px.line(trend_df, x="year", y="leverage", color="corplifestage", title=f"Average Leverage by Life Stage ({selected_industry})", markers=True)
        
    st.plotly_chart(fig_trend, use_container_width=True)
    st.divider()

    # --- SECTION 4: DYNAMIC IMPACT (LAGGED LEVERAGE) ---
    st.header("4. Dynamic Impact: Capital Structure Stickiness")
    lag_df = filtered_df.dropna(subset=['leverage', 'leverage_lag1'])
    fig_lag = px.scatter(lag_df, x="leverage_lag1", y="leverage", color="corplifestage", opacity=0.6,
                         title="Current vs. Previous Leverage (t vs t-1)")
    
    comp_lag_df = company_df.dropna(subset=['leverage', 'leverage_lag1'])
    if not comp_lag_df.empty:
        fig_lag.add_scatter(x=comp_lag_df['leverage_lag1'], y=comp_lag_df['leverage'],
                            mode='lines+markers', line=dict(color='black', width=2), name=f"{selected_company} Trajectory")
    st.plotly_chart(fig_lag, use_container_width=True)
    st.divider()

    # --- SECTION 6: THE QUANT ENGINE (ECONOMETRICS) ---
    st.header("5. The Quant Engine: Rigorous Determinant Analysis")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Econometric Settings")
    
    num_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    valid_x_cols = [c for c in num_cols if c not in ['year', 'indexdate', 'industrygroupcode', 'leverage']]
    
    # Hardcoded default variables matching your Stata model
    master_vars = ['prof', 'tang', 'dvnd', 'taxShield', 'pmShare', 'GFC', 'ibc2016', 'dcovid20less', 'returnIndexClosing']
    default_selection = [v for v in master_vars if v in valid_x_cols]
    
    reg_vars = st.sidebar.multiselect("Select Independent Variables (X):", valid_x_cols, default=default_selection)
    
    # The toggle to simulate "i.corplifestage"
    include_stage_dummies = st.sidebar.checkbox("Include Life-Stage Dummies (i.corplifestage)", value=True)
    
    if len(reg_vars) > 0:
        # Prepare columns to keep
        cols_to_keep = ['companyname', 'year', 'leverage'] + reg_vars
        if include_stage_dummies and 'corplifestage' in filtered_df.columns:
            cols_to_keep.append('corplifestage')
            
        reg_df = filtered_df[cols_to_keep].dropna()
        
        # Format year to integer for the Panel Index
        if pd.api.types.is_datetime64_any_dtype(reg_df['year']):
            reg_df['year'] = reg_df['year'].dt.year
        else:
            reg_df['year'] = pd.to_numeric(reg_df['year'], errors='coerce').astype(int)
            
        # Set Multi-Index required for PanelOLS
        panel_data = reg_df.set_index(['companyname', 'year'])
        
        Y = panel_data['leverage']
        X_inputs = panel_data[reg_vars].copy()
        
        # Dynamically create dummy variables for Life Stage (replicating i.corplifestage)
        if include_stage_dummies and 'corplifestage' in panel_data.columns:
            dummies = pd.get_dummies(panel_data['corplifestage'], drop_first=True, dtype=float, prefix="Stage")
            X_inputs = pd.concat([X_inputs, dummies], axis=1)
            
        X = sm.add_constant(X_inputs)
        
        try:
            mod_fe = PanelOLS(Y, X, entity_effects=True, drop_absorbed=True)
            res_fe = mod_fe.fit(cov_type='robust')
            
            mod_re = RandomEffects(Y, X)
            res_re = mod_re.fit(cov_type='robust')
            
            chi2, pval = hausman_test(res_fe, res_re)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Selection Verdict")
                if pval is not None:
                    if pval < 0.05:
                        st.success(f"**Fixed Effects (FE) Recommended.** (Hausman P-value: {pval:.4f})")
                        chosen_res = res_fe
                        model_name = "Fixed Effects"
                    else:
                        st.info(f"**Random Effects (RE) Recommended.** (Hausman P-value: {pval:.4f})")
                        chosen_res = res_re
                        model_name = "Random Effects"
                else:
                    st.warning("Hausman test failed (common with highly collinear variables). Defaulting to Fixed Effects.")
                    chosen_res = res_fe
                    model_name = "Fixed Effects"

            with col2:
                st.subheader("Model Fit")
                st.metric("Model Type", model_name)
                r2_val = chosen_res.rsquared_within if model_name == "Fixed Effects" else chosen_res.rsquared
                st.metric("R-Squared (Explained Variance)", f"{r2_val * 100:.2f}%")

            st.subheader(f"{model_name} Results (Coefficients & Significance)")
            results_df = pd.DataFrame({
                "Coefficient": chosen_res.params,
                "Std Error": chosen_res.std_errors,
                "T-Stat": chosen_res.tstats,
                "P-Value": chosen_res.pvalues
            })
            results_df['Significant?'] = results_df['P-Value'].apply(lambda x: "✅ Yes" if x < 0.05 else "❌ No")
            st.dataframe(results_df.style.format({"Coefficient": "{:.4f}", "Std Error": "{:.4f}", "T-Stat": "{:.2f}", "P-Value": "{:.4f}"}))
            
        except Exception as e:
            st.error(f"Econometric Engine Error: {e}. Try selecting different variables.")
    else:
        st.info("👈 Select at least one independent variable in the sidebar to run the Quant Engine.")
else:
    # --- SECTION 7: RESEARCH & INSIGHTS ---
    st.header("7. Research & Insights")
    st.markdown("Download the foundational white paper summarizing the econometric findings of the 25-year Indian corporate panel study.")
    
    try:
        with open("Capital_Structure_White_Paper.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="📥 Download White Paper (PDF)",
                           data=PDFbyte,
                           file_name="Life_Cycle_Leverage_Playbook.pdf",
                           mime='application/octet-stream')
    except FileNotFoundError:
        st.warning("White Paper PDF is currently being generated. Please check back shortly.")

else:
    st.error("Missing required data columns (like 'leverage' or 'companyname'). Please check your dataset.")