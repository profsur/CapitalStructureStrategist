import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
import numpy.linalg as la
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Capital Structure Strategist", layout="wide")
st.title("📊 Corporate Life-Cycle & Capital Structure Strategist")
st.markdown("Analyze how cash-flow life stages dictate leverage capacity and structural risk.")

# --- 2. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    df = pd.read_stata('sp401nf24y_furtherEd_oldCLS_modifiedDickinson02.dta')
    df = df.sort_values(by=['companyname', 'year'])
    df['leverage_lag1'] = df.groupby('companyname')['leverage'].shift(1)
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
    df = b.size
    try:
        chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
        pval = stats.chi2.sf(chi2, df)
        return chi2, pval
    except la.LinAlgError:
        return None, None

# --- 3. SIDEBAR: CONTROL PANEL ---
st.sidebar.header("Control Panel")
industries = ["All"] + list(df['industrygroup'].dropna().unique())
selected_industry = st.sidebar.selectbox("Select Industry Sector:", industries)

if selected_industry != "All":
    filtered_df = df[df['industrygroup'] == selected_industry]
else:
    filtered_df = df.copy()

companies = list(filtered_df['companyname'].dropna().unique())
selected_company = st.sidebar.selectbox("Select Target Company:", companies)

company_df = filtered_df[filtered_df['companyname'] == selected_company]
latest_year = company_df['year'].max()
latest_data = company_df[company_df['year'] == latest_year].iloc[0] if not company_df.empty else None

# --- MAIN DASHBOARD ---
if latest_data is not None:
    
    # [SECTIONS 1 TO 5 REMAIN THE SAME AS V1]
    st.header(f"1. Life-Stage Diagnostic: {selected_company} ({latest_year})")
    cols = st.columns(4)
    cols[0].metric("Current Life Stage", latest_data['corplifestage'])
    cols[1].metric("Operating CF (NCFO)", f"{latest_data['ncfo']:.2f}" if pd.notnull(latest_data['ncfo']) else "N/A")
    cols[2].metric("Investing CF (NCFI)", f"{latest_data['ncfi']:.2f}" if pd.notnull(latest_data['ncfi']) else "N/A")
    cols[3].metric("Financing CF (NCFF)", f"{latest_data['ncff']:.2f}" if pd.notnull(latest_data['ncff']) else "N/A")
    st.divider()

    st.header("2. Leverage Benchmarking by Life Stage")
    fig_bench = px.box(filtered_df, x="corplifestage", y="leverage", 
                       title=f"Leverage Distribution ({selected_industry})",
                       category_orders={"corplifestage": ["Startup", "Growth", "Maturity", "Shakeout1", "Shakeout2", "Shakeout3", "Decline", "Decay"]},
                       color="corplifestage")
    if pd.notnull(latest_data['leverage']):
        fig_bench.add_scatter(x=[latest_data['corplifestage']], y=[latest_data['leverage']], 
                              mode='markers', marker=dict(color='black', size=15, symbol='star'),
                              name=f"{selected_company}")
    st.plotly_chart(fig_bench, use_container_width=True)
    st.divider()

    st.header("3. Macro Aggregative Trends")
    trend_df = filtered_df.groupby(['year', 'corplifestage'])['leverage'].mean().reset_index()
    fig_trend = px.line(trend_df, x="year", y="leverage", color="corplifestage", title="Average Leverage Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.divider()

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
    
    st.header("5. Determinants Playground")
    num_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    valid_x_cols = [c for c in num_cols if c not in ['year', 'indexdate', 'industrygroupcode']]
    selected_x = st.selectbox("Select X-Axis Determinant:", valid_x_cols, index=valid_x_cols.index('ncfo') if 'ncfo' in valid_x_cols else 0)
    fig_scatter = px.scatter(filtered_df, x=selected_x, y="leverage", color="corplifestage", opacity=0.5)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.divider()

    # --- SECTION 6: THE QUANT ENGINE (ECONOMETRICS) ---
    st.header("6. The Quant Engine: Rigorous Determinant Analysis")
    st.write("Run actual panel regressions to test if visual trends hold up to statistical scrutiny. The tool will automatically run a Hausman test to select the right model.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Econometric Settings")
    reg_vars = st.sidebar.multiselect("Select Independent Variables (X):", 
                                      valid_x_cols, 
                                      default=['ncfo', 'ncfi'] if 'ncfo' in valid_x_cols else valid_x_cols[:2])
    
    if len(reg_vars) > 0:
        # Prepare Panel Data
        reg_df = filtered_df[['companyname', 'year', 'leverage'] + reg_vars].dropna()
        
        # We need year as an integer/datetime for panel data and set multi-index
        reg_df['year'] = pd.to_datetime(reg_df['year'], format='%Y').dt.year
        panel_data = reg_df.set_index(['companyname', 'year'])
        
        Y = panel_data['leverage']
        X = sm.add_constant(panel_data[reg_vars])
        
        try:
            # 1. Run Fixed Effects Model
            mod_fe = PanelOLS(Y, X, entity_effects=True, drop_absorbed=True)
            res_fe = mod_fe.fit(cov_type='robust')
            
            # 2. Run Random Effects Model
            mod_re = RandomEffects(Y, X)
            res_re = mod_re.fit(cov_type='robust')
            
            # 3. Run Hausman Test
            chi2, pval = hausman_test(res_fe, res_re)
            
            # --- UI RENDERING ---
            col1, col2 = st.columns(2)
            
            # Hausman Verdict
            with col1:
                st.subheader("Model Selection Verdict")
                if pval is not None:
                    if pval < 0.05:
                        st.success(f"**Fixed Effects (FE) Recommended.** (Hausman P-value: {pval:.4f})")
                        st.write("*Insight:* Unobserved, firm-specific characteristics (like corporate culture or management style) are heavily influencing capital structure. Random Effects would be biased.")
                        chosen_res = res_fe
                        model_name = "Fixed Effects"
                    else:
                        st.info(f"**Random Effects (RE) Recommended.** (Hausman P-value: {pval:.4f})")
                        st.write("*Insight:* Firm-specific characteristics are strictly uncorrelated with the independent variables. We can safely generalize across the panel.")
                        chosen_res = res_re
                        model_name = "Random Effects"
                else:
                    st.warning("Hausman test matrix inversion failed (common with highly collinear variables). Defaulting to Fixed Effects.")
                    chosen_res = res_fe
                    model_name = "Fixed Effects"

            # R-Squared and Stats
            with col2:
                st.subheader("Model Fit")
                st.metric("Model Type", model_name)
                # Handle different R2 attributes between FE and RE
                r2_val = chosen_res.rsquared_within if model_name == "Fixed Effects" else chosen_res.rsquared
                st.metric("R-Squared (Explained Variance)", f"{r2_val * 100:.2f}%")
                st.write(f"*Insight:* The selected variables explain {r2_val * 100:.2f}% of the variance in leverage.")

            # Summary Results Table
            st.subheader(f"{model_name} Results (Coefficients & Significance)")
            
            results_df = pd.DataFrame({
                "Coefficient": chosen_res.params,
                "Std Error": chosen_res.std_errors,
                "T-Stat": chosen_res.tstats,
                "P-Value": chosen_res.pvalues
            })
            
            # Add a visual "Significance" column
            results_df['Significant?'] = results_df['P-Value'].apply(lambda x: "✅ Yes" if x < 0.05 else "❌ No")
            st.dataframe(results_df.style.format({"Coefficient": "{:.4f}", "Std Error": "{:.4f}", "T-Stat": "{:.2f}", "P-Value": "{:.4f}"}))
            
        except Exception as e:
            st.error(f"Econometric Engine Error: {e}. Try selecting different variables.")
    else:
        st.info("👈 Select at least one independent variable in the sidebar to run the Quant Engine.")
else:
    st.warning("Please select a valid company with data available.")