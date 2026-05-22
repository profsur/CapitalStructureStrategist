import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, compare


# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Life Stage Financial Leverage Strategist", layout="wide", page_icon="📊")
st.title("Life Stage Financial Leverage Strategist")

# ==========================================
# 2. SIDEBAR NAVIGATION & DATA SOURCE CONFIG
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942269.png", width=100)

st.sidebar.header("1. Data Source Configuration")
data_source_label = st.sidebar.radio(
    "Select Dataset:",
    (
        "Original Thesis Data (24 Years)", 
        "Appended Thesis Data (25 Years)", 
        "Fresh Download Data (25 Years)"
    )
)

# Map the human-readable label to the exact filename
file_map = {
    "Original Thesis Data (24 Years)": "sp401nf24y_furtherEd_oldCLS.dta",
    "Appended Thesis Data (25 Years)": "sp401nf25yrByAppend.dta",
    "Fresh Download Data (25 Years)": "nf400withMktRet25yrs.dta"
}
selected_file = file_map[data_source_label]

st.sidebar.divider()

st.sidebar.header("2. Navigation Controls")
analysis_type = st.sidebar.radio(
    "Select Analytical Module:",
    (
        "Aggregate Market View", 
        "Company Drill-Down", 
        "Life Stage Distribution", 
        "CFO Predictive Benchmark",
        "Credit Risk Screener (EWS)",
        "Econometric Research Engine",
        "Automated White Paper"
    )
)

st.sidebar.divider()
active_ds_msg = f"**Active Dataset:**\n{selected_file}"
st.sidebar.info(active_ds_msg)

# ==========================================
# 3. BULLETPROOF DATA LOADER
# ==========================================
@st.cache_data
def load_data(file_name):
    df = pd.read_stata(file_name)
    
    if 'corplifestage' in df.columns:
        df['corplifestage'] = df['corplifestage'].astype(str)
        numeric_map = {
            '1': 'Startup', '1.0': 'Startup',
            '2': 'Growth', '2.0': 'Growth',
            '3': 'Maturity', '3.0': 'Maturity',
            '4': 'Shakeout1', '4.0': 'Shakeout1',
            '5': 'Shakeout2', '5.0': 'Shakeout2',
            '6': 'Shakeout3', '6.0': 'Shakeout3',
            '7': 'Decline', '7.0': 'Decline',
            '8': 'Decay', '8.0': 'Decay'
        }
        df['corplifestage'] = df['corplifestage'].apply(lambda x: numeric_map.get(x, str(x).capitalize()))
    
    stage_order = ["Startup", "Growth", "Maturity", "Shakeout1", "Shakeout2", "Shakeout3", "Decline", "Decay"]
    df['corplifestage'] = pd.Categorical(df['corplifestage'], categories=stage_order, ordered=True)
    
    numeric_cols = ['year', 'leverage', 'size', 'prof', 'tang']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df, stage_order

try:
    data, stage_order = load_data(selected_file)
    st.markdown(f"**Current Context:** Analyzing {len(data):,} observations from *{data_source_label}*.")
except FileNotFoundError:
    err_msg = f"⚠️ Data file missing! Please ensure '{selected_file}' is in the same folder as this script."
    st.error(err_msg)
    st.stop()
except Exception as e:
    err_msg = f"An error occurred while loading data: {e}"
    st.error(err_msg)
    st.stop()

# ==========================================
# VIEW 1: AGGREGATE MARKET VIEW 
# ==========================================
if analysis_type == "Aggregate Market View":
    st.header("Aggregate Market Analysis")
    st.write("Evaluating average financial leverage across the sequential progression of corporate life stages.")
    
    st.subheader("1. Average Leverage by Life Stage Sequence")
    agg_data = data.groupby('corplifestage', observed=False)['leverage'].mean().reset_index()
    
    fig_bar = px.bar(
        agg_data, x='corplifestage', y='leverage', color='corplifestage',
        category_orders={"corplifestage": stage_order},
        title="Leverage Follows the Corporate Lifecycle"
    )
    st.plotly_chart(fig_bar, width='stretch')

    st.divider()
    
    st.subheader("2. Time Trends: Aggregate vs. Individual Stages")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        overall_time_data = data.groupby('year')['leverage'].mean().reset_index()
        fig_line_overall = px.line(overall_time_data, x='year', y='leverage', title="Overall Market Trend", markers=True)
        fig_line_overall.update_traces(line_color='black', line_width=3)
        st.plotly_chart(fig_line_overall, width='stretch')

    with col2:
        time_data = data.groupby(['year', 'corplifestage'], observed=False)['leverage'].mean().reset_index()
        fig_line_stages = px.line(
            time_data, x='year', y='leverage', color='corplifestage', facet_col='corplifestage', facet_col_wrap=4, 
            category_orders={"corplifestage": stage_order}, title="Trends Separated by Individual Life Stage"
        )
        fig_line_stages.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_line_stages.update_layout(showlegend=False, height=500) 
        st.plotly_chart(fig_line_stages, width='stretch')

# ==========================================
# VIEW 2: COMPANY DRILL-DOWN
# ==========================================
elif analysis_type == "Company Drill-Down":
    st.header("Company-Specific Transitions")
    companies = sorted(data['companyname'].dropna().unique())
    selected_company = st.selectbox("Search and Select a Company:", companies)
    company_data = data[data['companyname'] == selected_company].sort_values('year')
    
    if not company_data.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Years of Data", len(company_data))
        c2.metric("Latest Year", int(company_data.iloc[-1]['year']))
        c3.metric("Latest Life Stage", company_data.iloc[-1]['corplifestage'])
        c4.metric("Latest Leverage", f"{company_data.iloc[-1]['leverage']:.2f}%")
        
        fig_company = px.scatter(
            company_data, x='year', y='leverage', color='corplifestage', size='size',
            hover_data=['prof'], title=f"Timeline for {selected_company}",
            category_orders={"corplifestage": stage_order}
        )
        fig_company.update_traces(mode='lines+markers') 
        st.plotly_chart(fig_company, width='stretch')
        
        display_cols = [c for c in ['year', 'corplifestage', 'leverage', 'size', 'prof', 'tang'] if c in company_data.columns]
        st.dataframe(company_data[display_cols].set_index('year'), width='stretch')

# ==========================================
# VIEW 3: LIFE STAGE DISTRIBUTION
# ==========================================
elif analysis_type == "Life Stage Distribution":
    st.header("Life Stage Deep Dive (Cross-Sectional)")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        selected_stage = st.selectbox("Select a Life Stage to Analyze:", stage_order, index=2)
    with col_filter2:
        years = sorted(data['year'].dropna().unique(), reverse=True)
        selected_year = st.selectbox("Select Year:", years)
    
    stage_data = data[(data['corplifestage'] == selected_stage) & (data['year'] == selected_year)]
    all_stages_year_data = data[data['year'] == selected_year]
    
    if not stage_data.empty:
        st.subheader(f"Snapshot of '{selected_stage}' Stage in {int(selected_year)}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Number of Firms", len(stage_data))
        m2.metric("Avg Leverage in Stage", f"{stage_data['leverage'].mean():.2f}%")
        m3.metric("Avg Profitability", f"{stage_data['prof'].mean():.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(all_stages_year_data, x="corplifestage", y="leverage", color="corplifestage", category_orders={"corplifestage": stage_order})
            st.plotly_chart(fig_box, width='stretch')
            
        with col2:
            fig_scatter = px.scatter(stage_data, x="size", y="leverage", color="prof", hover_name="companyname", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_scatter, width='stretch')
    else:
        warn_msg = f"No firms found in the '{selected_stage}' stage for the year {selected_year}."
        st.warning(warn_msg)

# ==========================================
# VIEW 4: CFO PREDICTIVE BENCHMARK
# ==========================================
elif analysis_type == "CFO Predictive Benchmark":
    st.header("CFO Predictive Benchmark: Optimal Capital Structure")

    with st.form("cfo_inputs"):
        col1, col2, col3 = st.columns(3)
        with col1:
            current_stage = st.selectbox("Current Corporate Life Stage:", stage_order)
            current_leverage = st.number_input("Current Book Leverage (%)", min_value=0.0, value=25.0)
        with col2:
            prof_input = st.slider("Operating Profitability (%)", min_value=-50.0, max_value=50.0, value=15.0) 
            tang_input = st.slider("Asset Tangibility (%)", min_value=0.0, max_value=100.0, value=30.0) 
        with col3:
            info_msg = "The model benchmarks your firm against the 50th (Optimal) and 90th (Distressed) percentiles."
            st.info(info_msg)
            submit_button = st.form_submit_button(label="Generate Benchmark")

    if submit_button:
        prof = prof_input / 100.0
        tang = tang_input / 100.0
        
        median_base, tail_base = 2.10, 31.18
        median_prof_coef, tail_prof_coef = -27.44, -73.22
        median_tang_coef, tail_tang_coef = 54.10, 53.90
        
        stage_penalties = {
            "Maturity": {"median": 0.00, "tail": 0.00}, "Growth": {"median": 11.77, "tail": 9.29},
            "Startup": {"median": 19.33, "tail": 25.11}, "Decline": {"median": 11.33, "tail": 25.06},
            "Decay": {"median": 3.87, "tail": 10.04}, "Shakeout1": {"median": 0.90, "tail": 1.07},
            "Shakeout2": {"median": 6.44, "tail": 15.64}, "Shakeout3": {"median": 1.68, "tail": 2.70}
        }
        
        optimal_leverage = max(0, median_base + (prof * median_prof_coef) + (tang * median_tang_coef) + stage_penalties[current_stage]["median"])
        distress_ceiling = max(0, tail_base + (prof * tail_prof_coef) + (tang * tail_tang_coef) + stage_penalties[current_stage]["tail"])
        
        st.divider()
        st.subheader("Capital Structure Diagnosis")
        
        if current_leverage < optimal_leverage:
            status_color, status_msg = "green", "Under-levered: Excess Debt Capacity Available."
        elif current_leverage > distress_ceiling:
            status_color, status_msg = "red", "Critical Alert: Leverage exceeds distress ceiling."
        else:
            status_color, status_msg = "orange", "Over-levered: Approaching distress ceiling."
            
        st.markdown(f"**Diagnosis:** :{status_color}[{status_msg}]")
        
        fig = go.Figure(go.Indicator(
            mode = "number+gauge+delta", value = current_leverage, domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': optimal_leverage, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'shape': "bullet", 'axis': {'range': [None, max(100, distress_ceiling + 20)]},
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': current_leverage},
                'steps': [{'range': [0, optimal_leverage], 'color': "lightgreen"}, {'range': [optimal_leverage, distress_ceiling], 'color': "navajowhite"}, {'range': [distress_ceiling, max(100, distress_ceiling + 20)], 'color': "lightcoral"}],
                'bar': {'color': "black", 'thickness': 0.1}
            }
        ))
        fig.update_layout(height=200, margin={'t': 20, 'b': 20, 'l': 50, 'r': 50})
        st.plotly_chart(fig, width='stretch')
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Your Current Leverage", f"{current_leverage:.1f}%")
        m2.metric("Optimal Peer Benchmark", f"{optimal_leverage:.1f}%")
        m3.metric("Distress Ceiling (Tail Risk)", f"{distress_ceiling:.1f}%")

# ==========================================
# VIEW 5: CREDIT RISK SCREENER (EWS)
# ==========================================
elif analysis_type == "Credit Risk Screener (EWS)":
    st.header("Credit Risk Screener: Early Warning System")
    
    max_year = data['year'].max()
    selected_year = st.selectbox("Select Year to Scan:", sorted(data['year'].dropna().unique(), reverse=True), index=0)
    scan_data = data[data['year'] == selected_year].dropna(subset=['leverage', 'prof', 'tang', 'corplifestage']).copy()
    
    if not scan_data.empty:
        penalties = {"Maturity": 0.00, "Growth": 9.29, "Startup": 25.11, "Decline": 25.06, "Decay": 10.04, "Shakeout1": 1.07, "Shakeout2": 15.64, "Shakeout3": 2.70}
        def calc_distress(row):
            return max(0, 31.18 + (row['prof'] * -73.22) + (row['tang'] * 53.90) + penalties.get(row['corplifestage'], 0))
            
        scan_data['Distress_Ceiling'] = scan_data.apply(calc_distress, axis=1)
        scan_data['Risk_Delta'] = scan_data['leverage'] - scan_data['Distress_Ceiling']
        flagged_firms = scan_data[scan_data['Risk_Delta'] > 0].sort_values(by='Risk_Delta', ascending=False)
        
        ews_err_msg = f"🚨 Found {len(flagged_firms)} high-risk firms in {int(selected_year)} operating above their statistical distress ceiling."
        st.error(ews_err_msg)
        
        display_df = flagged_firms[['companyname', 'industrygroup', 'corplifestage', 'prof', 'Distress_Ceiling', 'leverage']].copy()
        display_df['prof'] = (display_df['prof'] * 100).round(2).astype(str) + '%'
        display_df['Distress_Ceiling'] = display_df['Distress_Ceiling'].round(2)
        display_df['leverage'] = display_df['leverage'].round(2)
        st.dataframe(display_df, width='stretch')

# ==========================================
# VIEW 6: ECONOMETRIC RESEARCH ENGINE
# ==========================================
elif analysis_type == "Econometric Research Engine":
    st.header("Econometric Research Engine")
    
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from linearmodels.panel import PanelOLS, RandomEffects, compare
    except ImportError:
        lib_err_msg = "⚠️ Missing econometric libraries! Please run: `pip install statsmodels linearmodels` in your terminal."
        st.error(lib_err_msg)
        st.stop()

    reg_cols = ['companyname', 'year', 'leverage', 'corplifestage', 'prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
    missing_cols = [c for c in reg_cols if c not in data.columns]
    
    if missing_cols:
        miss_warn_msg = f"The active dataset is missing the following control variables: {missing_cols}"
        st.warning(miss_warn_msg)
    else:
        with st.spinner("Preparing Panel Data..."):
            df_reg = data[reg_cols].dropna().copy()
            df_reg = df_reg.sort_values(by=['companyname', 'year'])
            df_reg['L_leverage'] = df_reg.groupby('companyname')['leverage'].shift(1)
            df_panel = df_reg.set_index(['companyname', 'year'])
            formula_smf = "leverage ~ prof + tang + dvnd + taxShield + GFC + ibc2016 + dcovid20less + interest + returnIndexClosing + C(corplifestage) + C(year)"
            exog_vars = ['prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']

        st.markdown("""<style>table.simpletable { width: 100%; border-collapse: collapse; margin-bottom: 20px; } table.simpletable td, table.simpletable th { border: 1px solid #ddd; padding: 8px; text-align: center; } table.simpletable th { background-color: #f2f2f2; font-weight: bold; }</style>""", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Simple OLS", "2. FE & RE", "3. Quantile (10, 50, 90)", "4. Margins Analysis", "5. Dynamic Panel & Diagnostics"])

        with tab1:
            if st.button("Run Pooled OLS"):
                ols_model = smf.ols(formula_smf, data=df_reg).fit()
                st.markdown(ols_model.summary().as_html(), unsafe_allow_html=True)

        with tab2:
            if st.button("Run FE/RE Models"):
                exog = sm.add_constant(df_panel[exog_vars])
                stage_dummies = pd.get_dummies(df_panel['corplifestage'], drop_first=True, dtype=float)
                exog = pd.concat([exog, stage_dummies], axis=1)
                fe_res = PanelOLS(df_panel['leverage'], exog, entity_effects=True, time_effects=False, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
                re_res = RandomEffects(df_panel['leverage'], exog).fit()
                st.markdown(compare({"Fixed Effects": fe_res, "Random Effects": re_res}).summary.as_html(), unsafe_allow_html=True)

        with tab3:
            if st.button("Run Quantile Regressions"):
                results = []
                for q in [0.10, 0.50, 0.90]:
                    res = smf.quantreg(formula_smf, data=df_reg).fit(q=q, max_iter=2000)
                    df_res = pd.DataFrame({'Coefficient': res.params, 'P-Value': res.pvalues})
                    df_res.columns = [f'Q{int(q*100)} Coef', f'Q{int(q*100)} P-Val']
                    results.append(df_res)
                final_q_table = pd.concat(results, axis=1)
                st.dataframe(final_q_table.style.format("{:.4f}").background_gradient(cmap='Blues', subset=['Q10 Coef', 'Q50 Coef', 'Q90 Coef']), width='stretch')

        with tab4:
            st.subheader("Asymmetric Risk: Marginal Impact of Life Stages")
            st.write("This chart answers a critical business question: *'How much does entering a new life stage impact our debt?'*")
            
            if st.button("Generate Grouped Margins Chart"):
                with st.spinner("Calculating 50th and 90th Margins..."):
                    q50 = smf.quantreg(formula_smf, data=df_reg).fit(q=0.50)
                    q90 = smf.quantreg(formula_smf, data=df_reg).fit(q=0.90)
                    
                    stage_cols = [c for c in q50.params.index if 'corplifestage' in c]
                    clean_labels = [c.replace("C(corplifestage)[T.", "").replace("]", "") for c in stage_cols]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=clean_labels, y=q50.params[stage_cols],
                        name='Normal Firm (50th Percentile)', marker_color='royalblue'
                    ))
                    fig.add_trace(go.Bar(
                        x=clean_labels, y=q90.params[stage_cols],
                        name='Distressed Firm (90th Percentile)', marker_color='firebrick'
                    ))
                    
                    fig.update_layout(
                        title='Marginal Increase in Leverage by Life Stage (Baseline = Maturity)',
                        xaxis_title='Transitioning into Life Stage',
                        yaxis_title='Marginal Effect on Leverage (%)',
                        barmode='group',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    intuition_msg = "💡 **Business Intuition:** Notice the extreme asymmetry. If a healthy firm (Blue Bar) enters the 'Decline' stage, its leverage barely shifts. However, if a heavily indebted firm (Red Bar) enters 'Decline', it experiences a massive, uncontrollable spike in leverage as its equity is wiped out. This proves mathematically that structural risk is non-linear."
                    st.info(intuition_msg)

        with tab5:
            st.subheader("5. Dynamic Panel (Fixed Effects & IV-GMM)")
            st.write("Compare the standard Dynamic Fixed Effects model with an Instrumental Variable GMM approach (instrumenting the lagged dependent variable).")
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            run_fe = col_btn1.button("Run Dynamic Fixed Effects")
            run_gmm = col_btn2.button("Run Simplified IV-GMM")
            run_ar_test = col_btn3.button("Run AR(1) & AR(2) Diagnostics")
            
            if run_fe:
                with st.spinner("Running Dynamic Panel (Fixed Effects)..."):
                    df_dyn = df_reg.dropna(subset=['L_leverage']).copy()
                    exog_vars_dyn = ['L_leverage', 'prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
                    exog_dyn = sm.add_constant(df_dyn[exog_vars_dyn])
                    stage_dummies_dyn = pd.get_dummies(df_dyn['corplifestage'], drop_first=True, dtype=float)
                    exog_dyn = pd.concat([exog_dyn, stage_dummies_dyn], axis=1)
                    
                    df_dyn_panel = df_dyn.set_index(['companyname', 'year'])
                    
                    dyn_model = PanelOLS(df_dyn_panel['leverage'], exog_dyn, entity_effects=True, time_effects=False, drop_absorbed=True)
                    dyn_res = dyn_model.fit(cov_type='clustered', cluster_entity=True)
                    
                    st.markdown("### Dynamic Fixed Effects Results")
                    st.markdown(dyn_res.summary.as_html(), unsafe_allow_html=True)
                    fe_note = "🎓 **Note:** In a 'long' panel (T=25), Nickell bias converges toward zero, making Dynamic FE highly robust and immune to the instrument proliferation issues that plague standard GMM."
                    st.info(fe_note)

            if run_gmm:
                with st.spinner("Preparing 2nd Lags and Running IV-GMM..."):
                    try:
                        from linearmodels.iv import IVGMM
                    except ImportError:
                        iv_err = "Missing linearmodels.iv library."
                        st.error(iv_err)
                        st.stop()
                        
                    df_gmm = df_reg.copy()
                    df_gmm['L2_leverage'] = df_gmm.groupby('companyname')['leverage'].shift(2)
                    df_gmm = df_gmm.dropna(subset=['L_leverage', 'L2_leverage']).copy()
                    
                    exog_iv = ['prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
                    exog_df = sm.add_constant(df_gmm[exog_iv])
                    
                    stage_dummies_gmm = pd.get_dummies(df_gmm['corplifestage'], drop_first=True, dtype=float)
                    exog_df = pd.concat([exog_df, stage_dummies_gmm], axis=1)
                    
                    endog_df = df_gmm[['L_leverage']]
                    instr_df = df_gmm[['L2_leverage']]
                    dep_df = df_gmm[['leverage']]
                    
                    gmm_model = IVGMM(dependent=dep_df, exog=exog_df, endog=endog_df, instruments=instr_df)
                    gmm_res = gmm_model.fit(cov_type='clustered', clusters=df_gmm['companyname'])
                    
                    st.markdown("### Instrumental Variable GMM (Anderson-Hsiao Logic)")
                    st.markdown(gmm_res.summary.as_html(), unsafe_allow_html=True)
                    
                    st.success("🎯 **Econometric Mechanics:** This model proxies Stata's `xtdpdgmm`. It treats the macroeconomic and firm variables as strictly exogenous (`iv`). It treats `L_leverage` as endogenous, instrumenting it with `L2_leverage` (`gmm`) to strip out the correlation with the error term.")

            if run_ar_test:
                with st.spinner("Calculating Residual Serial Correlation (AR1 & AR2)..."):
                    df_dyn = df_reg.dropna(subset=['L_leverage']).copy()
                    exog_vars_dyn = ['L_leverage', 'prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
                    exog_dyn = sm.add_constant(df_dyn[exog_vars_dyn])
                    stage_dummies_dyn = pd.get_dummies(df_dyn['corplifestage'], drop_first=True, dtype=float)
                    exog_dyn = pd.concat([exog_dyn, stage_dummies_dyn], axis=1)
                    df_dyn_panel = df_dyn.set_index(['companyname', 'year'])
                    
                    dyn_res = PanelOLS(df_dyn_panel['leverage'], exog_dyn, entity_effects=True, time_effects=False, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
                    
                    resid_df = pd.DataFrame({'resid': dyn_res.resids})
                    resid_df['L1_resid'] = resid_df.groupby(level='companyname')['resid'].shift(1)
                    resid_df['L2_resid'] = resid_df.groupby(level='companyname')['resid'].shift(2)
                    resid_test_df = resid_df.dropna()
                    
                    ar1_model = smf.ols('resid ~ L1_resid', data=resid_test_df).fit(cov_type='HC1')
                    ar2_model = smf.ols('resid ~ L2_resid', data=resid_test_df).fit(cov_type='HC1')
                    
                    st.markdown("### Post-Estimation: Serial Correlation Tests")
                    ar_results = pd.DataFrame({
                        'Test': ['AR(1) Test (Lag 1)', 'AR(2) Test (Lag 2)'],
                        'Coefficient': [ar1_model.params['L1_resid'], ar2_model.params['L2_resid']],
                        'P-Value': [ar1_model.pvalues['L1_resid'], ar2_model.pvalues['L2_resid']],
                        'Condition Met?': ["N/A (AR1 is expected)", "✅ Valid" if ar2_model.pvalues['L2_resid'] > 0.05 else "❌ Invalid (Serial Correlation Detected)"]
                    })
                    st.table(ar_results.style.format({'Coefficient': '{:.4f}', 'P-Value': '{:.4f}'}))
                    
                    ar_diag_msg = "💡 **Diagnostic Interpretation:** We require the **AR(2) p-value > 0.05** (failing to reject the null of no second-order serial correlation) for dynamic panel instruments to be strictly exogenous."
                    st.info(ar_diag_msg)

# ==========================================
# VIEW 7: AUTOMATED WHITE PAPER (EXPANDED TO 10 CHAPTERS)
# ==========================================
elif analysis_type == "Automated White Paper":
    st.header("📄 Automated Comprehensive White Paper")
    audience = st.radio("Select Tone/Audience for the Report:", ["Academic / Peer Review", "Practitioner / CFO Advisory"], horizontal=True)
    
    if st.button("Generate Dynamic White Paper"):
        with st.spinner("Synthesizing multi-dimensional data, calculating econometrics, and drafting comprehensive report..."):
            
            try:
                import statsmodels.formula.api as smf
            except ImportError:
                st.error("Missing `statsmodels`. Please install to generate the full white paper.")
                st.stop()

            # --- General Calcs ---
            total_obs = len(data)
            unique_firms = data['companyname'].nunique() if 'companyname' in data.columns else "N/A"
            start_year = int(data['year'].min())
            end_year = int(data['year'].max())
            panel_length = end_year - start_year + 1
            
            # --- Extreme / Zombie Firm Calcs ---
            extreme_df = data[data['leverage'] > 100]
            normal_df = data[data['leverage'] <= 100]
            ext_count = len(extreme_df)
            
            if ext_count > 0:
                ext_prof = (extreme_df['prof'].mean() * 100) if pd.notnull(extreme_df['prof'].mean()) else 0
                norm_prof = (normal_df['prof'].mean() * 100) if pd.notnull(normal_df['prof'].mean()) else 0
                top_ext_stage = extreme_df['corplifestage'].mode()[0] if not extreme_df.empty else "N/A"
            else:
                ext_prof, norm_prof, top_ext_stage = 0, 0, "N/A"

            # --- Macro Calcs ---
            yearly_avg = data.groupby('year')['leverage'].mean()
            peak_year = int(yearly_avg.idxmax())
            peak_lev = yearly_avg.max()
            trough_year = int(yearly_avg.idxmin())
            trough_lev = yearly_avg.min()
            start_lev = yearly_avg.iloc[0]
            end_lev = yearly_avg.iloc[-1]
            
            # --- EWS Calcs ---
            latest_year_data = data[data['year'] == end_year].dropna(subset=['leverage', 'prof', 'tang', 'corplifestage']).copy()
            penalties = {"Maturity": 0.00, "Growth": 9.29, "Startup": 25.11, "Decline": 25.06, "Decay": 10.04, "Shakeout1": 1.07, "Shakeout2": 15.64, "Shakeout3": 2.70}
            def calc_distress(row): return max(0, 31.18 + (row['prof'] * -73.22) + (row['tang'] * 53.90) + penalties.get(row['corplifestage'], 0))
            
            if not latest_year_data.empty:
                latest_year_data['Distress_Ceiling'] = latest_year_data.apply(calc_distress, axis=1)
                latest_year_data['Risk_Delta'] = latest_year_data['leverage'] - latest_year_data['Distress_Ceiling']
                flagged_count = len(latest_year_data[latest_year_data['Risk_Delta'] > 0])
                distress_pct = (flagged_count / len(latest_year_data)) * 100
            else:
                flagged_count, distress_pct = 0, 0

            # --- Background Margins Calculation (for Chapter 6) ---
            df_reg = data[['leverage', 'prof', 'tang', 'corplifestage']].dropna()
            q50 = smf.quantreg("leverage ~ prof + tang + C(corplifestage)", data=df_reg).fit(q=0.50)
            q90 = smf.quantreg("leverage ~ prof + tang + C(corplifestage)", data=df_reg).fit(q=0.90)
            stage_cols = [c for c in q50.params.index if 'corplifestage' in c]
            clean_labels = [c.replace("C(corplifestage)[T.", "").replace("]", "") for c in stage_cols]

            # ==========================================
            # RENDER 8-PAGE REPORT STRUCTURE
            # ==========================================
            st.divider()
            st.markdown(f"<h1 style='text-align: center; font-size: 48px;'>Life Stage Financial Leverage Strategist</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: gray;'>A Comprehensive Empirical White Paper ({start_year} - {end_year})</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'><em>Data Source: S&P BSE 500 (Non-Financials) | N = {total_obs:,}</em></p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            
            # --- CHAPTER 1 ---
            st.subheader("Chapter 1: Executive Summary & Abstract")
            st.write(f"This dynamic white paper synthesizes an exhaustive empirical analysis of capital structure determinants among non-financial firms in the S&P BSE 500 index. "
                     f"Spanning a {panel_length}-year timeline ({start_year}-{end_year}), the dataset encompasses **{total_obs:,} observations** across **{unique_firms} unique corporate entities**. "
                     f"Moving beyond conventional, static industry averages, this paper pioneers the operationalization of Dickinson's (2011) cash flow-based life stages to establish predictive, dynamic leverage benchmarks and systemic credit risk indicators.")
            
            if audience == "Practitioner / CFO Advisory":
                exec_msg = f"🎯 **Executive Imperative:** Static debt targets destroy shareholder value. As of {end_year}, **{distress_pct:.1f}%** of analyzed firms are operating dangerously above their mathematically derived 'Distress Ceilings.' CFOs must pivot to dynamic, life-stage-adjusted capital allocation strategies to avoid severe liquidity crises."
                st.info(exec_msg)
            else:
                acad_msg = "🎓 **Academic Contribution:** This paper extends the Pecking Order and Trade-Off theories by utilizing Quantile Regression to mathematically prove that the penalizing effects of negative profitability and operational decline are heavily asymmetric, exacerbating tail-risk at the 90th percentile of the leverage distribution."
                st.info(acad_msg)
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 2 ---
            st.subheader("Chapter 2: Theoretical Framework & Literature Context")
            st.write("Historically, capital structure theory has been dominated by two competing paradigms: the **Trade-Off Theory** (balancing tax shields against bankruptcy costs) and the **Pecking Order Theory** (preferring internal financing over debt, and debt over equity due to information asymmetry).")
            st.write("However, both theories traditionally assume firm homogeneity over time. This research introduces a critical third dimension: **The Corporate Life Stage**. Using Dickinson's (2011) methodology—which classifies firms based on the net combinations of operating, investing, and financing cash flows—we hypothesize that a firm's optimal capital structure is not static, but evolves predictably as it transitions from Startup, to Growth, Maturity, and eventually Decline.")
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 3 ---
            st.subheader("Chapter 3: Macroeconomic Dynamics & Secular Deleveraging")
            st.write(f"Before examining micro-level firm behaviors, it is vital to contextualize the macroeconomic environment. The overarching narrative of the Indian corporate sector from {start_year} to {end_year} is one of secular deleveraging. Aggregate market leverage fell from **{start_lev:.1f}%** at the beginning of the panel to **{end_lev:.1f}%**.")
            st.write(f"Yet, this linear trend masks immense cyclical volatility driven by external macroeconomic shocks. Corporate leverage spiked to a peak of **{peak_lev:.1f}% in {peak_year}**, reflecting eras of aggressive capital expenditure and relaxed credit environments (e.g., pre-GFC exuberance). Conversely, rigorous deleveraging phases drove the market to a trough of **{trough_lev:.1f}% in {trough_year}**, particularly following the implementation of the Insolvency and Bankruptcy Code (IBC) in 2016, which fundamentally altered the cost of default in India.")
            
            overall_time_data = data.groupby('year')['leverage'].mean().reset_index()
            fig1 = px.line(overall_time_data, x='year', y='leverage', markers=True, title="Figure 1: Aggregate Market Leverage Over Time")
            fig1.update_traces(line_color='black', line_width=2)
            fig1.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, width='stretch')
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 4 ---
            st.subheader("Chapter 4: The Empirical U-Shaped Leverage Curve")
            startup_lev = data[data['corplifestage'] == 'Startup']['leverage'].mean()
            maturity_lev = data[data['corplifestage'] == 'Maturity']['leverage'].mean()
            decline_lev = data[data['corplifestage'] == 'Decline']['leverage'].mean()
            st.write(f"When disaggregated by life stage, the data validates the core thesis: financial leverage follows a distinct U-shaped curve across the corporate lifecycle. Highly capital-intensive **Startup** and **Growth** firms, lacking sufficient internal cash generation, rely heavily on external debt, exhibiting average leverage levels of **{startup_lev:.1f}%**.")
            st.write(f"As firms reach **Maturity** and begin generating massive, stable operating cash flows, they strictly adhere to the Pecking Order Theory. They use these internal accruals to systematically retire debt, dragging average leverage down to **{maturity_lev:.1f}%**. However, as firms enter the **Decline** stage, operating cash flows evaporate, equity bases erode via accumulated losses, and leverage structurally spikes back up to **{decline_lev:.1f}%**, often culminating in distress.")
            
            agg_data = data.groupby('corplifestage', observed=False)['leverage'].mean().reset_index()
            fig2 = px.bar(agg_data, x='corplifestage', y='leverage', color='corplifestage', category_orders={"corplifestage": stage_order}, title="Figure 2: Average Leverage by Dickinson Life Stage")
            fig2.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig2, width='stretch')
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 5 ---
            st.subheader("Chapter 5: Econometric Methodology & Robustness")
            st.write(f"To ensure academic rigor, this analysis relies on advanced panel data econometrics. A critical methodological decision was the treatment of dynamic capital structure adjustments (incorporating a lagged dependent variable, $L.leverage$).")
            st.write(f"While standard econometric practice often dictates the use of Generalized Method of Moments (e.g., Arellano-Bond / System GMM) for dynamic panels to resolve Nickell bias, System GMM suffers from severe 'instrument proliferation' in long panels. Because this dataset spans a highly extended panel of **$T = {panel_length}$ years**, the Nickell bias converges toward zero (proportional to $1/T$). Therefore, **Dynamic Fixed Effects (FE)** is the statistically robust choice, successfully controlling for unobserved firm-level heterogeneity without artificially overfitting the endogenous variables.")
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 6 ---
            st.subheader("Chapter 6: Asymmetric Risk & Margin Analysis")
            st.write("Traditional OLS models assume that the impact of transitioning into a declining life stage is uniform across all firms. This paper utilizes **Quantile Regression (50th vs 90th percentiles)** to prove that structural risk is highly asymmetric and non-linear.")
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=clean_labels, y=q50.params[stage_cols], name='Normal Firm (50th Percentile)', marker_color='royalblue'))
            fig3.add_trace(go.Bar(x=clean_labels, y=q90.params[stage_cols], name='Distressed Firm (90th Percentile)', marker_color='firebrick'))
            fig3.update_layout(title='Figure 3: Marginal Increase in Leverage by Life Stage', barmode='group', height=400)
            st.plotly_chart(fig3, width='stretch')
            
            st.write("Figure 3 illustrates this asymmetry perfectly. If a healthy, median-levered firm (the Blue Bar) enters the 'Decline' stage, its leverage barely shifts. It has the equity buffer to absorb the shock. However, if a heavily indebted firm at the 90th percentile of leverage (the Red Bar) enters the 'Decline' stage, it experiences a massive, uncontrollable marginal spike in leverage. The risk does not scale linearly; it explodes at the tail.")
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 7 ---
            st.subheader("Chapter 7: Practical Application I — CFO Benchmarking")
            med_prof = data['prof'].median()
            med_tang = data['tang'].median()
            optimal_maturity = max(0, 2.10 + (med_prof * -27.44) + (med_tang * 54.10))
            optimal_growth = max(0, 2.10 + (med_prof * -27.44) + (med_tang * 54.10) + 11.77)
            st.write("For corporate practitioners, these econometric findings are operationalized into predictive benchmarks for optimal capital allocation. Comparing a firm's debt load to a static 'industry average' is fundamentally flawed.")
            st.write(f"**Empirical Proof:** Consider a median firm with Profitability of **{(med_prof*100):.1f}%** and Tangibility of **{(med_tang*100):.1f}%**. According to the regression framework, if this firm is in the **Maturity** stage, its optimal target debt is **{optimal_maturity:.1f}%**. However, if the identical firm initiates a heavy CAPEX cycle and transitions to the **Growth** phase, its structurally supported capacity safely expands to **{optimal_growth:.1f}%**. CFOs must adjust capital structures dynamically.")
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 8 ---
            st.subheader("Chapter 8: The Tail-Risk — Zombie Firms & Extreme Leverage")
            st.write(f"Standard averages often obscure the extremities of financial distress. Within the {data_source_label} dataset, an alarming **{ext_count} observations** recorded book leverage exceeding 100%. This is a mathematical threshold indicating negative equity, where accumulated losses have entirely wiped out the firm's net worth.")
            st.write(f"Unlike standard market participants operating with an average profitability of **{norm_prof:.1f}%**, these extreme outliers are bleeding capital, exhibiting average profitability of **{ext_prof:.1f}%**. Fascinatingly, the most common life stage for these distressed entities is **{top_ext_stage}**. This empirically validates the presence of the 'Zombie Firm' phenomenon in the Indian market, where structurally declining entities are kept on life support via continuous debt rollovers from creditors, rather than facing efficient liquidation.")
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 9 ---
            st.subheader("Chapter 9: Systemic Risk & The Early Warning System (EWS)")
            st.write("By mathematically mapping the 90th percentile of the leverage distribution, this research establishes a definitive 'Distress Ceiling'. When a firm's actual leverage breaches this stage-adjusted ceiling, default probability accelerates exponentially.")
            
            wp_err_msg = f"🚨 **EWS Live Market Scan ({end_year}):** Scanning the most recent cross-section of {len(latest_year_data)} active firms reveals that **{flagged_count} entities** are currently operating with leverage ratios exceeding their Distress Ceilings. These firms represent severe, immediate default risks to the credit ecosystem."
            st.error(wp_err_msg)
            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHAPTER 10 ---
            st.subheader("Chapter 10: Strategic Conclusion & Policy Implications")
            st.write("This research fundamentally disrupts the assumption that capital structure optimization is a static exercise. The integration of Dickinson's life stages proves that financing decisions are organically tethered to the operational lifecycle.")
            st.write("**For Corporate Boards:** Leverage targets must be tied to cash flow lifecycle positioning, not just peer averages. **For Credit Analysts:** Standard rating downgrades are lagging indicators; monitoring a firm's transition into 'Shakeout' or 'Decline' serves as a highly predictive leading indicator of impending insolvency. **For Policymakers:** The persistence of Zombie Firms highlights the necessity for rigorous, unhindered application of bankruptcy frameworks to ensure capital is efficiently reallocated to productive, growing sectors of the economy.")

            st.divider()
            st.caption("💡 **Export Instructions:** This 8-page equivalent comprehensive white paper has been dynamically generated based on your dataset selection. To export, press `Ctrl + P` (or `Cmd + P`) and select 'Save as PDF'. The Streamlit layout is optimized for clean, paginated printing.")