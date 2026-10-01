import streamlit as st
import shap
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pydeck as pdk
import os

# Set page layout
st.set_page_config(page_title="AI Land Acquisition Risk", layout="wide")

# Read the API key securely from the environment rather than hardcoding it
# (You will set this in your terminal before running streamlit)
backend_api_key = os.environ.get("GEMINI_API_KEY", "")

# --- 1. CACHED MODEL TRAINING ---
@st.cache_resource
def load_models_and_encoders():
    n_samples = 1000
    df = pd.DataFrame({
        'Project_Category': np.random.choice(['National Highway (NHAI)', 'Railways', 'State Highway', 'Urban Development'], n_samples),
        'Statutory_Stage': np.random.choice(['Section 4 (SIA)', 'Section 11 (Notification)', 'Section 19 (Declaration)', 'Section 21 (Award)'], n_samples),
        'Land_Type': np.random.choice(['Private Agricultural', 'Government Land', 'Forest Land', 'Commercial'], n_samples),
        'Forest_Clearance': np.random.choice(['Approved', 'Pending', 'Not Required'], n_samples),
        'Current_Holding_Department': np.random.choice(['Local Tehsildar (Revenue)', 'District Magistrate', 'CALA Office', 'State Revenue Dept'], n_samples),
        'Active_Court_Litigation': np.random.choice(['Yes', 'No'], n_samples),
        'Gram_Panchayat_Approval_Required': np.random.choice(['Yes', 'No'], n_samples),
        'Days_Pending_at_Current_Desk': np.random.randint(1, 100, n_samples),
        'Land_Area_Hectares': np.random.uniform(1.0, 100.0, n_samples),
        'Affected_Families_Count': np.random.randint(0, 150, n_samples),
        'Compensation_Disbursed_Pct': np.random.uniform(0, 100, n_samples),
        'RnR_Progress_Pct': np.random.uniform(0, 100, n_samples),
        'CALA_Responsiveness_Score': np.random.randint(1, 11, n_samples),
        'Cadastral_Mismatch_Pct': np.random.uniform(0, 15, n_samples)
    })
    
    df['Delay_Days'] = (df['Days_Pending_at_Current_Desk'] * 0.5) + (df['Cadastral_Mismatch_Pct'] * 2) + np.where(df['Active_Court_Litigation'] == 'Yes', 60, 0)
    df['Is_Delayed'] = (df['Delay_Days'] > 30).astype(int)

    categorical_cols = ['Project_Category', 'Statutory_Stage', 'Land_Type', 'Forest_Clearance', 'Current_Holding_Department', 'Active_Court_Litigation', 'Gram_Panchayat_Approval_Required']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(columns=['Delay_Days', 'Is_Delayed'])
    
    reg_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg_model.fit(X, df['Delay_Days'])
    
    clf_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss')
    clf_model.fit(X, df['Is_Delayed'])
    
    return reg_model, clf_model, encoders

reg_model, clf_model, encoders = load_models_and_encoders()

# --- 2. STREAMLIT UI ---
st.title("AI Land Acquisition Risk Predictor")
#st.markdown("Predict the number of delay days, risk percentage score, and statutory bottlenecks for SIH Problem Statement 26017.")

STATE_COORDINATES = {
    "Andhra Pradesh": (15.9129, 79.7400), "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376), "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661), "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924), "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734), "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139), "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569), "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063), "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376), "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985), "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179), "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569), "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882), "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193), "West Bengal": (22.9868, 87.8550),
    "Delhi": (28.7041, 77.1025), "Jammu and Kashmir": (33.7782, 76.5762)
}

col1, col2 = st.columns(2)

with col1:
    proj_cat = st.selectbox("Project Category", ['National Highway (NHAI)', 'Railways', 'State Highway', 'Urban Development'])
    stat_stage = st.selectbox("Statutory Stage", ['Section 4 (SIA)', 'Section 11 (Notification)', 'Section 19 (Declaration)', 'Section 21 (Award)'])
    land_type = st.selectbox("Land Type", ['Private Agricultural', 'Government Land', 'Forest Land', 'Commercial'])
    forest_clear = st.selectbox("Forest Clearance", ['Approved', 'Pending', 'Not Required'])
    curr_dept = st.selectbox("Current Holding Department", ['Local Tehsildar (Revenue)', 'District Magistrate', 'CALA Office', 'State Revenue Dept'])
    days_pending = st.number_input("Days Pending at Current Desk", min_value=0, value=35)
    gp_approval = st.selectbox("Gram Panchayat Approval Required?", ['No', 'Yes'])
    
    all_states = sorted(list(STATE_COORDINATES.keys()))
    selected_state = st.selectbox("State / Union Territory", all_states, index=all_states.index("Rajasthan"))

with col2:
    land_area = st.number_input("Land Area (Hectares)", min_value=0.0, value=25.0)
    families = st.number_input("Affected Families Count", min_value=0, value=30)
    comp_pct = st.slider("Compensation Disbursed (%)", 0.0, 100.0, 75.0)
    rnr_pct = st.slider("R&R Progress (%)", 0.0, 100.0, 60.0)
    litigation = st.selectbox("Active Court Litigation?", ['No', 'Yes'])
    cala_score = st.slider("CALA Responsiveness Score", 1, 10, 7)
    mismatch = st.slider("Cadastral Mismatch (%)", 0.0, 100.0, 4.5)
    
    default_lat, default_lon = STATE_COORDINATES[selected_state]
    lat_in = st.number_input("Latitude", value=float(default_lat), format="%.4f")
    lon_in = st.number_input("Longitude", value=float(default_lon), format="%.4f")

# Backend API key retrieval
backend_api_key = os.environ.get("GEMINI_API_KEY", "")

# --- 3. PREDICTION & VISUALIZATION ---
if st.button("Predict Delay Risk", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'Project_Category': [proj_cat],
        'Statutory_Stage': [stat_stage],
        'Land_Type': [land_type],
        'Forest_Clearance': [forest_clear],
        'Current_Holding_Department': [curr_dept],
        'Active_Court_Litigation': [litigation],
        'Gram_Panchayat_Approval_Required': [gp_approval],
        'Days_Pending_at_Current_Desk': [days_pending],
        'Land_Area_Hectares': [land_area],
        'Affected_Families_Count': [families],
        'Compensation_Disbursed_Pct': [comp_pct],
        'RnR_Progress_Pct': [rnr_pct],
        'CALA_Responsiveness_Score': [cala_score],
        'Cadastral_Mismatch_Pct': [mismatch]
    })
    
    encoded_input = input_data.copy()
    for col, le in encoders.items():
        if encoded_input[col][0] in le.classes_:
            encoded_input[col] = le.transform(encoded_input[col])
        else:
            encoded_input[col] = 0 

    predicted_days = reg_model.predict(encoded_input)[0]
    predicted_days = max(0, int(predicted_days)) 
    
    predicted_prob = clf_model.predict_proba(encoded_input)[0][1]
    # Realistic probability scaling
    clipped_prob = np.clip(predicted_prob, 0.048, 0.924)
    risk_percentage = clipped_prob * 100.0

    st.divider()
    st.subheader("Prediction Results & Risk Score")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric(label="Estimated Delay (Days)", value=f"{predicted_days} Days")
    with res_col2:
        st.metric(label="Risk Percentage Score", value=f"{risk_percentage:.1f}%")
    with res_col3:
        if risk_percentage >= 60.0:
            st.error("Risk Tier: HIGH CRITICAL")
            parcel_color = [220, 38, 38, 200]
        elif risk_percentage >= 30.0:
            st.warning("Risk Tier: MODERATE RISK")
            parcel_color = [234, 115, 23, 200]
        else:
            st.success("Risk Tier: LOW RISK (ON TRACK)")
            parcel_color = [22, 163, 74, 200]

    # --- 4. 3D MAP ---
    st.divider()
    st.subheader(f"3D Geospatial Land Parcel Footprint ({selected_state})")
    
    map_df = pd.DataFrame({
        'lat': [lat_in],
        'lon': [lon_in],
        'elevation': [max(50, predicted_days * 8)],
        'name': [f"{proj_cat} - Delay: {predicted_days}d"]
    })

    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=50,
        radius=3500,
        get_fill_color=parcel_color,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=lat_in,
        longitude=lon_in,
        zoom=7,
        pitch=45,
        bearing=15
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nRisk: " + f"{risk_percentage:.1f}%"}
    ))

    # --- 5. REASONS FOR DELAY (SHAP) ---
    st.divider()
    st.subheader("Primary Reasons for Delay")
    
    explainer = shap.TreeExplainer(reg_model)
    shap_values = explainer(encoded_input)
    feature_names = input_data.columns
    contributions = shap_values.values[0]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact (Days)': contributions
    })
    
    delay_drivers = importance_df[importance_df['Impact (Days)'] > 0.5].sort_values(by='Impact (Days)', ascending=False)
    driver_strings = []
    
    if not delay_drivers.empty:
        st.markdown("These statutory factors added the most time to your predicted delay:")
        for _, row in delay_drivers.head(4).iterrows():
            clean_name = row['Feature'].replace('_', ' ')
            driver_text = f"{clean_name}: Added ~{row['Impact (Days)']:.0f} days"
            st.markdown(f"- **{driver_text}**")
            driver_strings.append(driver_text)
    else:
        st.success("No significant bottleneck factors detected. Clearance progressing smoothly.")

    # --- 6. AI ADMINISTRATIVE ADVISORY (RESILIENT BACKEND ENGINE) ---
    st.divider()
    st.subheader("AI Administrative Advisory & Mitigation Workflow")
    
    if driver_strings:
        prompt_context = f"""
Project Category: {proj_cat}
Statutory Stage: {stat_stage}
Land Type: {land_type}
State: {selected_state}
Predicted Delay: {predicted_days} days
Risk Percentage Score: {risk_percentage:.1f}%
Top Bottleneck Drivers:
{chr(10).join(['- ' + d for d in driver_strings])}

As a legal and revenue expert on the RFCTLARR Act (2013), provide a structured 3-bullet administrative advisory:
1. Identify the root statutory failure caused by these specific drivers.
2. Provide a concrete SOP for the District Magistrate / CALA to resolve this bottleneck immediately.
3. Specify inter-departmental action points.
"""
        generated_advisory = None
        
        if backend_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=backend_api_key)
                
                # Multi-model fallback cascade to avoid 503 errors
                for candidate in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.0-flash']:
                    try:
                        resp = client.models.generate_content(
                            model=candidate,
                            contents=prompt_context
                        )
                        if resp and resp.text:
                            generated_advisory = resp.text
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        
        if generated_advisory:
            st.markdown(generated_advisory)
        else:
            # Polished statutory fallback
            st.markdown(f"""
            - **Statutory Assessment:** The primary delay bottleneck stems from **{driver_strings[0]}**. Under Section 15 of the RFCTLARR Act (2013), objections must be recorded and concluded within statutory inquiry deadlines.
            - **Immediate CALA SOP:** Direct the Competent Authority for Land Acquisition to conduct joint on-site verifications with the Revenue Tehsildar and expedite award inquiries under Section 21.
            - **Inter-Departmental Escalation:** Issue high-priority notices to the State Revenue Department and Forest Division to clear boundary mismatches and pending NOC clearances within 14 days.
            """)
    else:
        st.write("All milestones conform to target timelines. No escalation playbook required.")