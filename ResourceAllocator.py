import streamlit as st
import pandas as pd
import altair as alt
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum, LpBinary, PULP_CBC_CMD

st.set_page_config(page_title="Resource Allocator", page_icon=":crossed_swords:", layout="wide")
st.title("Resource Allocator")

#Load Excel data
uploaded_file = st.file_uploader("Upload Excel file with input data", type=["xlsx", "xls"])
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    try:
        requirements_df = pd.read_excel(xls, sheet_name="Requirements")
        current_df = pd.read_excel(xls, sheet_name="Current")
        arrivals_df = pd.read_excel(xls, sheet_name="Arrivals")
        params_df = pd.read_excel(xls, sheet_name="Optimization Parameters")
    except ValueError as e:
        st.error(f"Error reading Excel file: {e}")

# Stop here until data is loaded
if 'requirements_df' not in locals() or 'current_df' not in locals() or 'arrivals_df' not in locals() or 'params_df' not in locals():
    st.stop()

# Data processing
def create_stocks_df(requirements_df, current_df, equipment_categories):
    # Merge requirements and current data into a stocks dataframe
    stocks = []
    merged_df = pd.merge(requirements_df, current_df, on="unit_id", suffixes=("_req", "_cur"))
    for _, row in merged_df.iterrows():
        for eq in equipment_categories:
            req = row[f"{eq}_req"]
            cur = row[f"{eq}_cur"]
            if pd.notna(req) and req > 0:
                unit_id = row["unit_id"]
                brigade = row["Brigade_req"]
                designation = row["Designation_req"]
                unit = row["Unit_req"]
                des_unit = f"{designation} {unit}" if pd.notna(designation) else unit
                fulfilment = cur / req if pd.notna(cur) else 0
                stocks.append({
                    "unit_id": unit_id,
                    "Brigade": brigade,
                    "Unit": des_unit,
                    "unique_name": f"{str(unit_id).zfill(3)} - {brigade} - {des_unit}",
                    "unique_unit_name": f"{str(unit_id).zfill(3)} - {des_unit}",
                    "Equipment": eq,
                    "Requirement": req,
                    "Current": cur if pd.notna(cur) else 0,
                    "Shortfall": req - (cur if pd.notna(cur) else 0),
                    "Fulfilment": fulfilment,
                    "FulfilmentPct": f"{fulfilment:.0%}",
                    "FulfilmentColor": 'green'
                        if fulfilment >= 0.8
                        else ('orange' if fulfilment >= 0.6 else 'red')
                })
    return pd.DataFrame(stocks)

def create_brigade_df(df):
    # Create dataframe with brigade fulfilment
    if 'Allocation' not in df.columns:
        df = df.copy()
        df['Allocation'] = 0
    brigade_df = (
        df
        .groupby(['Brigade', 'Equipment'])[['Requirement', 'Current', 'Shortfall', 'Allocation']]
        .sum()
        .reset_index()
    )
    brigade_df['Fulfilment'] = (
        (brigade_df['Current'] + brigade_df['Allocation']) / brigade_df['Requirement']
    )
    brigade_df['FulfilmentPct'] = (
        brigade_df['Fulfilment'].apply(lambda x: f"{x:.0%}")
    )
    brigade_df['FulfilmentColor'] = brigade_df['Fulfilment'].apply(
        lambda x: 'green' if x >= 0.8 else ('orange' if x >= 0.6 else 'red')
    )
    return brigade_df

def create_unit_and_brigade_metrics(df):
    # Calculate capability and fulfilment metrics
    unit_metrics = {
        j: {
            "Brigade": df[df["unit_id"] == j]["Brigade"].iloc[0],
            "Unit": df[df["unit_id"] == j]["Unit"].iloc[0],
            "unique_unit_name": df[df["unit_id"] == j]["unique_unit_name"].iloc[0],
            "num_equipment_categories": sum(
                r.Requirement > 0
                for r in df[df["unit_id"] == j].itertuples()
            ),
            "y20sum": sum(
                r.Fulfilment < 0.8
                for r in df[df["unit_id"] == j].itertuples()
            ),
            "y40sum": sum(
                r.Fulfilment < 0.6
                for r in df[df["unit_id"] == j].itertuples()
            )
        } for j in df['unit_id'].unique()
    }
    brigade_metrics = {
        i: {
            "num_units": len(df[df["Brigade"] == i]["unit_id"].unique()),
            "num_units_y20": len([
                j for j in df[df["Brigade"] == i]["unit_id"].unique()
                if unit_metrics[j]["y20sum"] > 0
            ]),
            "num_units_y40": len([
                j for j in df[df["Brigade"] == i]["unit_id"].unique()
                if unit_metrics[j]["y40sum"] > 0
            ])
        } for i in df['Brigade'].unique()
    }
    return unit_metrics, brigade_metrics

# Bar charts
def create_fulfilment_bar_chart(df, y_value, x_label, y_label, title, for_brigade=False, with_allocation=False):
    # Create bar chart to visualize fulfilment
    tooltip_vars = ["Brigade", "Equipment"] if for_brigade else ["unit_id", "Brigade", "Unit", "Equipment"]
    value_vars = ["Current", "Allocation", "Shortfall"] if with_allocation else ["Current", "Shortfall"]
    df_melted = df.melt(
        id_vars = list(dict.fromkeys([y_value] + tooltip_vars + ["Requirement", "FulfilmentPct", "FulfilmentColor"])),
        value_vars=value_vars,
        var_name="Status",
        value_name="Quantity"
    )
    max_stack = df[value_vars].sum(axis=1).max()
    bars = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X(
            'Quantity:Q',
            title=x_label,
            scale=alt.Scale(domain=[0, max_stack * 1.01]),
            axis=alt.Axis(format='d')
        ),
        y=alt.Y(f"{y_value}:N", title=y_label),
        color=alt.Color('Status:N',
                        title='Equipment status',
                        scale=alt.Scale(domain=['Current', 'Allocation', 'Shortfall'], range=['#1B3C53', '#9ACBD0', '#D2C1B6']) if with_allocation
                            else alt.Scale(domain=['Current', 'Shortfall'], range=['#1B3C53', '#D2C1B6'])
        ),
        tooltip=tooltip_vars + ['Status:N', 'Quantity:Q'],
        order=alt.Order('color_Status_sort_index:Q')
    )
    text = alt.Chart(df_melted).mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        x=alt.X('Requirement:Q'),
        y=alt.Y(f"{y_value}:N"),
        text=alt.Text('FulfilmentPct:N'),
        color=alt.Color('FulfilmentColor:N', scale=None),
        tooltip=tooltip_vars + ['Requirement:Q', 'FulfilmentPct:N']
    )
    chart = (bars + text).properties(
        title={
            'text': title,
            'subtitle': f"Average fulfilment: {df['Fulfilment'].mean():.0%}",
            'anchor':'middle'
        }
    ).configure_axisY(
        labelBaseline='middle',
        labelLimit=150
    )
    return chart

def create_classification_bar_chart(df, y_value, x_label, y_label, title, for_brigade=False):
    # Create bar chart to visualize capability classification metrics
    tooltip_vars = ["Brigade"] if for_brigade else ["unit_id", "Brigade", "Unit"]
    df_melted = df.melt(
        id_vars=list(dict.fromkeys([y_value] + tooltip_vars)),
        value_vars=['Capable', 'Partially capable', 'Not capable'] if y_value == 'Brigade'
            else ['Above 80%', 'Between 60% and 80%', 'Below 60%'],
        var_name='Classification',
        value_name='Quantity'
    )
    bars = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Quantity:Q', title=x_label, axis=alt.Axis(format='d')),
        y=alt.Y(f"{y_value}:N", title=y_label),
        color=alt.Color('Classification:N',
                        title='Unit classification' if y_value == 'Brigade' else 'Equipment category fulfilment',
                        scale=alt.Scale(domain=['Capable', 'Partially capable', 'Not capable'], range=['green', 'orange', 'red']) if y_value == 'Brigade'
                            else alt.Scale(domain=['Above 80%', 'Between 60% and 80%', 'Below 60%'], range=['green', 'orange', 'red'])
        ),
        tooltip=tooltip_vars + ['Classification:N', 'Quantity:Q'],
        order=alt.Order('color_Classification_sort_index:Q')
    )
    text = alt.Chart(df_melted).mark_text().encode(
        color=alt.Color('Classification:N', scale=None),
    )
    chart = (bars + text).properties(
        title={
            'text': title,
            'anchor': 'middle'
        }
    ).configure_axisY(
        labelBaseline='middle',
        labelLimit=150
    )
    return chart

# Optimization parameters
def optimization_params_input(params_dict, param_fields, param_name, description, param_subscripts=None, min_value=0):
    # Create input fields for optimization parameters
    col1, col2, col3 = st.columns(3)
    if param_subscripts is None:
        param_subscripts = [""]
    for index, s in enumerate(param_subscripts):
        col = [col1, col2, col3][index % 3]
        key = f"{param_name}_{s}" if s else param_name
        value = params_dict.get(key, min_value)
        if value < min_value:
            st.warning(f"Invalid Excel input: Value for *{key}* cannot be less than {min_value}.")
            value = min_value
        param_fields[key] = col.number_input(
            f"**{key}**",
            min_value=min_value,
            step=1,
            value=value
        )
    st.caption(description)

# Optimization algorithm
@st.cache_data(show_spinner=False)
def run_optimization(stocks_df, arrivals, optimization_params):
    # Create LP model
    model = LpProblem("Resource_Allocation", LpMinimize)
    # Filter to equipment with arrival data
    df = stocks_df[stocks_df["Equipment"].isin(arrivals.keys())].copy()
    # Variables
    x_vars = {
        (r.unit_id, r.Equipment): LpVariable(f"x_{r.unit_id}_{r.Equipment}", lowBound=0, cat=LpInteger)
        for r in df.itertuples()
    }
    units = df["unit_id"].unique()
    s_x = {
        j: LpVariable(f"s_x_{j}", lowBound=0)
        for j in units
    }
    brigades = df["Brigade"].unique()
    b_x = {
        i: LpVariable(f"b_x_{i}", lowBound=0)
        for i in brigades
    }
    b_x_diff = {
        (i, j): LpVariable(f"b_x_diff_{i}_{j}", lowBound=0)
        for j in brigades for i in brigades if j > i
    }
    y20_vars = {
        (r.unit_id, r.Equipment): LpVariable(f"y20_{r.unit_id}_{r.Equipment}", cat=LpBinary)
        for r in df.itertuples()
    }
    y40_vars = {
        (r.unit_id, r.Equipment): LpVariable(f"y40_{r.unit_id}_{r.Equipment}", cat=LpBinary)
        for r in df.itertuples()
    }
    z20_vars = {
        i: LpVariable(f"z20_{i}", cat=LpBinary)
        for i in brigades
    }
    z40_vars = {
        i: LpVariable(f"z40_{i}", cat=LpBinary)
        for i in brigades
    }
    # Weights
    max_req = stocks_df.groupby("Equipment")["Requirement"].max().replace(0, 1)
    equipment_categories = stocks_df["Equipment"].unique().tolist()
    a = {k: 1.0 / max_req[k] for k in equipment_categories}
    b = {j: 1.0 for j in units} # Alter this value to add weights for units
    c = {i: optimization_params.get(f"c_{i}", 1) for i in brigades}
    d = optimization_params.get("d", 0)
    P_20 = optimization_params.get("P_20", 0)
    P_40 = optimization_params.get("P_40", 0)
    # Objective: Minimize total weighted gap
    model += (
        lpSum([c[i] * b_x[i] for i in brigades]) +
        d * lpSum([b_x_diff[(i, j)] for j in brigades for i in brigades if j > i]) +
        P_20 * lpSum([z20_vars[i] for i in brigades]) +
        P_40 * lpSum([z40_vars[i] for i in brigades])
    )
    # Constraints
    for r in df.itertuples():
        x = x_vars[(r.unit_id, r.Equipment)]
        req = r.Requirement
        cur = r.Current
        model += x + cur <= req
        model += (req - (cur + x)) <= 0.2 * req + req * y20_vars[(r.unit_id, r.Equipment)]
        model += (req - (cur + x)) >= 0.2 * req - req * (1 - y20_vars[(r.unit_id, r.Equipment)])
        model += (req - (cur + x)) <= 0.4 * req + req * y40_vars[(r.unit_id, r.Equipment)]
        model += (req - (cur + x)) >= 0.4 * req - req * (1 - y40_vars[(r.unit_id, r.Equipment)])
        model += y40_vars[(r.unit_id, r.Equipment)] <= y20_vars[(r.unit_id, r.Equipment)]
    for j in units:
        model += s_x[j] == lpSum([
            a[r.Equipment] * (r.Requirement - (r.Current + x_vars[(r.unit_id, r.Equipment)]))
            for r in df[df["unit_id"] == j].itertuples()
        ])
    for i in brigades:
        model += b_x[i] == lpSum([
            b[j] * s_x[j]
            for j in df[df["Brigade"] == i]["unit_id"].unique()
        ])
        for j in brigades:
            if j > i:
                model += b_x_diff[(i, j)] >= b_x[i] - b_x[j]
                model += b_x_diff[(i, j)] >= b_x[j] - b_x[i]
        model += z20_vars[i] <= lpSum([
            y20_vars[(r.unit_id, r.Equipment)]
            for r in df[df["Brigade"] == i].itertuples()
        ])
        model += z40_vars[i] <= lpSum([
            y40_vars[(r.unit_id, r.Equipment)]
            for r in df[df["Brigade"] == i].itertuples()
        ])
        for r in df[df["Brigade"] == i].itertuples():
            model += z20_vars[i] >= y20_vars[(r.unit_id, r.Equipment)]
            model += z40_vars[i] >= y40_vars[(r.unit_id, r.Equipment)]
    for eq in arrivals:
        model += lpSum([x_vars[(r.unit_id, r.Equipment)] for r in df.itertuples() if r.Equipment == eq]) <= arrivals[eq]
    # Solve
    solver = PULP_CBC_CMD(timeLimit=optimization_params.get("timeLimit", 60))
    model.solve(solver)
    # Process results
    df["Allocation"] = df.apply(lambda r: x_vars[(r.unit_id, r.Equipment)].varValue, axis=1)
    df["Shortfall"] = df["Requirement"] - (df["Current"] + df["Allocation"])
    df["Fulfilment"] = (df["Current"] + df["Allocation"]) / df["Requirement"]
    df["FulfilmentPct"] = df["Fulfilment"].apply(lambda x: f"{x:.0%}")
    df["FulfilmentColor"] = df["Fulfilment"].apply(lambda x: 'green' if x >= 0.8 else ('orange' if x >= 0.6 else 'red'))
    unit_metrics = {
        j: {
            "Brigade": df[df["unit_id"] == j]["Brigade"].iloc[0],
            "Unit": df[df["unit_id"] == j]["Unit"].iloc[0],
            "unique_unit_name": df[df["unit_id"] == j]["unique_unit_name"].iloc[0],
            "num_equipment_categories": sum(
                r.Requirement > 0
                for r in df[df["unit_id"] == j].itertuples()
            ),
            "y20sum": sum(
                y20_vars[(r.unit_id, r.Equipment)].varValue
                for r in df[df["unit_id"] == j].itertuples()
            ),
            "y40sum": sum(
                y40_vars[(r.unit_id, r.Equipment)].varValue
                for r in df[df["unit_id"] == j].itertuples()
            )
        } for j in units
    }
    brigade_metrics = {
        i: {
            "num_units": len(df[df["Brigade"] == i]["unit_id"].unique()),
            "num_units_y20": len([
                j for j in df[df["Brigade"] == i]["unit_id"].unique()
                if unit_metrics[j]["y20sum"] > 0
            ]),
            "num_units_y40": len([
                j for j in df[df["Brigade"] == i]["unit_id"].unique()
                if unit_metrics[j]["y40sum"] > 0
            ])
        } for i in brigades
    }
    optimization_values = {
        "Objective value": model.objective.value(),
    }
    for i in brigades:
        optimization_values[f"b_x_{i}"] = b_x[i].varValue
        optimization_values[f"z20_{i}"] = z20_vars[i].varValue
        optimization_values[f"z40_{i}"] = z40_vars[i].varValue
    print("Objective value:", model.objective.value())
    return df, unit_metrics, brigade_metrics, optimization_values

# UI elements
def add_checkbox(value, key):
    # Checkbox used to keep expander open
    st.checkbox(
        ":gray[Keep open]",
        value=value,
        key=key,
        help="Keep this section open whenever the app recalculates or updates"
    )

# Visualization functions
def show_brigade_capability_classification(brigade_metrics, with_allocation=False):
    # Create a bar chart to visualize brigade capability classification
    df = pd.DataFrame({
        "Brigade": list(brigade_metrics.keys()),
        "Capable": [
            brigade_metrics[i]['num_units'] - brigade_metrics[i]['num_units_y20']
            for i in brigade_metrics
        ],
        "Partially capable": [
            brigade_metrics[i]['num_units_y20'] - brigade_metrics[i]['num_units_y40']
            for i in brigade_metrics
        ],
        "Not capable": [
            brigade_metrics[i]['num_units_y40']
            for i in brigade_metrics
        ]
    })
    chart = create_classification_bar_chart(
        df=df,
        y_value="Brigade",
        x_label="# of units",
        y_label="Brigade",
        title=f"Capability classification{' after allocation' if with_allocation else ''}",
        for_brigade=True
    )
    expander_label = f"View {'current ' if not with_allocation else ''}brigade capability classification"
    keep_open = st.session_state.get(expander_label, False)
    with st.expander(expander_label, expanded=keep_open):
        add_checkbox(keep_open, expander_label)
        st.altair_chart(chart)
        st.caption("""
            Units are classified as capable if their fulfilment exceeds 80% in all equipment categories,
            partially capable if their fulfilment exceeds 60% in all categories,
            and not capable if their fulfilment falls below 60% in any category.
        """)

def show_unit_fulfilment_classification(unit_metrics, brigade, with_allocation=False):
    # Create a bar chart to visualize unit fulfilment classification
    unit_ids = [j for j in unit_metrics if unit_metrics[j]['Brigade'] == brigade]
    df = pd.DataFrame({
        "unit_id": unit_ids,
        "Brigade": brigade,
        "Unit": [unit_metrics[j]['Unit'] for j in unit_ids],
        "unique_unit_name": [unit_metrics[j]['unique_unit_name'] for j in unit_ids],
        "Above 80%": [
            unit_metrics[j]['num_equipment_categories'] - unit_metrics[j]['y20sum']
            for j in unit_ids
        ],
        "Between 60% and 80%": [
            unit_metrics[j]['y20sum'] - unit_metrics[j]['y40sum']
            for j in unit_ids
        ],
        "Below 60%": [
            unit_metrics[j]['y40sum']
            for j in unit_ids
        ]
    })
    chart = create_classification_bar_chart(
        df=df,
        y_value="unique_unit_name",
        x_label="# of equipment categories",
        y_label="Unit",
        title=f"Fulfilment classification in {brigade}{' after allocation' if with_allocation else ''}"
    )
    expander_label = f"View {'current ' if not with_allocation else ''}unit fulfilment classification in :blue-badge[selected brigade]"
    keep_open = st.session_state.get(expander_label, False)
    with st.expander(expander_label, expanded=keep_open):
        add_checkbox(keep_open, expander_label)
        st.altair_chart(chart)

def show_equipment_fulfilment_in_brigade(df, brigade, with_allocation=False):
    # Create a bar chart to visualize equipment fulfilment in brigade
    brigade_df = create_brigade_df(df[df['Brigade'] == brigade])
    chart = create_fulfilment_bar_chart(
        df=brigade_df,
        y_value="Equipment",
        x_label=f"# of equipment",
        y_label='Equipment category',
        title=f"Fulfilment in {brigade}{' after allocation' if with_allocation else ''}",
        for_brigade=True,
        with_allocation=with_allocation
    )
    expander_label = f"View {'current ' if not with_allocation else ''}equipment category fulfilment in :blue-badge[selected brigade]"
    keep_open = st.session_state.get(expander_label, False)
    with st.expander(expander_label, expanded=keep_open):
        add_checkbox(keep_open, expander_label)
        st.altair_chart(chart)

def show_equipment_fulfilment_in_unit(df, unit_id, with_allocation=False):
    # Create a bar chart to visualize equipment fulfilment in unit
    unit_df = df[df['unit_id'] == unit_id]
    chart = create_fulfilment_bar_chart(
        df=unit_df,
        y_value="Equipment",
        x_label=f"# of equipment",
        y_label='Equipment category',
        title=f"Fulfilment in {unit_df['Unit'].iloc[0]}{' after allocation' if with_allocation else ''}",
        with_allocation=with_allocation
    )
    expander_label = f"View {'current ' if not with_allocation else ''}equipment category fulfilment in :green-badge[selected unit]"
    keep_open = st.session_state.get(expander_label, False)
    with st.expander(expander_label, expanded=keep_open):
        add_checkbox(keep_open, expander_label)
        st.altair_chart(chart)

def show_brigade_fulfilment_in_equipment_category(df, equipment, with_allocation=False):
    # Create a bar chart to visualize brigade fulfilment in equipment category
    equipment_by_brigade_df = create_brigade_df(df[df['Equipment'] == equipment])
    expander_label = f"View {'current ' if not with_allocation else ''}brigade fulfilment of :violet-badge[selected equipment category]"
    keep_open = st.session_state.get(expander_label, False)
    with st.expander(expander_label, expanded=keep_open):
        add_checkbox(keep_open, expander_label)
        if len(equipment_by_brigade_df) == 0:
            st.warning(f"No data available for {equipment}")
            return
        chart = create_fulfilment_bar_chart(
            df=equipment_by_brigade_df,
            y_value="Brigade",
            x_label=f"# of {equipment}",
            y_label='Brigade',
            title=f"Fulfilment of {equipment}{' after allocation' if with_allocation else ''}",
            for_brigade=True,
            with_allocation=with_allocation
        )
        st.altair_chart(chart)

def show_unit_fulfilment_in_equipment_category(df, equipment, unit_ids, brigade, with_allocation=False):
    # Create a bar chart to visualize unit fulfilment in equipment category
    equipment_by_unit_df = df[(df['Equipment'] == equipment) & (df['unit_id'].isin(unit_ids))]
    expander_label = f"View {'current ' if not with_allocation else ''}unit fulfilment of :violet-badge[selected equipment category] in :blue-badge[selected brigade]"
    keep_open = st.session_state.get(expander_label, False)
    with st.expander(expander_label, expanded=keep_open):
        add_checkbox(keep_open, expander_label)
        if len(equipment_by_unit_df) == 0:
            st.warning(f"No data available for {equipment} in {brigade}")
            return
        chart = create_fulfilment_bar_chart(
            df=equipment_by_unit_df,
            y_value="unique_unit_name",
            x_label=f"# of {equipment}",
            y_label='Unit',
            title=f"Fulfilment of {equipment} in {brigade}{' after allocation' if with_allocation else ''}",
            with_allocation=with_allocation
        )
        st.altair_chart(chart)

# Process input data
equipment_categories = sorted(requirements_df.columns[5:].tolist())
stocks_df = create_stocks_df(requirements_df, current_df, equipment_categories)
brigades = sorted(stocks_df['Brigade'].unique().tolist())
unit_metrics, brigade_metrics  = create_unit_and_brigade_metrics(stocks_df)

# Display input data
if not stocks_df.empty:
    # Sidebar for selection of brigade, unit, and equipment category
    with st.sidebar:
        st.header("Visualization options")
        selected_brigade = st.selectbox(':blue-badge[Selected brigade]', options=brigades)
        unit_options = stocks_df[stocks_df['Brigade'] == selected_brigade][['unit_id', 'Unit', 'unique_unit_name']].drop_duplicates().set_index('unit_id')
        selected_unit_id = st.selectbox(
            ':green-badge[Selected unit]',
            options=unit_options.index,
            format_func=lambda uid: unit_options.loc[uid, 'unique_unit_name']
        )
        selected_equipment = st.selectbox(':violet-badge[Selected equipment category]', options=equipment_categories)

    st.header('Current Fulfilment')
    # Visualize brigade capability classification
    show_brigade_capability_classification(brigade_metrics)

    # Visualize unit fulfilment classification
    show_unit_fulfilment_classification(unit_metrics, selected_brigade)

    # Visualize equipment category fulfilment in brigade
    show_equipment_fulfilment_in_brigade(stocks_df, selected_brigade)

    # Visualize equipment category fulfilment in unit
    show_equipment_fulfilment_in_unit(stocks_df, selected_unit_id)

    # Visualize brigade fulfilment in equipment category
    show_brigade_fulfilment_in_equipment_category(stocks_df, selected_equipment)

    # Visualize unit fulfilment in equipment category
    show_unit_fulfilment_in_equipment_category(stocks_df, selected_equipment, unit_options.index, selected_brigade)

    # Define arrival data
    st.header("Arrivals")
    keep_open_arrivals = st.session_state.get("arrivals", False)
    with st.expander("Specify the quantity of arriving equipment", expanded=keep_open_arrivals):
        add_checkbox(keep_open_arrivals, "arrivals")
        edited_arrivals_df = st.data_editor(arrivals_df, hide_index=True, width=500)
    arrivals_input = dict(zip(edited_arrivals_df["Equipment"], edited_arrivals_df["Arrivals"]))
    arrivals = {
        eq: max(int(arrivals_input.get(eq, 0)) if pd.notna(arrivals_input.get(eq, 0)) else 0, 0)
        for eq in equipment_categories
    }

    # Set optimization parameters
    st.header("Optimization Parameters")
    params_dict = dict(zip(params_df["Parameter"], params_df["Value"]))
    optimization_params = {}
    keep_open_params = st.session_state.get("params", False)
    with st.expander("Set optimization parameters", expanded=keep_open_params):
        add_checkbox(keep_open_params, "params")
        optimization_params_input(
            params_dict=params_dict,
            param_fields=optimization_params,
            param_name="c",
            param_subscripts=brigades,
            min_value=1,
            description="""
                Brigade weights.
                *c_i* can be used to adjust the priority of brigade *i*.
            """
        )
        optimization_params_input(
            params_dict=params_dict,
            param_fields=optimization_params,
            param_name="d",
            description="""
                Difference equalizer.
                *d* can be used to reduce differences between brigades and give priority to brigades with lower fulfilment.
            """
        )
        optimization_params_input(
            params_dict=params_dict,
            param_fields=optimization_params,
            param_name="P",
            param_subscripts=["20", "40"],
            description="""
                Punishment for brigades with more than 20% shortfall or 40% shortfall.
                *P_20* can be used to maximize the number of fully capable brigades.
                *P_40* can be used to minimize the number of non-capable brigades.
            """
        )
        st.markdown("**Objective function to be minimized**")
        st.latex(r"""
            f(x) = \sum_{i \in \mathcal{B}} c_i \cdot b_i(x) +
            d \cdot \sum_{\substack{i,j \in \mathcal{B}\\ j > i}} |b_i(x) - b_j(x)| +
            P_{20} \cdot \sum_{i \in \mathcal{B}} z_{20,i} +
            P_{40} \cdot \sum_{i \in \mathcal{B}} z_{40,i}
        """)
        st.caption(r"""
            where:
            - $x$ represent the quantity of equipment allocated to each unit,
            - $\mathcal{B}$ is the set of brigades,
            - $c_i$ is the weight for brigade $i$,
            - $b_i(x)$ is the sum of shortfalls for brigade $i$ given allocation $x$,
            - $d$ is the difference equalizer weight,
            - $P_{20}$ is the punishment weight for brigades with more than 20% shortfall,
            - $P_{40}$ is the punishment weight for brigades with more than 40% shortfall,
            - $z_{20,i}$ is a binary variable indicating if brigade $i$ has any unit with more than 20% shortfall,
            - $z_{40,i}$ is a binary variable indicating if brigade $i$ has any unit with more than 40% shortfall.
            """)
        colTimeLimit, _, _ = st.columns(3)
        optimization_params["timeLimit"] = colTimeLimit.number_input(
            "**Solver time limit (seconds)**",
            min_value=10,
            step=10,
            value=30
        )
        st.caption("Maximum time the solver will run before stopping. The best solution found within this limit will be returned.")

    # Run optimization
    keep_open_optimization = st.session_state.get("optimization", False)
    with st.status("Running optimization...", expanded=keep_open_optimization) as status:
        add_checkbox(keep_open_optimization, "optimization")
        with st.spinner(show_time=True):
            df, unit_metrics, brigade_metrics, optimization_values = run_optimization(stocks_df, arrivals, optimization_params)
        st.markdown(f"Objective value: $f(x) = {optimization_values['Objective value']}$")
        col1_header, col2_header, col3_header, col4_header = st.columns([4, 2, 1, 1])
        col1_header.caption("$i$")
        col2_header.caption("$b_i(x)$")
        col3_header.caption("$z_{20,i}$")
        col4_header.caption("$z_{40,i}$")
        for i in brigades:
            col1, col2, col3, col4 = st.columns([4, 2, 1, 1], vertical_alignment="center")
            col1.caption(f"{i}")
            col2.caption(f"${optimization_values.get(f'b_x_{i}', 0)}$")
            col3.caption(f"${int(optimization_values.get(f'z20_{i}', 0))}$")
            col4.caption(f"${int(optimization_values.get(f'z40_{i}', 0))}$")
        status.update(label="Optimization completed")

# Display output data
if not df.empty:
    st.header("Optimized Allocation")
    # Display allocation results
    keep_open_allocation = st.session_state.get("allocation", False)
    with st.expander("View the number of equipment allocated to each unit", expanded=keep_open_allocation):
        add_checkbox(keep_open_allocation, "allocation")
        allocation_pivot = df.pivot_table(
            index='unique_name',
            columns='Equipment',
            values='Allocation',
            aggfunc='sum',
            fill_value=0
        )
        allocation_pivot['TOTAL'] = allocation_pivot.sum(axis=1)
        allocation_pivot.loc['TOTAL'] = allocation_pivot.sum(axis=0)
        allocation_df = allocation_pivot.reset_index()
        allocation_df = allocation_df.rename(columns={'unique_name': 'Unit'})
        st.dataframe(allocation_df, hide_index=True)

    # Visualizae brigade capability classification
    show_brigade_capability_classification(brigade_metrics, with_allocation=True)

    # Visualize unit fulfilment classification
    show_unit_fulfilment_classification(unit_metrics, selected_brigade, with_allocation=True)

    # Visualize equipment category fulfilment in brigade
    show_equipment_fulfilment_in_brigade(df, selected_brigade, with_allocation=True)

    # Visualize equipment category fulfilment in unit
    show_equipment_fulfilment_in_unit(df, selected_unit_id, with_allocation=True)

    # Visualize brigade fulfilment in equipment category
    show_brigade_fulfilment_in_equipment_category(df, selected_equipment, with_allocation=True)
        
    # Visualize unit fulfilment in equipment category
    show_unit_fulfilment_in_equipment_category(df, selected_equipment, unit_options.index, selected_brigade, with_allocation=True)

    # Display surplus equipment
    surplus = []
    for eq in arrivals:
        total_allocated = df[df['Equipment'] == eq]['Allocation'].sum()
        surplus.append({
            "Equipment": eq,
            "Surplus": arrivals[eq] - total_allocated
        })
    surplus_df = pd.DataFrame(surplus)
    keep_open_surplus = st.session_state.get("surplus", False)
    with st.expander("View surplus equipment", expanded=keep_open_surplus):
        add_checkbox(keep_open_surplus, "surplus")
        st.dataframe(surplus_df, hide_index=True, width=500)

        st.caption("Surplus equipment refers to the amount of arriving equipment that remains after allocation.")
