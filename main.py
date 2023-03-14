import streamlit as st
# Core Pkgs

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Agg")
import seaborn as sns



from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from download import download_button
from st_aggrid import GridUpdateMode, DataReturnMode



def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


st.set_option('deprecation.showPyplotGlobalUse', False)

activities = ["EDA", "Plots","Handle NULL Values", "Graph Prediction"]
choice = st.sidebar.selectbox("Select Activities", activities)

if choice == 'EDA':
    st.title("Exploratory Data Analysis")
    st.subheader("A Final Year Project")
    st.text("By Dishti Kundra and Aditya Prasad")

    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    st.info(
            f"""
                    Sample File: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                    """
    )

    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())


        if st.checkbox("Export data"):
            from st_aggrid import GridUpdateMode, DataReturnMode

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_side_bar()
            gridOptions = gb.build()

            st.success(
                f"""
                    💡 Tip! Hold the shift key when selecting rows to select multiple rows at once!
                    """
            )

            response = AgGrid(
                df,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=False,
            )

            df = pd.DataFrame(response["selected_rows"])
            
            st.subheader("Filtered data will appear below 👇 ")
            st.text("")

            st.table(df)

            st.text("")

            c29, c30, c31 = st.columns([1, 1, 2])


            CSVButton = download_button(
                df,
                "File.csv",
                "Download to CSV")



        if st.checkbox("Show Shape"):
            st.write(df.shape)

        if st.checkbox("Show Columns"):
            all_columns = df.columns.to_list()
            st.write(all_columns)

        if st.checkbox("Summary"):
            st.write(df.describe())

        if st.checkbox("Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:, -1].value_counts())

        if st.checkbox("Correlation Plot(Matplotlib)"):
            plt.matshow(df.corr())
            st.pyplot()

        if st.checkbox("Correlation Plot(Seaborn)"):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot()

        if st.checkbox("Pie Plot"):
            all_columns = df.columns.to_list()
            column_to_plot = st.selectbox("Select 1 Column", all_columns)
            pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot()



elif choice == 'Plots':
    st.subheader("Data Visualization")
    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())

        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:, -1].value_counts().plot(kind='bar'))
            st.pyplot()

        # Customizable Plot

        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))

            # Plot By Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'bar':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)



            elif type_of_plot == 'line':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            # Custom Plot
            elif type_of_plot:
                cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()

elif choice == 'Handle NULL Values':
    def handle_missing_values(df, method):
        if method == "Deleting the Missing values":
            df = df.dropna()
        elif method == "Deleting the Entire Row":
            df = df.dropna(axis=0)
        elif method == "Deleting the Entire Column":
            df = df.dropna(axis=1)
        elif method == "Imputing the Missing Value":
            df = df.fillna(value=0) # replace missing values with 0
        elif method == "Replacing With Arbitrary Value":
            arbitrary_value = st.text_input("Enter the arbitrary value:")
            df = df.fillna(value=arbitrary_value)
        elif method == "Replacing With Mean":
            mean = df.mean()
            df = df.fillna(value=mean)
        elif method == "Replacing With Mode":
            mode = df.mode().iloc[0]
            df = df.fillna(value=mode)
        elif method == "Replacing With Median":
            median = df.median()
            df = df.fillna(value=median)
        elif method == "Replacing with Previous Value – Forward Fill":
            df = df.fillna(method='ffill')
        elif method == "Replacing with Next Value – Backward Fill":
            df = df.fillna(method='bfill')
        elif method == "Interpolation":
            df = df.interpolate()
        elif method == "Imputing Missing Values For Categorical Features":
            category = st.text_input("Enter the categorical feature:")
            value = st.text_input("Enter the value for imputation:")
            df[category] = df[category].fillna(value=value)
        elif method == "Impute the Most Frequent Value":
            df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
        elif method == "Impute the Value “missing”, which treats it as a Separate Category":
            df = df.fillna(value='missing')
            
        return df
    # create a sample dataframe
    st.subheader("Handle Missing Values")
    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())
        
        # display the dataframe
        st.write("Original Dataframe:")
        st.write(df)

        # define the options for handling missing values
        options = ["Deleting the Missing values", "Deleting the Entire Row", "Deleting the Entire Column", 
                   "Imputing the Missing Value", "Replacing With Arbitrary Value", "Replacing With Mean", 
                   "Replacing With Mode", "Replacing With Median", "Replacing with Previous Value – Forward Fill", 
                   "Replacing with Next Value – Backward Fill", "Interpolation", 
                   "Imputing Missing Values For Categorical Features", "Impute the Most Frequent Value", 
                   "Impute the Value “missing”, which treats it as a Separate Category"]
        # get the user's choice for handling missing values from a dropdown
        choice = st.selectbox("Select the method for handling missing values:", options)

        # confirm the selected method with a dialogue box
        confirm = st.button("Confirm")
        if confirm:
            # handle missing values based on the user's choice
            df = handle_missing_values(df, choice)
            # display the resulting dataframe
            st.write("Resulting Dataframe:")
            st.write(df)

elif choice == 'Graph Prediction':    
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt

    def analyze_data(data):
        # Check data types of columns
        dtypes = data.dtypes.unique()

        # If all columns are numeric, suggest scatter plot
        if len(dtypes) == 1 and dtypes[0] != 'object':
            return "Scatter Plot"

        # If one column is categorical, suggest bar chart or pie chart
        elif len(dtypes) == 2 and 'object' in dtypes:
            categorical_col = data.select_dtypes(include='object').columns[0]
            if data[categorical_col].nunique() <= 10:
                return "Bar Chart"
            else:
                return "Pie Chart"

        # If two columns are numeric, suggest line plot or scatter plot
        elif len(dtypes) == 2 and 'object' not in dtypes:
            return "Line Plot" if abs(data.iloc[:,0].corr(data.iloc[:,1])) > 0.5 else "Scatter Plot"

        # If more than two columns are selected, suggest heatmap
        else:
            return "Heatmap"


    def generate_plot(data, suggested_graph_type):
        # Generate plot based on suggested graph type
        if suggested_graph_type == "Scatter Plot":
            plt.scatter(data.iloc[:,0], data.iloc[:,1])
            st.pyplot()
        elif suggested_graph_type == "Bar Chart":
            categorical_col = data.select_dtypes(include='object').columns[0]
            data_counts = data[categorical_col].value_counts()
            plt.bar(data_counts.index, data_counts.values)
            plt.xticks(rotation=90)
            st.pyplot()
        elif suggested_graph_type == "Pie Chart":
            categorical_col = data.select_dtypes(include='object').columns[0]
            data_counts = data[categorical_col].value_counts()
            plt.pie(data_counts.values, labels=data_counts.index, autopct='%1.1f%%')
            st.pyplot()
        elif suggested_graph_type == "Line Plot":
            plt.plot(data.iloc[:,0], data.iloc[:,1])
            st.pyplot()
        elif suggested_graph_type == "Heatmap":
            plt.imshow(data.corr())
            plt.colorbar()
            st.pyplot()


    st.title("Select File and Columns")

    # Allow user to upload file
    file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

    if file is not None:
        df = pd.read_csv(file)

        # Display list of columns to select
        columns = list(df.columns)
        selected_columns = st.multiselect("Select columns", columns)

        # Check if columns are selected
        if len(selected_columns) > 0:
            # Get data for selected columns
            data = df[selected_columns]

            # Analyze data using AI and suggest graph type
            suggested_graph_type = analyze_data(data)
            st.write(f"Suggested Graph Type: {suggested_graph_type}")

            # Generate plot based on suggested graph type
            generate_plot(data, suggested_graph_type)
