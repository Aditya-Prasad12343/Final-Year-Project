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
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.neural_network import MLPRegressor

    # Create a Streamlit App
    st.title("Graph Selector App")

    # Create a file uploader
    file = st.file_uploader("Upload your file", type=["csv", "xlsx", "txt"])

    if file is not None:
        # Read the file using pandas
        df = pd.read_csv(file)

        # Display the dataframe
        st.write("Selected Dataset:")
        st.dataframe(df)

        # Let the user select the columns
        selected_columns = st.multiselect("Select columns", df.columns)

        # Create a confirmation button
        confirm = st.button("Confirm")

        if confirm:
            # Display the selected columns
            st.write("Selected Columns:")
            st.write(selected_columns)

            # Analyze the selected columns using a neural network
            X = df[selected_columns]
            y = df[selected_columns[0]]
            reg = MLPRegressor().fit(X, y)

            # Suggest a graph to the user based on the neural network predictions
            prediction = reg.predict(X)
            corr = pd.DataFrame({"Actual": y, "Predicted": prediction}).corr().iloc[0, 1]
            if corr > 0.5:
                # Suggest a line plot
                st.write("We suggest using a line plot to show the relationship between the selected columns.")
                plot = sns.lineplot(data=df[selected_columns])
                st.pyplot(plot.figure)
            else:
                # Suggest a scatter plot
                st.write("We suggest using a scatter plot to show the relationship between the selected columns.")
                plot = sns.scatterplot(data=df[selected_columns])
                st.pyplot(plot.figure)

            # Suggest additional graph options based on the data type of the selected columns
            for col in selected_columns:
                dtype = df[col].dtype
                if dtype == "object":
                    # Suggest a bar plot for categorical columns
                    st.write(f"For column '{col}', we suggest using a bar plot to show the values.")
                    plot = sns.countplot(data=df[col])
                    st.pyplot(plot.figure)
                else:
                    # Suggest a histogram or box plot for numeric columns
                    st.write(f"For column '{col}', we suggest using a histogram or box plot to show the distribution.")
                    plot = sns.histplot(data=df[col], kde=True)
                    st.pyplot(plot.figure)
                    plot = sns.boxplot(data=df[col])
                    st.pyplot(plot.figure)
