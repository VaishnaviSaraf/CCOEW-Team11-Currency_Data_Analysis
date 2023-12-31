import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
import streamlit as st
import glob
# The function to fill missing data
def fill_missing():
    # Path to CSV files
    path = "C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset"
    files = glob.glob(path + "/*.csv")

    for fname in files:
        df = pd.read_csv(fname)
        for column in df:
            df[column] = df[column].fillna(method='ffill')

        for column in df:
            if column == 'Date':
                continue
            else:
                mean = df[column].mean()
                df[column] = df[column].fillna(mean)

        df.to_csv(fname, index=False)
        pass

# Function to calculate appreciation or depreciation
def calculate_appreciation_depreciation(currency_data, currency1, currency2):
    try:
        cu = list(map(float, currency_data[currency1]))
        cu1 = list(map(float, currency_data[currency2]))
        rate_current = cu[-1]  # Exchange rate of currency1 in 2021
        rate_previous = cu1[-2]  # Exchange rate of currency2 in 2021

        if rate_current > rate_previous:
            return "Appreciation"
        elif rate_current < rate_previous:
            return "Depreciation"
        else:
            return "No Change"

    except KeyError:
        return "Error: Selected currencies not found in the currency DataFrame."
    
# Streamlit page configuration
st.set_page_config(page_title="Team 11", page_icon=":bar_graph:", layout="wide")

# Calling function to fill missing data
fill_missing()

# headline in the sidebar
st.sidebar.markdown("<h2 style='text-align: center; color:#004d3d;'>NAVIGATION MENU</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(" ", ["Home", "Currency Exchange", "Display Graph", "Prediction Feature", "Currency Converter"])

# Loading initial CSV file
df = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2012.csv")

# Sidebar buttons

#select 1
if page == "Home":
    st.markdown("<h1 style='text-align: center;margin-top:5px; border: solid 2px; color: white;  background: linear-gradient(to bottom, #003366 0%, #33cccc 100%);'>NORTHERN TRUST Demystifying Hackathon TEAM 11 <br></h1>", unsafe_allow_html=True)
    st.image("bgimg3.jpg", use_column_width=True)
    st.markdown("<h6 style='text-align: center; border: solid 2px; color: black; background: #f4f4f4'><br>Project by: <br></h6>", unsafe_allow_html=True)
    
    html_content = """
    <h6 style='text-align: left; border: solid 2px; color: black; background: #f4f4f4'>
    
    <ul>
        <li>Srushti Mahadik</li>
        <li>Vaishnavi Saraf</li>
        <li>Ojashri Thakur</li>
        <li>Priya Waykos</li>
        <li>Aahana Kulkarni</li>
    </ul>
    </h6>
    """
    st.markdown(html_content, unsafe_allow_html=True)
    
#select 2    
elif page == "Currency Exchange":
    st.markdown("<h1 style='text-align: center; color: white;  background: linear-gradient(to bottom, #003366 0%, #33cccc 100%);'>Currency Conversion Page </h1><br><br>", unsafe_allow_html=True)

    # Loading initial CSV file
    df = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2012.csv")
    left_column, right_column = st.columns(2)
    col1, col2, col3 = st.columns(3)

    option1 = 'Select Currency-1'  # Initializing option1 variable
    option2 = 'Select Currency-2'   # Initializing option variable
    with left_column:
        c1 = st.selectbox('Currency Conversion from :', ['Select Currency-1'] + list(df.columns[1:]), index=0 if option1 == 'Select Currency-1' else None)

    with right_column:
        c2 = st.selectbox('To:', ['Select Currency-2'] + list(df.columns[1:]), index=0 if option2 == 'Select Currency-2' else None)


# Loading the currency data for calculations
    currency = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2021.csv")

# currency conversion calculations
    conversion_attempted = False  # Flag to track if a conversion was attempted
    if c1 != 'Select Currency-1' and c2 != 'Select Currency-2':
        conversion_attempted = True
        try:
            cu = list(map(float, currency[c1]))
            cu1 = list(map(float, currency[c2]))
            temp = cu[5]
            temp1 = cu1[5]
            cal = (1 / float(temp)) * float(temp1)

            left_column, right_column = st.columns(2)
            with left_column:
                st.info(" 1 {} = ".format(c1))

            with right_column:
                st.info("{0} {1}".format(cal, c2))

        except KeyError:
            st.error(f"Selected column '{c1}' or '{c2}' not found in the currency DataFrame.")
    if c1 != 'Select Currency-1' and c2 != 'Select Currency-2':
        
    # Calculating appreciation/depreciation
        appreciation_depreciation = calculate_appreciation_depreciation(currency, c1, c2)

    # Display the result
        st.write(f"Appreciation/Depreciation between {c1} and {c2}: {appreciation_depreciation}")

#select 3
elif page == "Display Graph":
    st.markdown("<h1 style='text-align: center;color: white;  background: linear-gradient(to bottom, #003366 0%, #33cccc 100%);'>Display Graph Page </h1><br><br>", unsafe_allow_html=True)
    
    left_column, right_column = st.columns(2)
    
    with left_column:
        option1 = st.selectbox('Currency Conversion from :', ['Currency-1'] + list(df.columns[1:]))

    with right_column:
        option = st.selectbox('TO', ['Currency-2'] + list(df.columns[1:]))

    if option1 != 'Currency-1' and option != 'Currency-2':
        year = st.selectbox('Select year', ('2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'))

        path = "C:/Users/DELL/Desktop/NT_Team11/NTRS_APAC_TEAM-1-main/NTRS_APAC_TEAM-1-main/hack/Currency_Conversion_Test_Data/Exchange_Rate_Report_{}.csv".format(year)
        df = pd.read_csv(path)

# for selecting time frame
        left_column, middle_column, right_column = st.columns(3)
        with middle_column:
            st.markdown("<h4 style='text-align: center; color: black; background: #E09A7A'> Select time frame from below </h4>", unsafe_allow_html=True)
            time_frame = st.radio(
                "",
                ["Weekly", "Monthly", "Quaterly", "Yearly"],
                key="visibility",
            )

# for fetching data based on selected time frame
        def get_yearly_data(option):
            years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
            values = []

            for i in range(len(years)):
                path = "C:/Users/DELL/Desktop/NT_Team11/NTRS_APAC_TEAM-1-main/NTRS_APAC_TEAM-1-main/hack/Currency_Conversion_Test_Data/Exchange_Rate_Report_{}.csv".format(years[i])
                df = pd.read_csv(path)

                if option in df.columns:
                    values.append(df[option].mean())
                else:
                    values.append(None)

            values = pd.DataFrame(values, index=years, columns=[option])
            return values

        def get_montly_data(option):
            data = df[[df.columns[0],option]].copy()
            data['month'] = pd.DatetimeIndex(data[data.columns[0]]).month
            avg = data.groupby(pd.PeriodIndex(data[data.columns[0]], freq="M"))[option].mean()
            avg = avg.to_frame()
            avg['mon'] = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

            avg = avg.set_index(avg['mon'])
            dat = avg[[option]].copy()
            return dat

        def get_quaterly_data(option):
            data = df[[df.columns[0],option]].copy()
            data['month'] = pd.DatetimeIndex(data[data.columns[0]]).month
            avg = data.groupby(pd.PeriodIndex(data[data.columns[0]], freq="M"))[option].mean()
            avg = avg.to_frame()
            avg['mon'] = [0,1,2,3,4,5,6,7,8,9,10,11]
            avg = avg.set_index(avg['mon'])
            dat = avg[[option]].copy()
            temp = list(map(int, avg[option]))
            quarter = []
            sum = 0
            for x in avg['mon']:
                if x %4 == 0:
                    if x == 0:
                        sum = 0
                    else:
                        sum = sum / 4 
                        quarter.append(sum)
                        sum = 0
                sum = sum + temp[x]
            sum = sum / 4
            quarter.append(sum)
            quar = dat[option].copy()
            value = pd.DataFrame()
            value['val'] = quarter

            return value
        if time_frame == "Weekly":
            if option in df.columns:
                data = df[[df.columns[0], option]].copy()
                row = len(df.axes[0])
                data[data.columns[0]] = pd.date_range(data[df.columns[0]].iloc[0], data[df.columns[0]].iloc[-1], periods=row)
                data = data.set_index(df.columns[0])
                minimum = data.min()
                maximum = data.max()
                min_date = data.index.min().strftime("%Y-%m-%d") if not data.empty else None
                max_date = data.index.max().strftime("%Y-%m-%d") if not data.empty else None
                l, r = st.columns(2)
                
                with l:
                    st.info("MINIMUN value: {} at {}".format(minimum.iloc[0], min_date))
                with r:
                    st.info("MAXIMUM value: {} at {}".format(maximum.iloc[0], max_date))
                    
                fig = px.line(data, x=data.index, y=option,
                      labels={option: f"{option} Line Chart"},
                      title=f"{option} Line Chart",
                      line_group=None, color_discrete_sequence=['green'])  

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Selected column '{option}' not found in the DataFrame.")


        elif time_frame == "Monthly":
            if option in df.columns:
                data_month = get_montly_data(option)
                minimum = data_month.min()
                maximum = data_month.max()
                min_date = data_month.index.min()
                max_date = data_month.index.max()
                l, r = st.columns(2)
                with l:
                    st.info("MINIMUM value: {} in the month of {}".format(minimum.iloc[0], min_date))
                with r:
                    st.info("MAXIMUM value: {} in the month of {}".format(maximum.iloc[0], max_date))

                fig = px.line(data_month, x=data_month.index, y=option,
                    labels={option: f"{option} Line Chart"},
                    title=f"{option} Line Chart",
                    line_group=None, color_discrete_sequence=['green'])  

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Selected column '{option}' not found in the DataFrame.")


        elif time_frame == "Quaterly":
            if option in df.columns:
                data_quat = get_quaterly_data(option)
                minimum = data_quat.min()
                maximum = data_quat.max()
                min_date = data_quat.index.min()
                max_date = data_quat.index.max()
                l, r= st.columns(2)
                with l:
                    st.info("MINIMUM value: {} in {}(st/nd/rd/th) Quarter".format(minimum.iloc[0], min_date+1))
                with r:
                    st.info("MAXIMUM value: {} at {}(st/nd/rd/th) Quarter".format(maximum.iloc[0], max_date+1))
                    
                fig = px.line(data_quat, x=data_quat.index, y='val',
                      labels={'val': f"{option} Line Chart"},
                      title=f"{option} Line Chart",
                      line_group=None, color_discrete_sequence=['green'])  

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Selected column '{option}' not found in the DataFrame.")

        elif time_frame == "Yearly":
            if option in df.columns:
                data_year = get_yearly_data(option)
                minimum = data_year.min()
                maximum = data_year.max()
                min_date = data_year.index.min()
                max_date = data_year.index.max()
                l,r = st.columns(2)
                with l:
                    st.info("MINIMUM value: {} in the year {}".format(minimum.iloc[0], min_date))
                with r:
                    st.info("MAXIMUM value: {} in the year {}".format(maximum.iloc[0], max_date))
                    
                fig = px.line(data_year, x=data_year.index, y=option,
                    labels={option: f"{option} Line Chart"},
                    title=f"{option} Line Chart",
                    line_group=None, color_discrete_sequence=['green']) 

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Selected column '{option}' not found in the DataFrame.")
                
#select 4
elif page == "Prediction Feature":
    st.markdown("<h1 style='text-align: center; color: white;  background: linear-gradient(to bottom, #003366 0%, #33cccc 100%);'>Prediction Feature Page </h1><br><br>", unsafe_allow_html=True)
    df_2012 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2012.csv")
    df_2013 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2013.csv")
    df_2014 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2014.csv")
    df_2015 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2015.csv")
    df_2016 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2016.csv")
    df_2017 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2017.csv")
    df_2018 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2018.csv")
    df_2019 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2019.csv")
    df_2020 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2020.csv")
    df_2021 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2021.csv")
    df_2022 = pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NT_Hackathon/Team 11/Project/Dataset/Exchange_Rate_Report_2022.csv")

    # Merging all the data into a single DataFrame
    frames = [df_2012, df_2013, df_2014, df_2015, df_2016, df_2017, df_2018, df_2019, df_2020, df_2021, df_2022]
    full_data = pd.concat(frames, axis=0, ignore_index=True)

    # Reseting the index to make 'Date' a column again
    full_data.reset_index(inplace=True)

    # Converting 'Date' column to datetime
    full_data['Date'] = pd.to_datetime(full_data['Date'], format='%d-%b-%y')

    # Sorting the DataFrame by 'Date'
    full_data.sort_values(by='Date', inplace=True)

    # Seting 'Date' as the index
    full_data.set_index('Date', inplace=True)

    # Removing whitespaces in column names
    full_data.columns = full_data.columns.str.strip()
    # Assuming your DataFrame is named 'full_data'
    full_data.columns = full_data.columns.str.replace(r'\s+\(', '(', regex=True)

    full_data = full_data.interpolate()  # To Interpolate missing values

    available_currencies = full_data.columns[1:]  # Assuming the first column is 'Date'
    print("Available Currencies:")
    for currency in available_currencies:
        print(currency)
        
    # Extracting currency codes from column names
    currency_codes = [col.split('(')[-1].split(')')[0] for col in full_data.columns[1:]]
    currency_dict = {code: col for code, col in zip(currency_codes, full_data.columns[1:])}

    #user input
    target_currency_code = st.text_input("Enter the target currency code:")

    # Checking if target_currency_code is not empty
    if target_currency_code:
        # Accessing the currency_dict using target_currency_code
        target_currency_code = target_currency_code.upper()
        target_currency = currency_dict.get(target_currency_code)
        if target_currency:
            st.write(f"Selected currency: {target_currency}")
        else:
            st.write(f"Currency with code '{target_currency_code}' not found.")
            st.write("Please enter a valid currency code.")
    else:
        st.write("Please enter a valid currency code.")

    if target_currency_code:

        from prophet import Prophet

        full_data.reset_index(inplace=True)

        # Choose a target 
        target_currency = currency_dict[target_currency_code]

        X = full_data.drop(target_currency, axis=1)

        y = full_data[target_currency]

        # Spliting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # To Select the relevant columns for the model
        selected_columns = ['Date', target_currency]
        df_prophet = full_data[selected_columns].copy()

        df_prophet.rename(columns={'Date': 'ds', target_currency: 'y'}, inplace=True)

        model_prophet = Prophet()
        model_prophet.fit(df_prophet)

        future = model_prophet.make_future_dataframe(periods=len(X_test))

        # Make predictions
        forecast = model_prophet.predict(future)

        # Ploting the forecast
        fig = model_prophet.plot(forecast)
        st.pyplot(fig)

#select 5
elif page == "Currency Converter":
    st.markdown("<h1 style='text-align: center; color: white;  background: linear-gradient(to bottom, #003366 0%, #33cccc 100%);'>Currency Converter </h1>", unsafe_allow_html=True)
    
    def currency_converter(base_currency, target_currency, amount, exchange_rate):
        if base_currency and target_currency and amount and exchange_rate:
            try:
                converted_amount = round(amount * exchange_rate, 2)
                return f"{converted_amount} "
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return "Please fill all fields"

    st.title("Currency Converter")
    option1 = 'Select Base Currency'  
    option2 = 'Select Target Currency'   
    base_currency = st.selectbox("Select Base Currency", ['Select Base Currency'] + list(df.columns[1:]), index=0 if option1 == 'Select Base Currency' else None)
  
    target_currency = st.selectbox("Select Target Currency",  ['Select Target Currency'] + list(df.columns[1:]), index=0 if option2 == 'Select Target Currency' else None)
  
    amount = st.number_input("Enter Amount", min_value=0.01, step=0.01, format="%.2f")
    currency = pd.read_csv("C:/Users/DELL/Desktop/NT_Team11/NTRS_APAC_TEAM-1-main/NTRS_APAC_TEAM-1-main/hack/Currency_Conversion_Test_Data/Exchange_Rate_Report_2021.csv")
    conversion_attempted = False  
    if base_currency != 'Select Base Currency' and target_currency != 'Select Target Currency':
        conversion_attempted = True
        
        cu = list(map(float, currency[base_currency]))
        cu1 = list(map(float, currency[target_currency]))
        temp = cu[5]
        temp1 = cu1[5]
        cal = (1 / float(temp)) * float(temp1)
        exchange_rate = cal

        if st.button("Convert"):
            result = currency_converter(cu, cu1, amount, exchange_rate)
            st.write(f"Converted Amount: {result}")

st.sidebar.write()