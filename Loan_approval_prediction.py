# ----------------------------------------------------------------importing libraries-----------------------------------------------------------------------------------------------------------------------------
import streamlit as st # Create App
import streamlit.components as components #View htlm
from PIL import Image # insret photo
import numpy as np # mathmatcal
import pandas as pd # Read Data And clean
import matplotlib.pyplot as plot # Create Viz
import seaborn as sns # Create Viz
import plotly.express as px # Create Viz
from collections import Counter # count Values each value
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#import pickle # load Model Ai
import base64 # Read Pdf
#--------------------------------------------------------------------Functions----------------------------------------------------------------------------------------------------------------------------------
# Data trend 
def Data_trend(value):
    men=value.mean() # Average
    miden_=value.median() # median
    if men > miden_ : # if Average more then median
        data_trend= "Right" # if True Right
    elif men < miden_ :
        data_trend= "Left"
    else :
        data_trend = "symmetric"
    return data_trend
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# most Value duplication
def most_common_Vales(value):
  value_counts = Counter(value)
  most_common, count = value_counts.most_common(1)[0]
  return most_common, count
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# max value and count duplication
def max_count_value(value):
    Max_value=value.max()
    count_value_max=np.count_nonzero(value == Max_value)
    return Max_value , count_value_max
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# min value and count duplication
def min_count_value(value):
    min_value=value.min()
    count_value_min=np.count_nonzero(value == min_value)
    return min_value , count_value_min
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# column describe 
def describe(col):
    if col.dtype !="object":
        des=(pd.DataFrame(col).describe().style.background_gradient(cmap='viridis', axis=1))
    else:
        des=(pd.DataFrame(col).describe(include="object").style.background_gradient(cmap='viridis', axis=1))
    return des
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Summary Numbercal
def summary(value):
    # max value and count duplication
    Max_value  , count_value =max_count_value(value)
    st.warning(f"The maximum value equals : {Max_value} --- The number of Value  {Max_value}  is equal :  {count_value}")
    # min value and count duplication
    Min_value ,count_value =min_count_value(value)
    st.warning(f"The minimum value equals : {Min_value} --- The number of value {Min_value}  is equal :  {count_value}")
    # most value duplication
    most_common_value, count = most_common_Vales(value)
    st.warning(f"The maximum duplication value equals :  {most_common_value}  --- The number of duplications is greater than the duplication value is equal to :  {count}")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#create_pairplot
def create_pairplot(df, numerical_cols):
    g = sns.pairplot(df[numerical_cols], diag_kind="kde")  # Use kdeplot for diagonal
    # Customize the plot using Matplotlib
    g.fig.suptitle('Pairplot of Numerical Features', y=1.02)
    plot.tight_layout()
    # Display the plot in Streamlit
    st.pyplot(g.fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#viwe Pdf
def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#create_boxplots
def create_boxplots(df):
    fig, ax = plot.subplots(figsize=(10, 6))
    df.plot.box(vert=True, ax=ax)
    ax.set_title("Boxplot for All Numerical Columns")
    ax.set_xlabel("Value")
    ax.set_ylabel("Column Name")
    st.pyplot(fig)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# configure page And read Data
st.set_page_config(layout='wide')
df = pd.read_csv(r'D:\project_finel\Loan approval prediction.csv')
df_copy =df.copy()
df_copy2=df_copy.drop(["loan_status","id"] , axis=1)
DataTrend=[]
colsN=[]
colsOb=[]
for i in df_copy2.columns:
    if df_copy2[i].dtype !="object":
        x=Data_trend(df_copy2[i])
        DataTrend.append(x)
        colsN.append(i)
    else:
        colsOb.append(i)
numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#sidebar And name page
st.sidebar.title("Page")
option = st.sidebar.selectbox(" ",["Home",'Read Data And Information','EDA',"Dashboard",'ML','NoteBook'])
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#select page
if option == 'Read Data And Information':
    st.title("Loan Appeoval Prediction")
    st.header("1 - Data ")
    st.write(df_copy)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#information about loans
    st.header("2 - Information ")
    st.subheader('* Numerical column')
    st.write(df_copy.describe().style.background_gradient(cmap='viridis', axis=1))
    st.subheader('* Object columns')
    st.write(df_copy.describe(include="object").style.background_gradient(cmap='viridis', axis=1))
    st.subheader('* Data Trend')
    trend_=pd.DataFrame({'Cols': colsN, 'DataTrend': DataTrend})
    st.write(trend_)
    st.subheader("*  Correlation Matrix")
    st.write(df_copy.corr(numeric_only=True).T.style.background_gradient(cmap='viridis', axis=1))
    st.write("*"*50)
    st.title("Thank You")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif option == 'EDA':
    st.title("EDA")
    st.write("*"*50)
    st.sidebar.title("EDA")
    #select analysis
    option_=st.sidebar.radio("Choose the type of analysis ",["Numerical columns","Object columns","Group Data","Scatter Plot",
    "Box plot and violin","Histgram","Pairplot","Correlation Heatmap","Boxplot"])
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#analysis col numbercal
    if option_=="Numerical columns":
        st.title("Numerical Features")
        options = st.selectbox("Select the column Numerical" , colsN)
        col11_11 , col11_22 =st.columns(2)
        with col11_11:
            des=describe(df_copy[options])
            titledes= "Describe Column : "+options
            st.write("*"*50)
            st.write(titledes)
            st.write(des)
        with col11_22:
            st.write("*"*50)
            titlesumm= "Summary Column : "+options
            st.write(titlesumm)
            # Data trend
            data_trend=Data_trend(df_copy[options])
            titleDataT= "Data Trend Column : " + options + " : "+data_trend
            st.warning(titleDataT) 
            summary(df_copy[options])
        st.write("*"*50)
        col12_11 , col12_22 =st.columns(2)
        with col12_11:
            title_= "BoxPolt "+options
            fig=px.box(y=df_copy[options],color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=title_)
            st.plotly_chart(fig)
        with col12_22:
            htitle=  'Histogram '+ options
            fig=px.histogram(df_copy[options],color_discrete_sequence=px.colors.qualitative.Dark24_r,
                template="seaborn",title=htitle )
            st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#analysis col object
    elif option_=="Object columns":
        st.title('Object columns')
        options2 = st.selectbox("Select the column object" , colsOb)
        #st.write("*"*50)
        col13_11 , col13_22 , col13_33=st.columns(3)
        group=df_copy.groupby(options2)[options2].count()
        with col13_11:
            des_=describe(df_copy[options2])
            titledes_= "Describe Column : "+options2
            st.write("*"*50)
            st.write(titledes_)
            st.write(des_)
        with col13_22: 
            st.write("*"*50)
            fig=px.bar(x=group.index , y=group.values , text=group.values ,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",
                title='Count VS '+options2)
            st.plotly_chart(fig)
        with col13_33:
            st.write("*"*50)
            fig=px.pie(names=group.index , values=group.values ,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",
                title='Pie VS '+options2)
            st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#group data on tow col and select box col and bar chart
    elif option_=="Group Data":
        st.title("Group Data ")
        st.subheader("one column object")
        status_map = {1: "Approved",0: "Not-Approved"} 
        df_copy['loan_status_cat'] = df_copy['loan_status'].map(status_map)
        colsOb1=colsOb
        colsOb1.append('loan_status_cat')
        col1 , col2  , col3 = st.columns(3)
        with col1:
            Object_columns = st.selectbox("Columns object" , colsOb1 ,key='A')
        with col2:
            Numerical = st.selectbox("Columns Numerical" , colsN)
        with col3:
            function_=st.selectbox("Select the functoin",['mean','count','sum','min','max','std','median','var'])
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        col1_11 , col2_22 = st.columns(2) 
        df_group=df_copy.groupby(Object_columns).agg({Numerical : function_})
        df_group=df_group.reset_index().sort_values(by=Object_columns,ascending=False)
        newname=function_+" " +Numerical
        df_group=df_group.rename({Numerical:newname},axis=1)
        with col1_11:
            fig=px.bar(x=df_group[Object_columns] , y=df_group[newname] , text=df_group[newname] ,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=Object_columns +" Vs "+newname)
            st.plotly_chart(fig)
        with col2_22:
            fig=px.pie(names=df_group[Object_columns] , values=df_group[newname] , 
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=Object_columns +" Vs "+newname)
            st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#insert select groub nore date
        st.write("*"*50)
        st.title("Group Data ")
        st.subheader("more column object")
        col_11 , col_12 , col_13 = st.columns(3)
        with col_11:
            Object_columns_ = st.multiselect("Columns object" , colsOb1,default=[colsOb[0],colsOb[-1]])
        with col_12:
            Numerical_ = st.selectbox("Columns Numerical" , colsN,key="AA")
        with col_13:
            function_1=st.selectbox("Select the functoin",['mean','count','sum','min','max','std','median','var'],key="AB")
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#group data more col and chart bar
        col1_11 , col2_22 = st.columns(2)
        try:
            g=df_copy.groupby(Object_columns_).agg({Numerical_:function_1})
            g=g.unstack()
            fig, ax = plot.subplots(figsize=(10, 6))
            g.plot(kind='bar', stacked=False,ax=ax, figsize=(10, 6) , color =sns.color_palette("Blues_r"))
            plot.xlabel(Numerical_)
            plot.ylabel(function_1)
            plot.title(Object_columns_[1] +" AND " + Object_columns_[0]+ ' BY '+ function_1 +" "+Numerical_)
            plot.xticks(rotation=45)
            plot.legend(title=Object_columns_[1] +"Category")
            st.pyplot(fig)
        except:
              st.title("select columns")
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#sctter plot
    elif option_=="Scatter Plot":
        st.title("Scatter Plot ")
        #insert select box
        col1_ , col2_, col3_ = st.columns(3)
        with col1_:
            colsN_1=colsN
            Numerical_1 = st.selectbox("Select the column Numerical " , colsN_1,key=1)
        with col2_:
            colsN_2=colsN
            Numerical_2 = st.selectbox("Select the column Numerical " , colsN_2,key=2)
        with col3_:
            sct_=colsOb
            sct_.append("None")
            sct_=reversed(sct_)
            Object_columns_ = st.selectbox("Select the column object for color" , sct_ )
        st.write("*"*50)
        #no color
        if Object_columns_ =="None":
            fig=px.scatter(x=df_copy[Numerical_1],y=df_copy[Numerical_2],title=Numerical_1 + " VS "+Numerical_2,
            color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn")
            st.plotly_chart(fig)
        #color
        else:
            fig=px.scatter(x=df_copy[Numerical_1],y=df_copy[Numerical_2],color=df_copy[Object_columns_],
            title=Numerical_1 + " VS "+Numerical_2,color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn")
            st.plotly_chart(fig)
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# box and vioin plot
    elif option_=="Box plot and violin":
        st.title("Box plot and violin ")
        col1_1 , col2_2= st.columns(2)
        with col1_1:
            Object_columns_3= st.selectbox("Columns object" , colsOb ,key='AB')
        with col2_2:
            Numerical_3 = st.selectbox("Columns Numerical " , colsN,key=3)
        st.write("*"*50)
        col3_1 ,col3_2 = st.columns(2)
        with col3_1:
            fig=px.box(x=df_copy[Object_columns_3],y=df_copy[Numerical_3] ,title="BoxPlot "+Object_columns_3+" Vs "+Numerical_3,
            color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn") 
            st.plotly_chart(fig)
        with col3_2:
            fig=px.violin(x=df_copy[Object_columns_3],y=df_copy[Numerical_3] ,title="violin "+Object_columns_3+" Vs "+Numerical_3,
            color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn") 
            st.plotly_chart(fig)
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Histgram
    elif option_ == 'Histgram':
        st.subheader("Histgram of Numerical Features")
        # Create the histogram plot
        fig, ax = plot.subplots(figsize=(15, 10))
        df_copy2.hist(bins=30, ax=ax, layout=(3, 3))
        plot.tight_layout()
        # Display the plot in Streamlit
        st.pyplot(fig)
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pairplot all data
    elif option_ == 'Pairplot':
        st.subheader("Pairplot of Numerical Features")
        numerical_cols=df_copy2.select_dtypes(include=['int64', 'float64']).columns
        create_pairplot(df_copy2, numerical_cols) 
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#corr Heatnp
    elif option_ == 'Correlation Heatmap':
        st.subheader("Correlation Heatmap of Numerical Features")
        # Calculate correlation matrix
        correlation = df_copy[numeric_cols].corr()
        # Create the heatmap using Seaborn
        fig, ax = plot.subplots(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        # Add title and adjust layout
        plot.title('Correlation Heatmap of Numerical Variables')
        plot.tight_layout()
        # Display the plot in Streamlit
        st.pyplot(fig)  
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Boxplot all data     
    elif option_ == 'Boxplot':
        st.subheader("Boxplot of Numerical Features ")
        create_boxplots(df_copy2)
    st.title("Thank You")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif option == 'ML':
    st.title("Model AI")
    st.write("*"*50)
    st.title("Predict loan status")
    st.subheader("Predict Approved OR Rejected The Lone")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Preprocessing Data LabelEncoder AND OUTLIYR
    #LabelEncoder for person_home_ownership
    st.write("Fill Form for Predict loan status And Click Botten Sunmit for show result")
    st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#create form And replace date object
    
    col31 , col32 , col33 ,col34 = st.columns(4)
    with col31 :
        age = st.number_input("age person")
        age=int(age)
        income = st.number_input("person_income ")
        income=int(income)
        ownershiplis=list(pd.unique(df_copy['person_home_ownership']))
        ownership=st.selectbox("ome_ownership",ownershiplis)
        iosh=ownershiplis.index(ownership)
    with col32:
        pperson_emp_length=st.number_input("emp_length")
        loan_intentlis=list(pd.unique(df_copy['loan_intent']))
        loan_intent=st.selectbox("loan_intent",loan_intentlis)
        ili=loan_intent.index(loan_intent)
        loan_gradelis=list(pd.unique(df_copy['loan_grade']))
        loan_grade=st.selectbox("loan_grade",loan_gradelis)
        ilg=loan_gradelis.index(loan_grade)
    with col33:
        loan_amnt=st.number_input("loan_amnt")
        loan_amnt=int(loan_amnt)
        loan_int_rate=st.number_input("loan_int_rate")
        loan_percent_income =st.number_input("loan_p_income")
    with col34:
        cb_person_default_on_filelis=list(pd.unique(df_copy['cb_person_default_on_file']))
        cb_person_default_on_file=st.selectbox("cb_p_defa_fle",cb_person_default_on_filelis)
        icpdof=cb_person_default_on_filelis.index(cb_person_default_on_file)
        cb_person_cred_hist_length =st.number_input("cb_p_cd_ht_len")
        cb_person_cred_hist_length=int(cb_person_cred_hist_length)
        #st.write("Click Sunmit for Show the result")
    st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#loade model AI
    st.title("Result loan_status ")
    ownership_list=list(pd.unique(df['person_home_ownership']))
    list_index=[0,1,2,3]
    person_home_ownership_code=pd.DataFrame({'value':ownership_list ,'code':list_index})
    df['person_home_ownership']=df['person_home_ownership'].replace({'RENT':0,'OWN':1,'MORTGAGE':2,'OTHER':3})
    #LabelEncoder for cb_person_default_on_file
    cb_person_default_on_file_list=list(pd.unique(df['cb_person_default_on_file']))
    list_index3=[0,1]
    person_home_ownership_code=pd.DataFrame({'value':cb_person_default_on_file_list ,'code':list_index3})
    df['cb_person_default_on_file']=df['cb_person_default_on_file'].replace({'N':0,'Y':1})
    #LabelEncoder for loan_intent
    loan_intent_list=list(pd.unique(df["loan_intent"]))
    list_index1=[0,1,2,3,4,5]
    loan_intent_code=pd.DataFrame({'value':loan_intent_list ,'code':list_index1})
    df["loan_intent"]=df["loan_intent"].replace({"EDUCATION":0,'MEDICAL':1,'PERSONAL':2,'VENTURE':3,'DEBTCONSOLIDATION':4,'HOMEIMPROVEMENT':5}) 
    #LabelEncoder for loan_grade
    loan_grade_list=list(pd.unique(df['loan_grade']))
    list_index2=[0,1,2,3,4,5,6]
    loan_grade_code=pd.DataFrame({'value':loan_grade_list ,'code':list_index2})
    df["loan_grade"]=df["loan_grade"].replace({ 'B':0,'C':1,'A':2,'D':3,'E':4,'F':5,'G':6})  
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # bast model
    #Data prtation
    X=df.drop(["id","loan_status"] ,axis=1)
    y=df["loan_status"]
    #RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=49)
    btn = st.button("Submit")
    #model = pickle.load(open(r'D:\project_finel\my_model.pkl','rb'))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#predict bottun
    if btn:
        #model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        #Result
        mm=str(np.round(accuracy*100,1))+"%"
        st.subheader(f'Algorithms : Random Forest --- Accuracy: {mm}')
        X_new=[age, income,iosh,pperson_emp_length,ili,ilg,loan_amnt,loan_int_rate,loan_percent_income,icpdof,cb_person_cred_hist_length]
        X_new_array = np.array(X_new)
        X_new_reshaped = X_new_array.reshape(1, -1)
        result = model.predict(X_new_reshaped)
        if result == 1:
            st.write("-----------------------------")
            st.title("The bank will Approved the loan")
            st.write("-----------------------------")
            st.title("Thank You")
        elif result == 0:
            st.write("-----------------------------")
            st.title("The bank will Rejected the loan")
            st.write("-----------------------------")
            st.title("Thank You")        
    st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif option =="Home":
    image1 = Image.open(r'D:\project_finel\machinfy_logo.jpeg')
    st.image(image1,width=50)
    st.title("Graduation project")
    st.title("Application Loan Approval Prediction")
    st.write("*"*50)
    st.subheader("Student Design : Mohamed Hasab Elnaby Badry" )
    st.subheader("Photo : ")
    image = Image.open(r'D:\project_finel\I_am.jpg')
    st.image(image,width=200 )
    st.subheader("Mail : mohammedhasebelnabi@gmail.com")
    st.subheader('Tel : 01115460524 - 01555156024')
    st.write("*"*50)
    st.subheader("Academy :Machinfy")
    st.subheader("Group : DS 402")
    st.write("*"*50)
    st.title("Thank You Machinfy")  
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#select dashboard
elif option == "Dashboard":
    st.sidebar.subheader("Select Dashboard")
    options3=st.sidebar.radio(" ",['Loans','Age','Income'])
    if options3=="Age":
        st.title("Dashboard : Age Analysis")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#filtre
        st.sidebar.header("Filters")
        filter_= st.sidebar.selectbox("Filter loane status" , ["All","Approved", "Rejected"] )
        if filter_ == "All":
            df_filter=df_copy
        elif filter_ == "Approved":
            df_filter= df_copy[df_copy["loan_status"]==1]
        else:
            df_filter= df_copy[df_copy["loan_status"]==0]
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#values card
        loan_count=df_filter["person_age"].count()
        AV_age =df_filter["person_age"].mean()
        Renge_age=max(df_filter["person_age"])-min(df_filter["person_age"])
        cb_person_cred_hist_length_=df_filter["cb_person_cred_hist_length"].mean()
        AV_emp_length=df_filter["person_emp_length"].mean()
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#insert card 
        col41, col42, col43,col44,col45 = st.columns(5)
        col41.metric("Count Loan",loan_count,"N" )
        col42.metric("AVERAGE Person Age ",np.round(AV_age,1),"T_D")
        col43.metric("Range Person Age", Renge_age,"T_D" )
        col44.metric("Average length of credit history", np.round(cb_person_cred_hist_length_,2),"T_D")
        col45.metric("Average length of time a person works", np.round(AV_emp_length,2),"T_D")
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#group age and bar chart
        co1421,col422=st.columns(2)
        with co1421:
            loan_count_by_age = df_filter.groupby('person_age')['person_income'].count().reset_index()
            # Rename the columns for clarity
            loan_count_by_age.columns = ['person_age', 'loan_count']
            # Convert the grouped data into a DataFrame
            loan_count_by_age = pd.DataFrame(loan_count_by_age)
            # Create a new column 'age_group' by categorizing 'person_age' into defined bins
            loan_count_by_age['age_group'] = pd.cut(loan_count_by_age['person_age'], 
                                                    bins=[0,30,  50, 100], 
                                                    labels=[ 'Young Adult',  'Middle Age', 'Senior'])
            # Group the data by 'age_group' and sum the 'loan_count' for each age group
            loan_count_by_age = loan_count_by_age.groupby('age_group')['loan_count'].sum().reset_index()
            # Display the final DataFrame with loan counts by age group
            loan_count_by_age=pd.DataFrame(loan_count_by_age)
            fig=px.bar(y=loan_count_by_age['loan_count'],x=loan_count_by_age['age_group'],text=loan_count_by_age['loan_count'],
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",
                title='Distribution of loan_count by age_group' )
            st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#scatter
            with col422:
                fig=px.scatter(x=df_filter['person_age'],y=df_filter['cb_person_cred_hist_length'],
                title="person_age  VS cb_person_cred_hist_length correlation Equle 0.87",
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn")
                st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#group data and selectbox in dashboard age
        col1 , col2  = st.columns(2)
        with col1:
            Object_columns = st.selectbox("Columns object" , colsOb ,key='A')
        with col2:
            function_=st.selectbox("Select the functoin",['mean','count','min','max','std','median','var'])
        st.write("*"*50)
        col1_11 , col2_22 = st.columns(2) 
        df_group=df_filter.groupby(Object_columns).agg({'person_age' : function_})
        df_group=df_group.reset_index().sort_values(by=Object_columns,ascending=False)
        newname=function_+" " + 'person_age'
        df_group=df_group.rename({'person_age':newname},axis=1)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#insert bar chart dashboard age
        with col1_11:
            fig=px.bar(x=df_group[Object_columns] , y=df_group[newname] , text=df_group[newname] ,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=Object_columns +" Vs "+newname)
            st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#insert pie chart in dashboard age     
        with col2_22:
            fig=px.pie(names=df_group[Object_columns] , values=df_group[newname] , 
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=Object_columns +" Vs "+newname)
            st.plotly_chart(fig)
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif options3=="Loans":
        st.title("Dashboard : Loans Analysis")
        # Create a sidebar for filtering options
        st.sidebar.header("Filters")
        # Add filters for loan status, loan amount, and person age
        loan_status_filter = st.sidebar.multiselect("Loan Status",  ["Approved", "Rejected"])
        if loan_status_filter==["Approved", "Rejected"]:
            loan_status_filter=[1,0]
        elif loan_status_filter==["Approved"]:
            loan_status_filter=[1]
        elif loan_status_filter==["Rejected"]:
            loan_status_filter=[0]
        else:
            loan_status_filter=[1,0]
        loan_amnt_filter = st.sidebar.slider("Loan Amount Range", int(df_copy["loan_amnt"].min()), 
        int(df_copy["loan_amnt"].max()), (int(df_copy["loan_amnt"].min()), int(df_copy["loan_amnt"].max())))
        person_age_filter = st.sidebar.slider("Person Age Range", int(df_copy["person_age"].min()), 
        int(df_copy["person_age"].max()), (int(df_copy["person_age"].min()), int(df_copy["person_age"].max())))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Add colunms for Data frame   
        filtered_df = df_copy.copy()
        list_col=['person_home_ownership','loan_intent','loan_grade']
        Text_col='cb_onfile'
        for i in list_col:
            col_=Text_col+"_"+i
            filtered_df[col_]=filtered_df[i]+" _ "+filtered_df['cb_person_default_on_file']
        filtered_df["cost_laon"]=filtered_df['loan_amnt'] *(filtered_df['loan_int_rate']/100)
        filtered_df["Total_amount_due"]=filtered_df['loan_amnt']+filtered_df["cost_laon"]
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Filter the data based on the selected options
        if loan_status_filter:
            filtered_df = filtered_df[filtered_df["loan_status"].isin(loan_status_filter)]
        if loan_amnt_filter:
            filtered_df = filtered_df[(filtered_df["loan_amnt"] >= loan_amnt_filter[0]) & (filtered_df["loan_amnt"] <= loan_amnt_filter[1])]
        if person_age_filter:
            filtered_df = filtered_df[(filtered_df["person_age"] >= person_age_filter[0]) & (filtered_df["person_age"] <= person_age_filter[1])]
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Values card
        col51,col52,col53,col54=st.columns(4)
        loan_count=filtered_df['loan_status'].count()
        most_value=filtered_df['loan_amnt'].mode()
        most_value=most_value[0]
        cont_most_value=(filtered_df[filtered_df['loan_amnt']==most_value]).count()
        cont_most_value=cont_most_value[0]
        Total_loan=filtered_df['loan_amnt'].sum()
        av_inco=filtered_df['person_income'].mean()
        Total_loan_due=filtered_df['Total_amount_due'].sum()
        av_emp_lenth=filtered_df['person_emp_length'].mean()
        Total_cost=filtered_df['cost_laon'].sum()
        av_cred_lenth=filtered_df['cb_person_cred_hist_length'].mean()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #insert card
        with col51:
           st.metric("Count Loan",loan_count,"N" )
           st.metric("Most recurring loan",str(most_value) +" $","Count "+str(cont_most_value))
        with col52:
            st.metric("Total Loans",Total_loan,"$")
            st.metric("Average income per person",np.round(av_inco,2),"$")
        with col53:
            st.metric("Total Loans due",np.round(Total_loan_due,2),"$")
            st.metric("Average length of service of a person",np.round(av_emp_lenth,2),"T_D")
        with col54:
            st.metric("Total Profit\\Cost Loans",np.round(Total_cost,2),"$")
            st.metric("Average length of credit history",np.round(av_cred_lenth,2),"T_D")
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        col521,col522=st.columns(2)
        with col521:
            st.subheader("Consumer bureau defalt on file Analysis")

            col531,col532,col533=st.columns(3)
            #insert select box
            with col531:
                options4=st.selectbox("Object",['cb_onfile_person_home_ownership','cb_onfile_loan_intent','cb_onfile_loan_grade','cb_person_default_on_file'],key='A')
            with col532:
                options6=st.selectbox("Loans cols",['loan_amnt','Total_amount_due','cost_laon'])
            with col533:
                options5=st.selectbox("Fincattion",['count','mean','sum','min','max','std','median','var'])
            #group Data
            df_group=filtered_df.groupby(options4).agg({options6 : options5})
            df_group=df_group.reset_index().sort_values(by=options4,ascending=False)
            newname=options5+" " +options6
            df_group=df_group.rename({options6:newname},axis=1)
            options7=st.selectbox('Choose a chart type',['Bar','Line','Pie'])
            #insert chart
            if options7=='Bar':
                fig=px.bar(x=df_group[options4] , y=df_group[newname] , text=df_group[newname] ,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=options4 +" Vs "+newname)
                st.plotly_chart(fig)
            elif options7=="Line":
                fig=px.line(x=df_group[options4] , y=df_group[newname] , text=df_group[newname] ,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=options4 +" Vs "+newname)
                st.plotly_chart(fig)
            else:
                fig=px.pie(names=df_group[options4] , values=df_group[newname] , 
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=options4 +" Vs "+newname)
                st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#scater plot select columns
        with col522:
            st.title("Scatter Plot ")
            col2_, col3_ = st.columns(2)
            with col2_:
                colsN_2=colsN
                Numerical_2 = st.selectbox("Select the column Numerical " , colsN_2,key=1)
            with col3_:
                sct_=colsOb
                sct_.append("None")
                sct_=reversed(sct_)
                Object_columns_ = st.selectbox("Select the column object for color" , sct_ )
            st.write("*"*50)
            if Object_columns_ =="None":
                fig=px.scatter(x=df_copy['person_income'],y=df_copy[Numerical_2],title='person_income' + " VS "+Numerical_2,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn")
                st.plotly_chart(fig)
            else:
                fig=px.scatter(x=df_copy['person_income'],y=df_copy[Numerical_2],color=df_copy[Object_columns_],
                title='person_income' + " VS "+Numerical_2,color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn")
                st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#chart histgram by loan_status
        viridis_colors = {  0: '#440154',   1: '#3B528B' }
        g = sns.FacetGrid(filtered_df, col="loan_status", hue="loan_status", height=5, aspect=1.5, palette=viridis_colors)
        g.map(sns.histplot, 'loan_amnt', kde=True)
        g.add_legend()
        plot.subplots_adjust(top=0.85)
        g.fig.suptitle('Loan Amount Distribution by Loan Status')
        st.pyplot(g.fig)
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif options3=="Income":
        st.title("Dashboard : Incomes Analysis")
        # Create a sidebar for filtering options
        st.sidebar.header("Filters")
        # Add filters for loan status, loan amount, and person age ,person_income
        loan_status_filter = st.sidebar.multiselect("Loan Status",  ["Approved", "Rejected"])
        if loan_status_filter==["Approved", "Rejected"]:
            loan_status_filter=[1,0]
        elif loan_status_filter==["Approved"]:
            loan_status_filter=[1]
        elif loan_status_filter==["Rejected"]:
            loan_status_filter=[0]
        else:
            loan_status_filter=[1,0]
        # insert Filter Tool
        loan_income_filter = st.sidebar.slider("Person Income Range", int(df_copy["person_income"].min()), 
        int(df_copy["person_income"].max()), (int(df_copy["person_income"].min()), int(df_copy["person_income"].max())))
        loan_amnt_filter = st.sidebar.slider("Loan Amount Range", int(df_copy["loan_amnt"].min()),
         int(df_copy["loan_amnt"].max()), (int(df_copy["loan_amnt"].min()), int(df_copy["loan_amnt"].max())))
        person_age_filter = st.sidebar.slider("Person Age Range", int(df_copy["person_age"].min()),
         int(df_copy["person_age"].max()), (int(df_copy["person_age"].min()), int(df_copy["person_age"].max())))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Create df copy 
        filtered_df = df_copy.copy()
        list_col=['person_home_ownership','loan_intent','loan_grade']
        Text_col='cb_onfile'
        #Rename Columns New
        for i in list_col:
            col_=Text_col+"_"+i
            filtered_df[col_]=filtered_df[i]+" _ "+filtered_df['cb_person_default_on_file']
        # Add colunms for Data frame 
        filtered_df["cost_laon"]=filtered_df['loan_amnt'] *(filtered_df['loan_int_rate']/100)
        filtered_df["Total_amount_due"]=filtered_df['loan_amnt']+filtered_df["cost_laon"]
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Filter the data based on the selected options
        if loan_status_filter:
            filtered_df = filtered_df[filtered_df["loan_status"].isin(loan_status_filter)]
        if loan_income_filter:
            filtered_df = filtered_df[(filtered_df["person_income"] >= loan_income_filter[0]) & (filtered_df["person_income"] <= loan_income_filter[1])]
        if loan_amnt_filter:
            filtered_df = filtered_df[(filtered_df["loan_amnt"] >= loan_amnt_filter[0]) & (filtered_df["loan_amnt"] <= loan_amnt_filter[1])]
        if person_age_filter:
            filtered_df = filtered_df[(filtered_df["person_age"] >= person_age_filter[0]) & (filtered_df["person_age"] <= person_age_filter[1])]
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Values card
        loan_count=filtered_df['loan_status'].count()
        most_value=filtered_df['person_income'].mode()
        most_value=most_value[0]
        cont_most_value=(filtered_df[filtered_df['person_income']==most_value]).count()
        cont_most_value=cont_most_value[0]
        median_income=filtered_df['person_income'].median()
        av_inco=filtered_df['person_income'].mean()
        range_income=(int(df_copy["person_income"].max())-int(df_copy["person_income"].min()))
        Total_loan_due=filtered_df['Total_amount_due'].sum()
        #insert card
        col51_1,col52_1,col53_1=st.columns(3)
        with col51_1:
           st.metric("Count Loan",loan_count,"N" )
           st.metric("Most recurring Income",str(most_value) +" $","Count "+str(cont_most_value))
        with col52_1:
            st.metric("Average income per person",np.round(av_inco,2),"$")
            st.metric("Total Loans due",np.round(Total_loan_due,2),"$")
        with col53_1:
            st.metric("Median income per person",np.round(median_income,2),"$")
            st.metric("Median income per person",np.round(range_income,2),"$")
        st.write("*"*50)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#insert select chart bar line pie group data
        col51_12,col52_11=st.columns(2)
        with col51_12:
            #insert select box columns
            col1 , col2  = st.columns(2)
            with col1:
                Object_columns = st.selectbox("Columns object" , colsOb ,key='A')
            with col2:
                function_=st.selectbox("Select the functoin",['mean','count','min','max','std','median','var'])
            #group by income
            df_group=filtered_df.groupby(Object_columns).agg({'person_income' : function_})
            df_group=df_group.reset_index().sort_values(by=Object_columns,ascending=False)
            newname=function_+" " + 'person_income'
            df_group=df_group.rename({'person_income':newname},axis=1)
            # insert charts
            chart=st.selectbox('Choose a chart type',['Bar','Line','Pie'])
            if chart =="Bar":
                fig=px.bar(x=df_group[Object_columns] , y=df_group[newname] , text=df_group[newname] ,
                    color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=Object_columns +" Vs "+newname)
                st.plotly_chart(fig)
            elif chart=="Line":
                fig=px.line(x=df_group[Object_columns] , y=df_group[newname] , text=df_group[newname] ,
                    color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=Object_columns +" Vs "+newname)
                st.plotly_chart(fig)
            else:
                fig=px.pie(names=df_group[Object_columns] , values=df_group[newname] , 
                    color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn",title=Object_columns +" Vs "+newname)
                st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#insert scater ploy
        with col52_11:
            st.title("Scatter Plot ")
            col2_, col3_ = st.columns(2)
            with col2_:
                colsN_2=colsN
                Numerical_2 = st.selectbox("Select the column Numerical " , colsN_2,key=1)
            with col3_:
                sct_=colsOb
                sct_.append("None")
                sct_=reversed(sct_)
                Object_columns_ = st.selectbox("Select the column object for color" , sct_ )
            st.write("*"*50)
            if Object_columns_ =="None":
                fig=px.scatter(x=df_copy['person_income'],y=df_copy[Numerical_2],title='person_income' + " VS "+Numerical_2,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn")
                st.plotly_chart(fig)
            else:
                fig=px.scatter(x=df_copy['person_income'],y=df_copy[Numerical_2],color=df_copy[Object_columns_],
                title='person_income' + " VS "+Numerical_2,color_discrete_sequence=px.colors.qualitative.Dark24_r,template="seaborn")
                st.plotly_chart(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#chart histgram by loan_status
        viridis_colors = {  0: '#440154',   1: '#3B528B' }
        g = sns.FacetGrid(filtered_df, col="loan_status", hue="loan_status", height=5, aspect=1.5, palette=viridis_colors)
        g.map(sns.histplot, 'person_income', kde=True)
        g.add_legend()
        plot.subplots_adjust(top=0.85)
        g.fig.suptitle('Person Income Distribution by Loan Status')
        st.pyplot(g.fig)
        st.write("*"*50)
    st.title("Thank You")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif option == "NoteBook":
    list_note=["EDA","Model"]
    option_note=st.sidebar.radio("Select NoteBook",list_note)
    st.title("This NōtBo͝ok : "+ option_note)
    if option_note == "EDA":
        with open(r'D:\project_finel\Loan approval prediction.html',"r") as f:
            html_content = f.read()
        components.v1.html(html_content, width=1600, height=850, scrolling=True)
    elif option_note=="Model":
        with open(r'D:\project_finel\Model Al for Loan approval.html',"r") as f:
            html_content = f.read()
        components.v1.html(html_content, width=1600, height=850, scrolling=True)
    st.title("Thank You")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
