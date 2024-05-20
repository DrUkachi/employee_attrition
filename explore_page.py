import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from apps.core.config import Config


@st.cache_data
def load_data():
    config = Config()

    data_path = config.training_data_path
    file_name = "hr_employee_churn_data.csv"
    data = pd.read_csv(f"{data_path}/{file_name}")
    return data


hr_data = load_data()


def show_explore_page():
    st.title("Explore Employee Attrition Data")

    st.write(
        """Employee Attrition Analysis"""
    )

    figure = plt.figure(figsize=(10, 4))
    sns.countplot(x='left', data=hr_data, palette='tab10')

    st.write("""The Number of Employees who stayed or left""")

    st.pyplot(figure)

    # Calculate the total number of employees
    total_employees = hr_data['left'].count()

    # Calculate the number of employees who left
    num_left = hr_data[hr_data['left'] == 1].shape[0]

    # Calculate the number of employees who did not leave
    num_not_left = hr_data[hr_data['left'] == 0].shape[0]

    # Calculate the percentage of employees who left
    percent_left = (num_left / total_employees) * 100

    # Calculate the percentage of employees who did not leave
    percent_not_left = (num_not_left / total_employees) * 100

    st.write(f"Percentage of employees who left: {percent_left:.2f}%")
    st.write(f"Percentage of employees who did not leave: {percent_not_left:.2f}%")

    st.write("""### Key Takeaway
    For the period captured in this data, a lower number of employees have left 
    when compared the number of employees who are still with the organization.
    """)

    # Create a Seaborn catplot
    sns.set(style="whitegrid")

    # Use sns.catplot and get the FacetGrid's figure
    g = sns.catplot(x='left', col='promotion_last_5years', kind='count', data=hr_data, palette='tab10')
    g.fig.suptitle('Distribution of Promotion in the Last Five Years by Left Status', y=1.05)

    # Display the plot in Streamlit
    st.write("""## Promotion in the Last Five Years""")
    st.pyplot(g.fig)

    st.write("""### Key Takeaway
    For the period captured in this data, only 2% of the population have had a 
    promotion in the last 5 years.
        
    94% of people who get promoted are still in the company, wheres only 75% of the 
    population of those who do not get promoted still stay in the company.
        
    Employees getting promoted within 5 years can help reduce employee attrition
        """)

    # Create a Seaborn catplot
    sns.set(style="whitegrid")

    # Use sns.catplot and get the FacetGrid's figure
    g = sns.catplot(x='left', col='salary', kind='count', data=hr_data, palette='tab10')
    g.fig.suptitle('Distribution of Salary category by Left Status', y=1.05)

    # Display the plot in Streamlit
    st.write("""## Salary Category""")
    st.pyplot(g.fig)

    st.write("""### Key Takeaway
    For the period captured in this data, low salaried employees are 
    the highest in population as expected.

    Higher salaried employees have the lowest attrition rate (~7%) when compared 
    to the rest of the salary categories

    Low salaried employees have the highest attrition rate (~30%)
            """)

    # Create a Seaborn catplot
    sns.set(style="whitegrid")

    # Use sns.distplot and get the FacetGrid's figure
    g = sns.displot(data=hr_data, x='satisfaction_level', hue='left', kind='kde', palette='viridis', fill=True)
    g.fig.suptitle('Distribution of Satisfaction Level by Left Status', y=1.05)

    # Display the plot in Streamlit
    st.write("""## Satisfaction Level""")
    st.pyplot(g.fig)

    st.write("""### Key Takeaway
    Employees with higher satisfaction level tend to stay in the company when compared 
    to employees with lower satisfaction level

    Employees who stayed tend to have higher satisfaction levels, particularly 
    around 0.7.
    
    Employees who left are more likely to have lower satisfaction levels, with 
    concentrations around 0.1 and 0.4.
            """)

    # Use sns.distplot and get the FacetGrid's figure
    g = sns.displot(data=hr_data, x='last_evaluation', hue='left', kind='kde', palette='viridis', fill=True)
    g.fig.suptitle('Distribution of Last Evaluation by Left Status', y=1.05)

    # Display the plot in Streamlit
    st.write("""## Last Evaluation""")
    st.pyplot(g.fig)

    st.write("""### Key Takeaway
    Employees who stayed tend to have higher last_evaluation scores, 
    particularly around 0.85.
    
    Employees that eventually leave have also have a bimodal distribution on 
    0.45 and 0.9 with lower density between these two numbers when compared 
    with the other category.
    
    Employees who left are split between lower and higher last_evaluation scores, 
    with notable concentrations around 0.45 and 0.9.
                """)

    # Use sns.boxplot and get the Axes object
    fig, ax = plt.subplots()
    sns.boxplot(x='left', y='number_project', data=hr_data, palette='viridis', ax=ax)
    ax.set_title('Box Plot of Number of Projects by Left Status')

    # Display the plot in Streamlit
    st.write("""## Number of Projects by Left Status""")
    st.pyplot(fig)

    st.write(
        """
    ### Key Takeaway
    Employees who left the company handled a wider range of projects (from 2 to 7) compared to those 
    who stayed (from 2 to 5).
    
    The IQR for employees who left is larger, indicating greater variability in the number of projects they handled.
    
    Employees handling a higher number of projects may experience higher stress or workload, 
    which could contribute to higher turnover rates.
    
    The wider range and higher median number of projects among those who left suggest that project load might 
    be a factor influencing employees' decisions to leave.
    """
             )

    # Use sns.boxplot and get the Axes object
    fig, ax = plt.subplots()
    sns.boxplot(x='left', y='time_spend_company', data=hr_data, palette='viridis', ax=ax)

    ax.set_title('Box Plot of Time spent in the company by Left Status')

    # Display the plot in Streamlit
    st.write("""## Time spent in the company by Left Status""")
    st.pyplot(fig)

    st.write(
        """
    ### Key Takeaway
    Those who stay in the company, spend less company time compared to those that do not stay. 
    This is likely because of fatigue and stress.
    """)


