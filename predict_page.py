import pandas as pd
import traceback

from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response

from apps.prediction.predict_model import PredictModel
from apps.core.config import Config
import streamlit as st


def single_prediction_streamlit(form_submission):
    """Performs a single prediction using a trained model.

    This function retrieves configuration parameters, prepares prediction data
    from a POST request, and initializes a PredictModel object. It then calls
    the model's single_predict_from_model method to perform a single prediction
    on the provided data. Finally, it returns a Response object with the
    predicted output on success or an error message detailing any exceptions.

    **Expected request:**

    - Method: POST
    - Form data:
        - satisfaction_level (float)
        - last_evaluation (float)
        - number_project (int)
        - average_monthly_hours (int)
        - time_spend_company (int)
        - work_accident (int)  # Assuming this refers to number of accidents
        - promotion_last_5years (int)
        - salary (object)  # Data type might need clarification based on usage

    **Returns:**

    - Response:
        - A Response object with the predicted output on success.
        - A Response object with an error message on exception.
    """
    try:
        config = Config()
        # get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        print('Streamlit Test')

        data = pd.DataFrame(data=[form_submission],
                            columns=['empid', 'satisfaction_level', 'last_evaluation', 'number_project',
                                     'average_montly_hours', 'time_spend_company', 'Work_accident',
                                     'promotion_last_5years', 'salary'])
        # using dictionary to convert specific columns
        convert_dict = {'empid': int,
                        'satisfaction_level': float,
                        'last_evaluation': float,
                        'number_project': int,
                        'average_montly_hours': int,
                        'time_spend_company': int,
                        'Work_accident': int,
                        'promotion_last_5years': int,
                        'salary': object}

        data = data.astype(convert_dict)

        # object initialization
        predict_model = PredictModel(run_id, data_path)
        # prediction the model
        output = predict_model.single_predict_from_model(data)

        if output == 1:
            return"The employee might be part of the attrition."
        return "The employee might not be part of the attrition."

    except ValueError as ve:
        tb = traceback.format_exc()
        return f"Error Occurred! {ve}\n{tb}"

    except KeyError as ke:
        tb = traceback.format_exc()
        return f"Error Occurred! {ke}\n{tb}"

    except Exception as e:
        tb = traceback.format_exc()
        return f"Error Occurred! {e}\n{tb}"


def show_predict_page():
    st.title("Employee Attrition Prediction Page")

    st.write("""### Please provide the information on the attributes of the employee""")

    salary_options = ['low', 'high']

    with st.form(key="employee_attribute"):
        satisfaction_level = st.text_input(label="Satisfaction Level")
        last_evaluation = st.text_input(label="Last Evaluation")
        number_project = st.text_input(label="Number of Projects")
        average_monthly_hours = st.text_input(label="Average Monthly Hours")
        time_spend_company = st.text_input(label="Years Spent in the Company")
        work_accident = st.text_input(label="Work Accident")
        promotion_last_5years = st.text_input(label="Promotion in the Last 5 years")
        salary = st.selectbox(label="Salary", options=salary_options, index=None)

        submit_button = st.form_submit_button(label="Submit Employee Information")

        if submit_button:
            st.write("You have submitted the Employee Information")
            form_submission = [0, satisfaction_level, last_evaluation, number_project, average_monthly_hours,
                               time_spend_company,
                               work_accident, promotion_last_5years, salary]

            result = single_prediction_streamlit(form_submission)
            st.subheader(result)
