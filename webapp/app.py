import flask
import pickle
import pandas as pd

with open(f'model/stu_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        absence = flask.request.form['StudentAbsenceDays']
        survey = flask.request.form['ParentAnsweringSurvey']
        relation = flask.request.form['Relation']
        satisfaction = flask.request.form['ParentschoolSatisfaction']
        announcements = flask.request.form['AnnouncementsView']
        raised = flask.request.form['raisedhands']
        resources = flask.request.form['VisITedResources']
        discussion = flask.request.form['Discussion']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[absence, survey, relation, satisfaction,
                                         announcements, raised, resources, discussion]],
                                       columns=['StudentAbsenceDays','ParentAnsweringSurvey',
                                                'Relation', 'ParentschoolSatisfaction', 
                                                'AnnouncementsView', 'raisedhands',
                                                'VisITedResources', 'Discussion'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'StudentAbsenceDays':absence,
                                                     'ParentAnsweringSurvey':survey,
                                                     'Relation':relation,
                                                     'ParentschoolSatisfaction':satisfaction,
                                                     'AnnouncementsView':announcements,
                                                     'raisedhands':raised,
                                                     'VisITedResources':resources,
                                                     'Discussion':discussion},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()