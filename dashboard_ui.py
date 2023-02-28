import boto3
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Select

import utils.dashboard as db

data_path = "dashboard_data"

s3client = boto3.client('s3')

session_dict, exp_name = db.scrape_files_from_folder(data_path="dashboard_data")

split_name = "test"
fold_num = "FoldNum-0"
sessions = session_dict[split_name][fold_num]
session_ids = list(sessions)
sensor_names = ["acc", "vel", "angvel", "mag", "accu"]

classses_names = ["indoors", "outdoors"]


def update_data(attrname, old, new):
    idx = select_session.value
    print(idx)
    _, y_source = dash.get_datasets(idx, "accu", split_name, fold_num)
    y_source.data.update(y_source.data)
    dash.update(y_source)


# Interactive
select_session = Select(title="Session id", value=session_ids[0], options=session_ids)
select_sensor = Select(title="Sensor name", value="accu", options=sensor_names)

select_sensor.on_change('value', update_data)
select_session.on_change('value', update_data)

dash = db.Bokeh_Eval_Dashboard(experiment_name=exp_name,
                               classses_names=classses_names,
                               sessions_dict=session_dict)

_, y_source = dash.get_datasets(session_ids[0], "accu", split_name, fold_num)
dash.update(y_source)

# Create layouts
app_title = dash.div
widgets = column(select_session, select_sensor)
main_row = row(widgets)
series = column(dash.pred_series, dash.true_series)
images = row(dash.roc_image, dash.cm_image)
layout = column(app_title, main_row, series, images, sizing_mode='scale_width')

curdoc().add_root(layout)
curdoc().title = "Interactive Graph"
