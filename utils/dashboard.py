import os
import shutil
from io import BytesIO
from pathlib import Path

import bokeh
import boto3
import numpy as np
import pandas as pd
from PIL import Image
from bokeh.models import ColumnDataSource, Div
from bokeh.palettes import Blues4
from bokeh.plotting import figure
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix


def prepare_data_dashboard(session_dfs,
                           model_params,
                           split_name,
                           foldnum,
                           session_id,
                           dashboard_plots=["cm", "roc_curve_ovr"]):

    exp_name = model_params["experiment-name"]

    dashboard_session_path = f"./dashboard_data/{exp_name}/{split_name}/{foldnum}/{session_id}"

    if os.path.exists(dashboard_session_path):
        shutil.rmtree(dashboard_session_path)

    os.makedirs(dashboard_session_path, exist_ok=True)

    classes_names = model_params["classes-names"]

    preds = session_dfs[1]
    np.savetxt(os.path.join(dashboard_session_path, 'predictions.csv'), preds, delimiter=',')
    # labels.to_csv(os.path.join(dashboard_session_path, 'predictions.csv'), index=False, encoding='utf-8')

    # raw_data = session_dfs[0]
    # raw_data.to_csv(os.path.join(dashboard_session_path,'raw_data.csv'), index=False, encoding='utf-8')

    y_true, y_pred = preds[:, [0, 1]], preds[:, [2, 3]]

    for plot in dashboard_plots:
        if plot == "cm":

            cm_matrix = confusion_matrix(np.argmax(y_true, 1), np.argmax(y_pred, 1))
            file_name = os.path.join(dashboard_session_path, "cm.png")
            vis.plot_confusion_matrix(cm_matrix, classes_names, title=f"test_session:{session_id}", save_path=file_name)

        elif plot == "roc_curve_ovr":
            # create a dict from the class names
            classes_names_dict = {}
            for i, b in enumerate(classes_names):
                classes_names_dict[b] = i

            file_name = os.path.join(dashboard_session_path, "roc_curve_ovr.png")
            vis.plot_roc_curve_ovr(y_true, y_pred, classes_names_dict, save_path=file_name)


def scrape_files_from_folder(data_path):
    session_dict = {}
    exp_names = []
    for root, dirs, files in os.walk(data_path):

        for file in files:
            # append the file name to the list
            keys = root.split(os.sep)

            session_id = keys[-1]
            fold_num = f"FoldNum-{keys[-2]}"
            split_name = keys[-3]
            exp_names.append(keys[-4])

            if split_name not in session_dict:
                session_dict[split_name] = {}
            if fold_num not in session_dict[split_name]:
                session_dict[split_name][fold_num] = {}
            if session_id not in session_dict[split_name][fold_num]:
                session_dict[split_name][fold_num][session_id] = {}

            file_path = os.path.join(root, file)

            if file_path.endswith("png") and not file_path.startswith('.'):
                base_name = Path(file_path).stem
                session_dict[split_name][fold_num][session_id][base_name] = file_path
            elif file_path.endswith("csv") and not file_path.startswith('.'):
                session_dict[split_name][fold_num][session_id]["prediction"] = file_path
            elif file_path.endswith("json") and not file_path.startswith('.'):
                session_dict[split_name][fold_num][session_id]["Hparams"] = file_path

    return session_dict, exp_names[0]


def load_raw_session_data(session_id, exp_name, classses_names, pred_path):
    s3client = boto3.client('s3')
    prefix = f"data/{exp_name}/{session_id}/X.npz"
    obj = s3client.get_object(Bucket="true-machinelearning-training-data", Key=prefix)
    with np.load(BytesIO(obj['Body'].read()), allow_pickle=True) as d:
        X = np.array(d["X"])

    sensor_names = ["acc", "vel", "angvel", "mag", "accu"]
    num_features = X.shape[-1]
    x_raw = X.reshape(-1, num_features)

    y = genfromtxt(pred_path, delimiter=',', dtype="float")

    def _calculate_timestep(y_t_center):
        f = 20
        t_step = 1
        stride = t_step * f
        window_size = 10

        delta_step_half = (window_size * f) / 2
        start_step = int((y_t_center * f - delta_step_half))
        stop_step = int((y_t_center * f + delta_step_half))
        return np.arange(start_step, stop_step, 1)

    res = []
    for y_intervall in y:
        t_inter = _calculate_timestep(y_intervall[-1])
        y_inter = np.tile(y_intervall[0:-1], (200, 1))
        res_inter = np.hstack((y_inter, np.expand_dims(t_inter, axis=1)))

        res.append(res_inter)

    pred_all = np.array(res).reshape(-1, 5)

    colum_names = classses_names + ["pred-" + n for n in classses_names] + ["time"]
    y_all = pd.DataFrame(pred_all, columns=colum_names).sort_values(by="time", ascending=True)

    x_df = pd.DataFrame(x_raw, columns=sensor_names)
    x_df["time"] = np.array(y_all["time"])

    return x_df, y_all


def load_raw_session_data_1(session_id, exp_name, classses_names, pred_path):
    s3client = boto3.client('s3')
    prefix = f"data/{exp_name}/{session_id}/X.npz"
    obj = s3client.get_object(Bucket="true-machinelearning-training-data", Key=prefix)
    with np.load(BytesIO(obj['Body'].read()), allow_pickle=True) as d:
        X = np.array(d["X"])

    sensor_names = ["acc", "vel", "angvel", "mag", "accu"]
    num_features = X.shape[-1]
    x_raw = X.reshape(-1, num_features)

    y = genfromtxt(pred_path, delimiter=',', dtype="float")

    colum_names = classses_names + ["pred-" + n for n in classses_names] + ["time"]
    y_all = pd.DataFrame(y, columns=colum_names).sort_values(by="time", ascending=True)

    x_df = pd.DataFrame(x_raw, columns=sensor_names)
    x_df["time"] = np.array(x_df.index)

    return x_df, y_all


def load_raw_session_data_2(session_id, exp_name, classses_names, pred_path):
    y = genfromtxt(pred_path, delimiter=',', dtype="float")

    colum_names = classses_names + ["pred-" + n for n in classses_names] + ["time"]
    y_all = pd.DataFrame(y, columns=colum_names).sort_values(by="time", ascending=True)

    return y_all


def attach_imge_bokeh(imge_path):
    im = Image.open(imge_path).convert('RGBA')
    xdim, ydim = im.size
    # print("Dimensions: ({xdim}, {ydim})".format(**locals()))
    # Create an array representation for the image `img`, and an 8-bit "4
    # layer/RGBA" version of it `view`.
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))

    # Copy the RGBA image into view, flipping it so it comes right-side up
    # with a lower-left origin
    view[:, :, :] = np.flipud(np.asarray(im))

    # Display the 32-bit RGBA image
    dim = max(xdim, ydim)

    return img, dim, xdim, ydim


def bokeh_imshow(im, color_mapper=None, plot_height=400):
    """
    Display an image in a Bokeh figure.
    """
    # Get shape
    n, m = im.shape

    # Set up figure with appropriate dimensions
    plot_width = int(m / n * plot_height)
    p = bokeh.plotting.figure(plot_height=plot_height, plot_width=plot_width,
                              x_range=[0, m], y_range=[0, n],
                              tools='pan,box_zoom,wheel_zoom,reset,resize')

    # Set color mapper; we'll do Viridis with 256 levels by default
    if color_mapper is None:
        color_mapper = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256))

    # Display the image
    im_bokeh = p.image(image=[im[::-1, :]], x=0, y=0, dw=m, dh=n,
                       color_mapper=color_mapper)

    return p


def get_dataset():
    pass


def make_plot():
    pass


def update_plot(attrname, old, new):
    # city = city_select.value
    # plot.title.text = "Weather data for " + cities[city]['title']
    #
    # src = get_dataset(df, cities[city]['airport'], distribution_select.value)
    # source.data.update(src.data)
    pass


def make_image():
    pass


class Bokeh_Eval_Dashboard():

    def __init__(self, experiment_name, classses_names, sessions_dict):
        self.experiment_name = experiment_name
        self.classses_names = classses_names
        self.sessions_dict = sessions_dict

        self.sensor_names = ["acc", "vel", "angvel", "mag", "accu"]

        self.pred_series = figure(x_axis_label="Snippets t_center of size 10s", y_axis_label="[_]", width=2000,
                                  height=300)

        self.true_series = figure(x_axis_label="Snippets t_center of size 10s", y_axis_label="[_]", width=2000,
                                  height=300)

        self.roc_image = figure(x_range=(0, 1200), y_range=(0, 1200),  # Specifying xdim/ydim isn't quire right :-(
                                width=800, height=800, )

        self.cm_image = figure(x_range=(0, 1200), y_range=(0, 1200),  # Specifying xdim/ydim isn't quire right :-(
                               width=800, height=800, )

        self.div = Div(text="text",
                       style={'font-size': '100%', 'color': 'blue'},
                       width=100)

    def get_datasets(self, session_id, sensor_name, split_name, fold_num):

        self.split_name = split_name
        self.fold_num = fold_num
        self.cm_img_path = self.sessions_dict[self.split_name][self.fold_num][session_id]["cm"]
        self.roc_img_path = self.sessions_dict[self.split_name][self.fold_num][session_id]["roc_curve_ovr"]
        self.pred_path = self.sessions_dict[self.split_name][self.fold_num][session_id]["prediction"]
        x_df, y_df = load_raw_session_data_1(session_id=session_id,
                                             exp_name=self.experiment_name,
                                             classses_names=self.classses_names,
                                             pred_path=self.pred_path)

        x_raw_source = ColumnDataSource(dict(x=list(x_df.index), y=list(x_df[sensor_name])))
        y_source = ColumnDataSource(data=y_df)
        self.id = session_id

        return x_raw_source, y_source

    def plot_images(self):
        # ----------------------------------------------------
        #      figure
        # ----------------------------------------------------
        # im_1 = Image.open(self.roc_img_path).convert('RGBA')
        self.roc_image.renderers = []
        img, dim, xdim, ydim = attach_imge_bokeh(self.roc_img_path)
        self.roc_image.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)
        self.roc_image.axis.visible = False
        self.roc_image.title.text = "ROC Curve One vs. Rest"
        # ----------------------------------------------------
        #      figure
        # ----------------------------------------------------
        # im_2 = Image.open(self.cm_img_path).convert('RGBA')
        self.cm_image.renderers = []
        img, dim, xdim, ydim = attach_imge_bokeh(self.cm_img_path)
        self.cm_image.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)
        self.cm_image.axis.visible = False
        self.cm_image.title.text = "Confusion Matrix"

    def set_title(self):
        template = ("""
              <h1 style="text-align: center"> Experiment: {exp_name}/ {session_id} </h1>
              <h2>Split : {split_name}</h2>
              <h3>Fold number : {fold_num}</h3>
              """)
        # # # initial text
        text = template.format(exp_name=self.experiment_name,
                               session_id=self.id,
                               split_name=self.split_name,
                               fold_num=self.fold_num)
        self.div.text = text

    def plot_prediction(self, y_source):
        # ----------------------------------------------------
        #      Prediction
        # ----------------------------------------------------
        # self.pred_series = figure(x_axis_label="Snippets t_center of size 10s", y_axis_label="[_]", width=2000,
        #                           height=300)
        self.pred_series.renderers = []
        colors = ["green", "red"]
        for class_name, color in zip(self.classses_names, colors):
            self.pred_series.line(x='time', y=f"pred-{class_name}", source=y_source, color=color, line_width=2)

    def plot_true(self, y_source):
        # ----------------------------------------------------
        #      True
        # ----------------------------------------------------
        self.true_series.renderers = []
        colors = ["green", "red"]
        for class_name, color in zip(self.classses_names, colors):
            self.true_series.line(x='time', y=class_name, source=y_source, color=color, line_width=2)
        self.true_series.x_range = self.pred_series.x_range
        self.true_series.y_range = self.pred_series.y_range

    def update(self, y_source):

        y_source.data.update(y_source.data)
        self.plot_prediction(y_source)
        self.plot_true(y_source)
        self.plot_images()
        self.set_title()
