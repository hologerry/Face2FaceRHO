import os
import sys
import time

import numpy as np
import visdom

from PIL import Image

from . import html, util


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer:
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(
                server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True
            )
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, "web")
            self.img_dir = os.path.join(self.web_dir, "images")
            print("create web directory %s..." % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write("================ Training Loss (%s) ================\n" % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print(
            "\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n"
        )
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, aspect_ratio=1.0, width=256):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[2:4]
                # h, w = 256,256
                height = int(width * h / float(w))
                h = height
                w = width
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (
                    w,
                    h,
                )
                title = self.name
                label_html = ""
                label_html_row = ""
                images = []
                idx = 0
                for label, image in visuals.items():
                    if label == "drv_face_mask":
                        image_numpy = util.show_mask(image)
                    else:
                        image_numpy = util.tensor2im(image, np.uint8)
                    image_numpy = np.array(Image.fromarray(image_numpy).resize((h, w)))
                    image_numpy = image_numpy.transpose([2, 0, 1])
                    label_html_row += "<td>%s</td>" % label
                    images.append(image_numpy)
                    idx += 1
                    if idx % ncols == 0:
                        label_html += "<tr>%s</tr>" % label_html_row
                        label_html_row = ""
                white_image = np.ones_like(image_numpy) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += "<td></td>"
                    idx += 1
                if label_html_row != "":
                    label_html += "<tr>%s</tr>" % label_html_row
                # pane col = image row
                try:
                    self.vis.images(
                        images, nrow=ncols, win=self.display_id + 1, padding=2, opts=dict(title=title + " images")
                    )
                    label_html = "<table>%s</table>" % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2, opts=dict(title=title + " labels"))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label), win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, "epoch%.3d_%s.png" % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, "Experiment name = %s" % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header("epoch [%d]" % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = "epoch%.3d_%s.png" % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, "plot_data"):
            self.plot_data = {"X": [], "Y": [], "legend": list(losses.keys())}
        self.plot_data["X"].append(epoch + counter_ratio)
        self.plot_data["Y"].append([losses[k] for k in self.plot_data["legend"]])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data["X"])] * len(self.plot_data["legend"]), 1),
                Y=np.array(self.plot_data["Y"]),
                opts={
                    "title": self.name + " loss over time",
                    "legend": self.plot_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": "loss",
                },
                win=self.display_id,
            )
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = "(epoch: %d, iters: %d, time: %.3f, data: %.3f) " % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += "%s: %.3f " % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)

    # losses: dictionary of error labels and values
    def plot_current_validation_error(self, epoch, counter_ratio, losses):
        if not hasattr(self, "plot_validation_data"):
            self.plot_validation_data = {"X": [], "Y": [], "legend": list(losses.keys())}
        self.plot_validation_data["X"].append(epoch + counter_ratio)
        self.plot_validation_data["Y"].append([losses[k] for k in self.plot_validation_data["legend"]])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_validation_data["X"])] * len(self.plot_validation_data["legend"]), 1),
                Y=np.array(self.plot_validation_data["Y"]),
                opts={
                    "title": self.name + " validation error over time",
                    "legend": self.plot_validation_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": "error",
                },
                win=self.display_id + 1,
            )
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()
