"""
Helper tools to show images from the VOC dataset.
"""
import numpy as np
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches, patheffects

from utils import open_image, to_voc, valid_box



class VOCPlotter:
    """A helper class to visualize samples from VOC dataset.

    The dataset is expected to contain images, bounding boxes, and object
    classes. Then the class can be used to show these images and their
    annotations.
    """
    def __init__(self, id2cat=None, **fig_kwargs):
        self.id2cat = id2cat
        self.fig = None
        self.fig_kwargs = fig_kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.pause(0.001)
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def plot_boxes(self, images, boxes, classes, dims=(3, 4)):
        if self.fig is not None:
            plt.close(self.fig)

        fig, axes = plt.subplots(*dims, **self.fig_kwargs)
        n_colors = min(12, len(np.unique(classes)))
        cmap = get_cmap(n_colors)
        colors_list = [cmap(float(x)) for x in range(n_colors)]

        for i, ax in enumerate(axes.flat):
            ax.axis('off')
            image = images[i]
            image_classes = [c for c in classes[i] if c > 0]
            image_boxes = [b for b in boxes[i].reshape(-1, 4) if valid_box(b)]
            if image.shape[0] == 3 and len(image.shape) == 3:
                image = image.transpose(1, 2, 0)
            self.plot_image(image, ax=ax)
            for j, (box, target) in enumerate(zip(image_boxes, image_classes)):
                if box[2] <= 0:
                    continue
                box = to_voc(box)
                color = colors_list[j % n_colors]
                if self.id2cat:
                    target = self.id2cat.get(target, target)
                add_rect(ax, box, color=color)
                add_text(ax, box[:2], f'{j}: {target}', color=color)

        self.fig = fig

    def plot_image(self, image, grid=False, n_cells=8, ax=None):
        if not ax:
            if self.fig is not None:
                self.fig.close()
            fig, ax = plt.subplots(**self.fig_kwargs)
            self.fig = fig

        ax.imshow(image)
        if grid:
            width, height = image.shape[:2]
            ax.set_xticks(np.linspace(0, width, n_cells))
            ax.set_yticks(np.linspace(0, height, n_cells))
            ax.grid()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return ax


class ImagePlotter:
    """Helper class to visualize VOC dataset images and bounding boxes."""

    def __init__(self, root, annotations, files, categories, **fig_kwargs):
        self.root = root
        self.annotations = annotations
        self.files = files
        self.categories = categories
        self.fig_kwargs = fig_kwargs

    def show(self, index, ax=None):
        """
        Plots an image and bounding box with specific ID value from the VOC
        dataset and shows matplotlib interface with the image.
        """
        self.draw(index, ax)
        plt.show()

    def draw(self, index, ax=None):
        """
        Plots and image and bounding box with specific ID value from the VOC
        dataset, and returns the axes object.
        """
        annotation = self.annotations[index]
        image = open_image(self.root / self.files[index])
        ax = self.draw_image(image, ax=ax)
        for bbox, category in annotation:
            bbox = to_voc(bbox)
            classes = self.categories[category]
            add_rect(ax, bbox)
            add_text(ax, bbox[:2], classes, size=16)
        return ax

    def show_images(self, images, targets, class_names, dims=(3, 4),
                    figsize=(12, 12), grid=True):

        fig, axes = plt.subplots(*dims, figsize=figsize)
        for i, ax in enumerate(axes.flat):
            image = images[i]
            [non_zero] = np.nonzero(targets[i] > 0.4)
            self.draw_image(image, ax=ax, grid=grid)
            if len(non_zero) > 0:
                classes = '\n'.join([class_names[index] for index in non_zero])
                add_text(ax, (0, 0), classes)
            else:
                classes = '<NONE>'
                add_text(ax, (0, 0), classes, color='salmon')
        plt.tight_layout()
        plt.show()

    def show_ground_truth(self, images, boxes, classes, dims=(3, 4),
                          figsize=(12, 12)):

        n_colors = 12
        cmap = get_cmap(n_colors)
        colors_list = [cmap(float(x)) for x in range(n_colors)]

        fig, axes = plt.subplots(*dims, figsize=figsize)
        for i, ax in enumerate(axes.flat):
            image = images[i]
            image_classes = classes[i]
            image_boxes = [box for box in boxes[i].reshape(-1, 4)]
            self.draw_image(image, ax=ax)
            for j, (box, target) in enumerate(zip(image_boxes, image_classes)):
                if box[2] <= 0:
                    continue
                box = to_voc(box)
                color = colors_list[j % n_colors]
                add_rect(ax, box, color=color)
                add_text(ax, box[:2], f'{j}: {target}', color=color)

    def draw_image(self, image, grid=False, n_cells=8, ax=None):
        if not ax:
            fig, ax = plt.subplots(**self.fig_kwargs)
        ax.imshow(image)
        if grid:
            width, height = image.shape[:2]
            ax.set_xticks(np.linspace(0, width, n_cells))
            ax.set_yticks(np.linspace(0, height, n_cells))
            ax.grid()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return ax


def get_cmap(n):
    color_norm = mcolors.Normalize(vmin=0, vmax=n - 1)
    return cmx.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba


def add_rect(ax, bbox, outline=4, color='white'):
    """Adds a stroke rectangle to the axes."""

    rect = patches.Rectangle(
        bbox[:2], *bbox[-2:], fill=False, edgecolor=color, lw=2)
    patch = ax.add_patch(rect)
    add_outline(patch, outline)


def add_text(ax, xy, text, size=14, outline=1, color='white'):
    """Adds a text object to the axes."""

    text = ax.text(
        *xy, text, va='top', color=color, fontsize=size, weight='bold')
    add_outline(text, outline)


def add_outline(obj, lw=4):
    """Adds outline effect to the graphical object."""

    effects = [
        patheffects.Stroke(linewidth=lw, foreground='black'),
        patheffects.Normal()]
    obj.set_path_effects(effects)
