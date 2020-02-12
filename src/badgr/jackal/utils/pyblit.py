from matplotlib.collections import LineCollection as PltLineCollection
from matplotlib.patches import Polygon as PltPolygon
from matplotlib.collections import PatchCollection
import numpy as np


class Axis(object):

    def __init__(self, ax, pyblits):
        self.ax = ax
        self._pyblits = pyblits
        self._is_first_draw = True

    def draw(self):
        if not self._is_first_draw:
            self.ax.draw_artist(self.ax.patch)
            for pyblit in self._pyblits:
                for artist in pyblit.artists:
                    self.ax.draw_artist(artist)
            self.ax.figure.canvas.blit(self.ax.bbox)
        self._is_first_draw = False


class Arrow(object):

    def __init__(self, ax):
        self._ax = ax
        self._arrow = None

    @property
    def artists(self):
        return [self._arrow]

    def draw(self, x, y, dx, dy, **kwargs):
        if self._arrow is not None:
            self._arrow.remove()

        self._arrow = self._ax.arrow(x, y, dx, dy, **kwargs)


class Bar(object):

    def __init__(self, ax):
        self._ax = ax
        self._rects = None

    @property
    def artists(self):
        return self._rects

    def draw(self, x, y, **kwargs):
        if self._rects is None:
            self._rects = self._ax.bar(x, y, **kwargs).get_children()
        else:
            for y_i, rect_i in zip(y, self._rects):
                rect_i.set_height(y_i)


class Barh(object):

    def __init__(self, ax):
        self._ax = ax
        self._rects = None

    @property
    def artists(self):
        return self._rects

    def draw(self, y, width, **kwargs):
        if self._rects is None:
            self._rects = self._ax.barh(y, width, **kwargs).get_children()
        else:
            for width_i, rect_i in zip(width, self._rects):
                rect_i.set_width(width_i)


class Imshow(object):

    def __init__(self, ax):
        self._ax = ax
        self._imshow = None

    @property
    def artists(self):
        return [self._imshow]

    def draw(self, im, **kwargs):
        if self._imshow is None:
            self._imshow = self._ax.imshow(im, **kwargs)
            self._ax.get_xaxis().set_visible(False)
            self._ax.get_yaxis().set_visible(False)
        else:
            self._imshow.set_data(im)


class Legend(object):

    def __init__(self, ax):
        self._ax = ax
        self._legend = None

    @property
    def artists(self):
        return [self._legend]

    def draw(self, **kwargs):
        if self._legend is None:
            self._legend = self._ax.legend(**kwargs)
        else:
            pass


class Line(object):

    def __init__(self, ax):
        self._ax = ax
        self._line = None

    @property
    def artists(self):
        return [self._line]

    def draw(self, x, y, **kwargs):
        if self._line is None:
            self._line = self._ax.plot(x, y, **kwargs)[0]
        else:
            self._line.set_xdata(x)
            self._line.set_ydata(y)
            if 'color' in kwargs:
                self._line.set_color(kwargs['color'])


class LineCollection(object):

    def __init__(self, ax):
        self._ax = ax
        self._lc = None

    @property
    def artists(self):
        return [self._lc]

    def draw(self, x, y, **kwargs):
        xy = np.stack([x, y], axis=1)
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])

        if self._lc is None:
            self._lc = PltLineCollection(segments)
            self._ax.add_collection(self._lc)
        else:
            self._lc.set_segments(segments)
        if 'color' in kwargs:
            self._lc.set_color(kwargs['color'])


class BatchLineCollection(object):

    def __init__(self, ax):
        self._ax = ax
        self._lc = None

    @property
    def artists(self):
        return [self._lc]

    def draw(self, x, y, **kwargs):
        segments = []
        for x_i, y_i in zip(x, y):
            xy_i = np.stack([x_i, y_i], axis=1)
            xy_i = xy_i.reshape(-1, 1, 2)
            segments_i = np.hstack([xy_i[:-1], xy_i[1:]])
            segments.append(segments_i)
        segments = np.concatenate(segments, axis=0)

        if self._lc is None:
            self._lc = PltLineCollection(segments)
            self._ax.add_collection(self._lc)
        else:
            self._lc.set_segments(segments)
        if 'color' in kwargs:
            self._lc.set_color(np.reshape(kwargs['color'], [len(segments), -1]))
        if 'linewidth' in kwargs:
            self._lc.set_linewidth(kwargs['linewidth'])
        self._lc.set_joinstyle('round')
        self._lc.set_capstyle('round')


class Polygon(object):

    def __init__(self, ax):
        self._ax = ax
        self._patch = None

    @property
    def artists(self):
        return [self._patch]

    def draw(self, points, **kwargs):
        if self._patch is not None:
            self._patch.remove()
            self._patch = None
        polygon = PltPolygon(points, True)
        self._patch = PatchCollection([polygon], **kwargs)
        self._ax.add_collection(self._patch)


class Scatter(object):

    def __init__(self, ax):
        self._ax = ax
        self._path_collection = None

    @property
    def artists(self):
        return [self._path_collection]

    def draw(self, x, y, c, **kwargs):
        if self._path_collection is None:
            self._path_collection = self._ax.scatter(x, y, c=c, **kwargs)
        else:
            self._path_collection.set_offsets(np.vstack((x, y)).T)
            self._path_collection.set_facecolors(c)


class Text(object):

    def __init__(self, ax):
        self._ax = ax
        self._text = None

    @property
    def artists(self):
        return [self._text]

    def draw(self, x, y, text, **kwargs):
        if self._text is None:
            self._text = self._ax.text(x, y, text, **kwargs)
        else:
            self._text.set_text(text)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    f, ax = plt.subplots(1, 1)

    lc = LineCollection(ax)
    axis = Axis(ax, [lc])

    is_shown = False
    while True:
        time.sleep(0.25)

        lc.draw(np.random.random(8), np.random.random(8), color=np.random.random([8, 3]))
        axis.draw()

        if not is_shown:
            plt.show(block=False)
            plt.pause(0.1)

