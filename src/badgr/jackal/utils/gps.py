from io import BytesIO
import numpy as np
import os
from PIL import Image
import requests
import sys
import utm


def latlong_to_utm(latlong):
    """
    :param latlong: latlong or list of latlongs
    :return: utm (easting, northing)
    """
    latlong = np.array(latlong)
    if len(latlong.shape) > 1:
        return np.array([latlong_to_utm(p) for p in latlong])

    easting, northing, _, _ = utm.from_latlon(*latlong)
    return np.array([easting, northing])


def utm_to_latlong(u, zone_number=10, zone_letter='S'):
    u = np.array(u)
    if len(u.shape) > 1:
        return np.array([utm_to_latlong(u_i, zone_number=zone_number, zone_letter=zone_letter) for u_i in u])


    easting, northing = u
    return utm.to_latlon(easting, northing, zone_number=zone_number, zone_letter=zone_letter)


EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * np.pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

class GPSPlotter(object):
    """
    Lots taken from: https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python
    """

    def __init__(self,
                 nw_latlong=(37.915585, -122.336621),
                 se_latlong=(37.914514, -122.334064),
                 zoom=19,
                 satellite_img_fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rfs_satellite.png'),
                 google_maps_api_key=None):
        self.nw_latlong = nw_latlong
        self.se_latlong = se_latlong
        self._zoom = zoom
        self._satellite_img_fname = satellite_img_fname

        if not os.path.exists(self._satellite_img_fname):
            assert google_maps_api_key is not None
            self._save_satellite_image(google_maps_api_key)
        assert os.path.exists(self._satellite_img_fname)
        self._img = np.array(Image.open(self._satellite_img_fname))

        x_c0, y_c0 = self.latlong_to_pixels(*self.nw_latlong)
        x_c1, y_c1 = self.latlong_to_pixels(*self.se_latlong)

        self._bottom_left_pixel = np.array([min(x_c0, x_c1), min(y_c0, y_c1)])
        self._top_right_pixel = np.array([max(x_c0, x_c1), max(y_c0, y_c1)])

        self._plt_latlong_and_compass_bearing_dict = dict()
        self._plt_utms_dicts = dict()

    @property
    def satellite_image(self):
        return self._img.copy()

    def latlong_to_coordinate(self, latlong):
        latlong = np.array(latlong)
        if len(latlong.shape) > 1:
            return np.array([self.latlong_to_coordinate(l_i) for l_i in latlong])

        pixel_absolute = np.array(self.latlong_to_pixels(*latlong))
        # assert np.all(pixel_absolute >= self._bottom_left_pixel) and np.all(pixel_absolute <= self._top_right_pixel)
        pixel = pixel_absolute - self._bottom_left_pixel
        return pixel

    def utm_to_coordinate(self, utm):
        return self.latlong_to_coordinate(utm_to_latlong(utm))

    def compass_bearing_to_dcoord(self, compass_bearing):
        offset = -np.pi / 2.
        dx, dy = np.array([np.cos(compass_bearing + offset), -np.sin(compass_bearing + offset)])
        return np.array([dx, dy])

    def plot_latlong_and_compass_bearing(self, ax, latlong, compass_bearing, blit=True, color='r'):
        return self.plot_utm_and_compass_bearing(ax, latlong_to_utm(latlong), compass_bearing, blit=blit, color=color)

    def plot_utm_and_compass_bearing(self, ax, utm, compass_bearing,
                                     blit=True, color='r', arrow_length=15, arrow_head_width=10):
        x, y = self.utm_to_coordinate(utm)
        dx, dy = arrow_length * self.compass_bearing_to_dcoord(compass_bearing)

        if ax not in self._plt_latlong_and_compass_bearing_dict:
            imshow = ax.imshow(np.flipud(self._img), origin='lower')
            arrow = ax.arrow(x, y, dx, dy, color=color, head_width=arrow_head_width)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self._plt_latlong_and_compass_bearing_dict[ax] = (imshow, arrow)
        else:
            imshow, arrow = self._plt_latlong_and_compass_bearing_dict[ax]
            arrow.remove()
            arrow = ax.arrow(x, y, dx, dy, color=color, head_width=arrow_head_width)
            self._plt_latlong_and_compass_bearing_dict[ax] = (imshow, arrow)

            if blit:
                ax.draw_artist(ax.patch)
                ax.draw_artist(imshow)
                ax.draw_artist(arrow)
                ax.figure.canvas.blit(ax.bbox)

    def plot_latlong_density(self, ax, latlongs, include_image=True, include_colorbar=False, **kwargs):
        xy = filter(lambda x: x is not None,
                    [self.latlong_to_coordinate(latlong) for latlong in latlongs])
        xy = np.array(list(xy))

        if include_image:
            ax.imshow(np.flipud(np.array(Image.fromarray(self._img).convert('L'))), cmap='gray', origin='lower')

        gridsize = 20
        hb = ax.hexbin(
            xy[:, 0], xy[:, 1],
            gridsize=gridsize,
            **kwargs
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if include_colorbar:
            cb = ax.figure.colorbar(hb, ax=ax)

    #######################
    ### Google maps API ###
    #######################

    def latlong_to_pixels(self, lat, lon):
        mx = (lon * ORIGIN_SHIFT) / 180.0
        my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
        my = (my * ORIGIN_SHIFT) / 180.0
        res = INITIAL_RESOLUTION / (2 ** self._zoom)
        px = (mx + ORIGIN_SHIFT) / res
        py = (my + ORIGIN_SHIFT) / res
        return px, py

    def pixels_to_latlong(self, px, py):
        res = INITIAL_RESOLUTION / (2 ** self._zoom)
        mx = px * res - ORIGIN_SHIFT
        my = py * res - ORIGIN_SHIFT
        lat = (my / ORIGIN_SHIFT) * 180.0
        lat = 180 / np.pi * (2 * np.arctan(np.exp(lat * np.pi / 180.0)) - np.pi / 2.0)
        lon = (mx / ORIGIN_SHIFT) * 180.0
        return lat, lon

    def _save_satellite_image(self, google_maps_api_key):

        ullat, ullon = self.nw_latlong
        lrlat, lrlon = self.se_latlong

        # Set some important parameters
        scale = 1
        maxsize = 640

        # convert all these coordinates to pixels
        ulx, uly = self.latlong_to_pixels(ullat, ullon)
        lrx, lry = self.latlong_to_pixels(lrlat, lrlon)

        # calculate total pixel dimensions of final image
        dx, dy = lrx - ulx, uly - lry

        # calculate rows and columns
        cols, rows = int(np.ceil(dx / maxsize)), int(np.ceil(dy / maxsize))

        # calculate pixel dimensions of each small image
        bottom = 120
        largura = int(np.ceil(dx / cols))
        altura = int(np.ceil(dy / rows))
        alturaplus = altura + bottom

        # assemble the image from stitched
        final = Image.new("RGB", (int(dx), int(dy)))
        for x in range(cols):
            for y in range(rows):
                dxn = largura * (0.5 + x)
                dyn = altura * (0.5 + y)
                latn, lonn = self.pixels_to_latlong(ulx + dxn, uly - dyn - bottom / 2)
                position = ','.join((str(latn), str(lonn)))
                print(x, y, position)
                urlparams = {'center': position,
                             'zoom': str(self._zoom),
                             'size': '%dx%d' % (largura, alturaplus),
                             'maptype': 'satellite',
                             'sensor': 'false',
                             'scale': scale}
                if google_maps_api_key is not None:
                    urlparams['key'] = google_maps_api_key

                url = 'http://maps.google.com/maps/api/staticmap'
                try:
                    response = requests.get(url, params=urlparams)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(e)
                    sys.exit(1)

                im = Image.open(BytesIO(response.content))
                final.paste(im, (int(x * largura), int(y * altura)))

        final.save(self._satellite_img_fname)
