from pyproj import Transformer


class CoordTransformer:
    def __init__(self):
        # Define transformers for local and global coordinate systems
        self.transformer_lonlat_global = Transformer.from_crs(
            "epsg:4326", "epsg:28992", always_xy=True
        )
        self.transformer_global_lonlat = Transformer.from_crs(
            "epsg:28992", "epsg:4326", always_xy=True
        )
        self.transformer_global_utm = Transformer.from_crs(
            "epsg:28992", "epsg:32631", always_xy=True
        )
        self.transformer_utm_global = Transformer.from_crs(
            "epsg:32631", "epsg:28992", always_xy=True
        )

    def t_lonlat_global(self, lon, lat):
        return self.transformer_lonlat_global.transform(lon, lat)

    def t_global_nl(self, x, y):
        return self.transformer_global_lonlat.transform(x, y)

    def t_utm_global(self, x, y):
        return self.transformer_utm_global.transform(x, y)

    def t_global_utm(self, x, y):
        return self.transformer_global_utm.transform(x, y)
