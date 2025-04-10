"""The is a temporary solution to resolve pyvista volume scalar bar issues."""

import numpy as np
from pyvista import Plotter
import sys
import warnings
import matplotlib
import pyvista
from pyvista.core.errors import MissingDataError, PyVistaDeprecationWarning
from pyvista.core.utilities.arrays import get_array, raise_not_matching
from pyvista.core.utilities.helpers import is_pyvista_dataset, wrap
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.plotting.colors import get_cmap_safe
from pyvista.plotting.mapper import (
    FixedPointVolumeRayCastMapper,
    GPUVolumeRayCastMapper,
    OpenGLGPUVolumeRayCastMapper,
    SmartVolumeMapper,
    UnstructuredGridVolumeRayCastMapper,
)
from pyvista.plotting.volume import Volume
from pyvista.plotting.volume_property import VolumeProperty


class volume_mixer:
    def add_volume(
        self,
        volume,
        scalars=None,
        clim=None,
        resolution=None,
        opacity="linear",
        n_colors=256,
        cmap=None,
        flip_scalars=False,
        reset_camera=None,
        name=None,
        ambient=None,
        categories=False,
        culling=False,
        multi_colors=False,
        blending="composite",
        mapper=None,
        scalar_bar_args=None,
        show_scalar_bar=None,
        annotations=None,
        pickable=True,
        preference="point",
        opacity_unit_distance=None,
        shade=False,
        diffuse=0.7,  # TODO: different default for volumes
        specular=0.2,  # TODO: different default for volumes
        specular_power=10.0,  # TODO: different default for volumes
        render=True,
        log_scale=False,
        **kwargs,
    ):

        # Handle default arguments

        # Supported aliases
        clim = kwargs.pop("rng", clim)
        cmap = kwargs.pop("colormap", cmap)
        culling = kwargs.pop("backface_culling", culling)

        if "scalar" in kwargs:
            raise TypeError(
                "`scalar` is an invalid keyword argument for `add_mesh`. Perhaps you mean `scalars` with an s?"
            )
        assert_empty_kwargs(**kwargs)

        if show_scalar_bar is None:
            show_scalar_bar = self._theme.show_scalar_bar or scalar_bar_args

        # Avoid mutating input
        if scalar_bar_args is None:
            scalar_bar_args = {}
        else:
            scalar_bar_args = scalar_bar_args.copy()
        # account for legacy behavior
        if "stitle" in kwargs:  # pragma: no cover
            # Deprecated on ..., estimated removal on v0.40.0
            warnings.warn(USE_SCALAR_BAR_ARGS, PyVistaDeprecationWarning)
            scalar_bar_args.setdefault("title", kwargs.pop("stitle"))

        if culling is True:
            culling = "backface"

        if mapper is None:
            # Default mapper choice. Overridden later if UnstructuredGrid
            mapper = self._theme.volume_mapper

        # only render when the plotter has already been shown
        if render is None:
            render = not self._first_time

        # Convert the VTK data object to a pyvista wrapped object if necessary
        if not is_pyvista_dataset(volume):
            if isinstance(volume, np.ndarray):
                volume = wrap(volume)
                if resolution is None:
                    resolution = [1, 1, 1]
                elif len(resolution) != 3:
                    raise ValueError("Invalid resolution dimensions.")
                volume.spacing = resolution
            else:
                volume = wrap(volume)
                if not is_pyvista_dataset(volume):
                    raise TypeError(
                        f"Object type ({type(volume)}) not supported for plotting in PyVista."
                    )
        else:
            # HACK: Make a copy so the original object is not altered.
            #       Also, place all data on the nodes as issues arise when
            #       volume rendering on the cells.
            volume = volume.cell_data_to_point_data()

        if name is None:
            name = f"{type(volume).__name__}({volume.memory_address})"

        if isinstance(volume, pyvista.MultiBlock):
            from itertools import cycle

            cycler = cycle(["Reds", "Greens", "Blues", "Greys", "Oranges", "Purples"])
            # Now iteratively plot each element of the multiblock dataset
            actors = []
            for idx in range(volume.GetNumberOfBlocks()):
                if volume[idx] is None:
                    continue
                # Get a good name to use
                next_name = f"{name}-{idx}"
                # Get the data object
                block = wrap(volume.GetBlock(idx))
                if resolution is None:
                    try:
                        block_resolution = block.GetSpacing()
                    except AttributeError:
                        block_resolution = resolution
                else:
                    block_resolution = resolution
                if multi_colors:
                    color = next(cycler)
                else:
                    color = cmap

                a = self.add_volume(
                    block,
                    resolution=block_resolution,
                    opacity=opacity,
                    n_colors=n_colors,
                    cmap=color,
                    flip_scalars=flip_scalars,
                    reset_camera=reset_camera,
                    name=next_name,
                    ambient=ambient,
                    categories=categories,
                    culling=culling,
                    clim=clim,
                    mapper=mapper,
                    pickable=pickable,
                    opacity_unit_distance=opacity_unit_distance,
                    shade=shade,
                    diffuse=diffuse,
                    specular=specular,
                    specular_power=specular_power,
                    render=render,
                    show_scalar_bar=show_scalar_bar,
                )

                actors.append(a)
            return actors

        # Make sure structured grids are not less than 3D
        # ImageData and RectilinearGrid should be olay as <3D
        if isinstance(volume, pyvista.StructuredGrid):
            if any(d < 2 for d in volume.dimensions):
                raise ValueError("StructuredGrids must be 3D dimensional.")

        if isinstance(volume, pyvista.PolyData):
            raise TypeError(
                f"Type {type(volume)} not supported for volume rendering as it is not 3D."
            )
        elif not isinstance(
            volume,
            (pyvista.ImageData, pyvista.RectilinearGrid, pyvista.UnstructuredGrid),
        ):
            volume = volume.cast_to_unstructured_grid()

        # Override mapper choice for UnstructuredGrid
        if isinstance(volume, pyvista.UnstructuredGrid):
            # Unstructured grid must be all tetrahedrals
            if not (volume.celltypes == pyvista.CellType.TETRA).all():
                volume = volume.triangulate()
            mapper = "ugrid"

        if mapper == "fixed_point" and not isinstance(volume, pyvista.ImageData):
            raise TypeError(
                f'Type {type(volume)} not supported for volume rendering with the `"fixed_point"` mapper. Use `pyvista.ImageData`.'
            )
        elif isinstance(volume, pyvista.UnstructuredGrid) and mapper != "ugrid":
            raise TypeError(
                f'Type {type(volume)} not supported for volume rendering with the `{mapper}` mapper. Use the "ugrid" mapper or simply leave as None.'
            )

        if opacity_unit_distance is None and not isinstance(
            volume, pyvista.UnstructuredGrid
        ):
            opacity_unit_distance = volume.length / (np.mean(volume.dimensions) - 1)

        if scalars is None:
            # Make sure scalars components are not vectors/tuples
            scalars = volume.active_scalars
            # Don't allow plotting of string arrays by default
            if scalars is not None and np.issubdtype(scalars.dtype, np.number):
                scalar_bar_args.setdefault("title", volume.active_scalars_info[1])
            else:
                raise MissingDataError("No scalars to use for volume rendering.")

        title = "Data"
        if isinstance(scalars, str):
            title = scalars
            scalars = get_array(volume, scalars, preference=preference, err=True)
            scalar_bar_args.setdefault("title", title)
        elif not isinstance(scalars, np.ndarray):
            scalars = np.asarray(scalars)

        if scalars.ndim != 1:
            if scalars.ndim != 2:
                raise ValueError(
                    "`add_volume` only supports scalars with 1 or 2 dimensions"
                )
            if scalars.shape[1] != 4 or scalars.dtype != np.uint8:
                raise ValueError(
                    "`add_volume` only supports scalars with 2 dimensions that have 4 components of datatype np.uint8.\n\n"
                    f"Scalars have shape {scalars.shape} and dtype {scalars.dtype.name!r}."
                )

        if not np.issubdtype(scalars.dtype, np.number):
            raise TypeError(
                "Non-numeric scalars are currently not supported for volume rendering."
            )
        if scalars.ndim != 1:
            if scalars.ndim != 2:
                raise ValueError(
                    "`add_volume` only supports scalars with 1 or 2 dimensions"
                )
            if scalars.shape[1] != 4 or scalars.dtype != np.uint8:
                raise ValueError(
                    f"`add_volume` only supports scalars with 2 dimension that have 4 components of datatype np.uint8, scalars have shape {scalars.shape} and datatype {scalars.dtype}"
                )
            if opacity != "linear":
                opacity = "linear"
                warnings.warn("Ignoring custom opacity due to RGBA scalars.")

        # Define mapper, volume, and add the correct properties
        mappers_lookup = {
            "fixed_point": FixedPointVolumeRayCastMapper,
            "gpu": GPUVolumeRayCastMapper,
            "open_gl": OpenGLGPUVolumeRayCastMapper,
            "smart": SmartVolumeMapper,
            "ugrid": UnstructuredGridVolumeRayCastMapper,
        }
        if not isinstance(mapper, str) or mapper not in mappers_lookup.keys():
            raise TypeError(
                f"Mapper ({mapper}) unknown. Available volume mappers include: {', '.join(mappers_lookup.keys())}"
            )
        self.mapper = mappers_lookup[mapper](theme=self._theme)

        # Set scalars range
        min_, max_ = None, None
        if clim is None:
            if scalars.dtype == np.uint8:
                clim = [0, 255]
            else:
                min_, max_ = np.nanmin(scalars), np.nanmax(scalars)
                clim = [min_, max_]
        elif isinstance(clim, float) or isinstance(clim, int):
            clim = [-clim, clim]

        if log_scale:
            if clim[0] <= 0:
                clim = [sys.float_info.min, clim[1]]

        # # data must be between [0, 255], but not necessarily UINT8
        # # Preserve backwards compatibility and have same behavior as VTK.
        # if scalars.dtype != np.uint8 and clim != [0, 255]:
        #     # must copy to avoid modifying inplace and remove any VTK weakref
        #     scalars = np.array(scalars)
        #     clim = np.asarray(clim, dtype=scalars.dtype)
        #     scalars.clip(clim[0], clim[1], out=scalars)
        #     if log_scale:
        #         out = matplotlib.colors.LogNorm(clim[0], clim[1])(scalars)
        #         scalars = out.data * 255
        #     else:
        #         if min_ is None:
        #             min_, max_ = np.nanmin(scalars), np.nanmax(scalars)
        #         np.true_divide((scalars - min_), (max_ - min_) / 255, out=scalars, casting='unsafe')
        scalars = np.array(scalars, dtype=float)
        clim = np.asarray(clim, dtype=scalars.dtype)
        scalars.clip(clim[0], clim[1], out=scalars)
        if log_scale:
            out = matplotlib.colors.LogNorm(clim[0], clim[1])(scalars)
            scalars = out.data * 255
        else:
            np.true_divide(
                (scalars - clim[0]),
                (clim[1] - clim[0]) / 255,
                out=scalars,
                casting="unsafe",
            )
        volume[title] = scalars
        volume.active_scalars_name = title

        # Scalars interpolation approach
        if scalars.shape[0] == volume.n_points:
            self.mapper.scalar_mode = "point"
        elif scalars.shape[0] == volume.n_cells:
            self.mapper.scalar_mode = "cell"
        else:
            raise_not_matching(scalars, volume)

        self.mapper.scalar_range = clim

        if isinstance(cmap, pyvista.LookupTable):
            self.mapper.lookup_table = cmap
        else:
            if cmap is None:
                cmap = self._theme.cmap

            cmap = get_cmap_safe(cmap)
            if categories:
                if categories is True:
                    n_colors = len(np.unique(scalars))
                elif isinstance(categories, int):
                    n_colors = categories

            if flip_scalars:
                cmap = cmap.reversed()

            # Set colormap and build lookup table
            self.mapper.lookup_table.apply_cmap(cmap, n_colors)
            self.mapper.lookup_table.apply_opacity(opacity)
            self.mapper.lookup_table.scalar_range = clim
            self.mapper.lookup_table.log_scale = log_scale
            if isinstance(annotations, dict):
                self.mapper.lookup_table.annotations = annotations

        self.mapper.dataset = volume
        self.mapper.blend_mode = blending
        self.mapper.update()

        self.volume = Volume()
        self.volume.mapper = self.mapper

        self.volume.prop = VolumeProperty(
            lookup_table=self.mapper.lookup_table,
            ambient=ambient,
            shade=shade,
            specular=specular,
            specular_power=specular_power,
            diffuse=diffuse,
            opacity_unit_distance=opacity_unit_distance,
        )

        if scalars.ndim == 2:
            self.volume.prop.independent_components = False
            show_scalar_bar = False

        actor, prop = self.add_actor(
            self.volume,
            reset_camera=reset_camera,
            name=name,
            culling=culling,
            pickable=pickable,
            render=render,
        )

        # Add scalar bar if scalars are available
        if show_scalar_bar and scalars is not None:
            self.add_scalar_bar(**scalar_bar_args)

        self.renderer.Modified()
        return actor


class ModPlotter(volume_mixer, Plotter):
    pass


def pv_imagedata(dataset, grid, name="data"):
    """Create a PyVista ImageData object from a numpy array and its grid.

    The origin of the PyVista object is not the same as the grid origin.
    In mrfmsim, the grid origin is defined as the middle of the grid.
    In pyvista plots, the origin is defined as the lower left corner
    (southwest corner) of the grid.
    """

    image_data = pyvista.ImageData()
    image_data.dimensions = grid.grid_shape
    image_data.origin = [
        grid.grid_extents[0][0],
        grid.grid_extents[1][0],
        grid.grid_extents[2][0],
    ]
    image_data.spacing = grid.grid_step

    # pyvsita sets the grid up as the F order, different from the mgrid
    # default generation
    image_data[name] = dataset.flatten(order="F")

    return image_data


def pv_plot_preset(preset):
    """Create a PyVista plot object with preset parameters.

    :param np.array dataset: the dataset to plot
    :param np.array grid: the grid of the dataset
    :param str plot_type: the type of plot to create
    :param kwargs: additional keyword arguments to pass to the plot function
    """

    p = ModPlotter()
    for key, value in preset.items():
        if isinstance(value, dict):
            getattr(p, key)(**value)
        else:
            setattr(p, key, value)
    return p


def pv_preset_volume(dataset, grid, name="data", **kwargs):
    """Create a preset for volume rendering.

    The keyword arguments are dictionary, and the content
    is updated to the present dictionary.
    """

    image_data = pv_imagedata(dataset, grid, name)

    preset_params = {
        "add_volume": {
            "volume": image_data,
            "clim": [dataset.min(), dataset.max()],
            "cmap": "viridis",
            "opacity": 0.8,
            "show_scalar_bar": False,
        },
        "window_size": (512, 768),
        "add_scalar_bar": {
            "title": "Data",
            "vertical": True,
            "position_x": 0.8,
            "position_y": 0.3,
            "fmt": "%.2e",
            "n_labels": 5,
        },
        "add_axes": {},
    }

    for key, value in kwargs.items():
        if key in preset_params and isinstance(preset_params[key], dict):
            preset_params[key].update(value)
        else:
            preset_params[key] = value
    return preset_params


# Defaults to times
pyvista.global_theme.font.family = "times"
