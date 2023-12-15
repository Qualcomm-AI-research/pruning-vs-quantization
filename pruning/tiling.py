# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from utils import ClassEnumOptions, MethodMap


class Tiler:
    def __init__(self, tile_size):
        self.tile_size = tile_size

    def __call__(self, x):
        raise NotImplementedError

    def inverse(self, x, orig_shape):
        raise NotImplementedError


class FlattenTiler(Tiler):
    def __call__(self, x):
        orig_shape = x.shape
        return x.view(-1, self.tile_size), orig_shape

    def inverse(self, x, orig_shape):
        return x.view(orig_shape)


class XYIOTiler(Tiler):
    def __call__(self, x):
        x = x.permute(2, 3, 0, 1)

        orig_shape = x.shape
        x = x.reshape(-1, self.tile_size)
        return x, orig_shape

    def inverse(self, x, orig_shape):
        x = x.reshape(orig_shape)
        return x.permute(2, 3, 0, 1)


class OXIYTiler(Tiler):
    def __call__(self, x):
        if x.shape[1] == 3:
            return None, None

        x = x.permute(0, 3, 1, 2)

        orig_shape = x.shape
        x = x.reshape(-1, self.tile_size)
        return x, orig_shape

    def inverse(self, x, orig_shape):
        x = x.reshape(orig_shape)
        return x.permute(0, 2, 3, 1)


class StructuredSparsityFlatten(Tiler):
    def __call__(self, x):
        if len(x.shape) > 2:
            x = x.permute(1, 0, 2, 3)
        else:
            x = x.permute(1, 0)

        orig_shape = x.shape

        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)

        return x, orig_shape

    def inverse(self, x, orig_shape):
        x = x.reshape(orig_shape)
        if len(x.shape) > 2:
            return x.permute(1, 0, 2, 3)
        else:
            return x.permute(1, 0)


class OXYITiler(Tiler):
    def __call__(self, x):
        if x.shape[1] == 3:
            return None, None

        if len(x.shape) == 2:
            return x.view(-1, self.tile_size), x.shape

        x = x.permute(0, 2, 3, 1)

        orig_shape = x.shape
        x = x.reshape(-1, self.tile_size)
        return x, orig_shape

    def inverse(self, x, orig_shape):
        x = x.reshape(orig_shape)

        if len(x.shape) == 2:
            return x

        return x.permute(0, 3, 1, 2)


def notile(x, *args, **kwargs):
    return x.view(1, -1)


class TileMethod(ClassEnumOptions):
    flatten = MethodMap(FlattenTiler)
    xyiotiler = MethodMap(XYIOTiler)
    oxiytiler = MethodMap(OXIYTiler)
    structured_flatten = MethodMap(StructuredSparsityFlatten)
    oxyitiler = MethodMap(OXYITiler)
