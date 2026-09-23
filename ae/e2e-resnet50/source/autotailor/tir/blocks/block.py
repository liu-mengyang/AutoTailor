import json


class Block(object):
    def __init__(self):
        self.type = 'basic'
        self.name = 'block'
        self.features = {}
        self.idx = 0
        self.paths = None
        self.update_dimensions = ["in_shape", "in_channel", "out_channel"]
        self.dynamic_dimensions = []

        self.verbose_level = None

    def recount(self):
        if hasattr(self, "flow"):
            for i, op in enumerate(self.flow):
                if isinstance(op, Block):
                    op.recount()
                op.id = i

    def bind_weight(self):
        pass

    def get_ops():
        pass

    def update_grad():
        pass

    def reorganize_weight(self, cfgs):
        pass

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
        """
        for k, v in kv_features.items():
            if k in self.features and k in self.update_dimensions:
                self.features[k] = v
            else:
                raise KeyError(f"{k} is not supported")

    def transform(self, kv_features: dict) -> None:
        """Transform features of the block.

        Args:
          kv_features: the dictionary of to transform features.

        Raises:
          KeyError: the error occured in updating non exist feature or not
                    not support to transform this dimension.
        """
        for k, v in kv_features.items():
            if k in self.features and k in self.dynamic_dimensions:
                if v <= self.features[f"max_{k}"]:
                    self.features[k] = v
            else:
                raise KeyError

    def mask(self):
        self.features["activated"] = False

    def unmask(self):
        self.features["activated"] = True

    def is_transformable(self):
        return self.features["trans_type"] == "active"

    def is_active(self):
        return self.features["activated"]

    def count_flops_params(self):
        pass

    def build(self, cache=None, block_id=None):
        raise NotImplementedError

    @property
    def info_dict(self):
        ret_dict = {}
        ret_dict['idx'] = self.idx
        ret_dict['name'] = self.name
        ret_dict['type'] = self.type

        if self.verbose_level == 'complex':
            ret_dict['features'] = self.features
        elif self.verbose_level == 'show':
            ret_dict['in_shape'] = self.features['in_shape']
            ret_dict['out_shape'] = self.features['out_shape']

        return ret_dict
