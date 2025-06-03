def parse_format_dtype(dtype):
    if dtype == "int64":
        return int
    if dtype == "str":
        return str
    if dtype == "list(int64)":
        return lambda x: [int(elem) for elem in x.split(",")]
    if dtype == "list(str)":
        return lambda x: [str(elem).strip() for elem in x.split(",")]


def init_empty(dtype):
    if dtype == "int64":
        return 0
    if dtype == "str":
        return ""
    if dtype == "list(int64)":
        return []
    if dtype == "list(str)":
        return []


def inverse_parse_format_dtype(dtype):
    if dtype == "int64":
        return str
    if dtype == "str":
        return str
    if dtype == "list(int64)":
        return lambda x: ", ".join([str(i) for i in x])
    if dtype == "list(str)":
        return lambda x: ", ".join([i for i in x])


def get_default(dict_, key_, func_):
    if key_ in dict_.keys():
        value = dict_[key_]
        if value is None:
            return ""
        return func_(value)
    else:
        return ""


def make_default_attributes(attribute_format):
    attributes = {}
    for key, dtype in attribute_format:
        attributes[key] = init_empty(dtype)

    return attributes


class ConsistencyChecker:
    def __init__(self, attribute_format):
        self.attribute_format = attribute_format

    def update_neighbours(self, self_id, element_attributes, feature_ids):
        element_attributes = self.add_neighbours(
            self_id, element_attributes, feature_ids, successors=True
        )
        element_attributes = self.add_neighbours(
            self_id, element_attributes, feature_ids, successors=False
        )

        element_attributes = self.remove_neighbours(
            self_id, element_attributes, feature_ids, successors=True
        )
        element_attributes = self.remove_neighbours(
            self_id, element_attributes, feature_ids, successors=False
        )

        return element_attributes

    def remove_neighbours(
        self, self_id, element_attributes, feature_ids, successors=True
    ):
        self_attributes = element_attributes[self_id]

        neighbour_type = "successors" if successors else "predecessors"
        self_type = "predecessors" if successors else "successors"
        if self_attributes[neighbour_type] is None:
            self_neighbours = []
        else:
            self_neighbours = self_attributes[neighbour_type]

        attributes_keys = element_attributes.keys()
        for feature_id_int in feature_ids:
            feature_id = str(feature_id_int)
            if feature_id not in attributes_keys:
                continue

            feature_attributes = element_attributes[feature_id]
            feature_neighbours = feature_attributes[self_type]

            if feature_neighbours is None:
                continue

            if feature_id not in self_neighbours and self_id in feature_neighbours:
                feature_neighbours.remove(self_id)

                element_attributes[feature_id][self_type] = feature_neighbours

        return element_attributes

    def add_neighbours(self, self_id, element_attributes, feature_ids, successors=True):
        if self_id not in element_attributes:
            self_attributes = make_default_attributes(self.attribute_format)
            self_attributes["lane_id"] = self_id
            element_attributes[self_id] = self_attributes
        else:
            self_attributes = element_attributes[self_id]

        neighbour_type = "successors" if successors else "predecessors"
        self_type = "predecessors" if successors else "successors"

        if self_attributes[neighbour_type] is None:
            return element_attributes

        for neighbour in self_attributes[neighbour_type]:
            neighbour_int = int(neighbour)

            if neighbour_int not in feature_ids:
                print(f"Lane {neighbour} does not exist.")

            if neighbour not in element_attributes.keys():
                neighbour_attributes = make_default_attributes(self.attribute_format)
                neighbour_attributes["lane_id"] = neighbour
            else:
                neighbour_attributes = element_attributes[neighbour]

                if neighbour_attributes[self_type] is None:
                    neighbour_attributes[self_type] = []

            if str(self_id) not in neighbour_attributes[self_type]:
                neighbour_attributes[self_type].append(str(self_id))
            element_attributes[neighbour] = neighbour_attributes

        return element_attributes
