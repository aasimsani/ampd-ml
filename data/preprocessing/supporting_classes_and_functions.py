"""
Copied from https://github.com/aasimsani/artificial_manga_panel_dataset
which was used to generate the data for this
"""
import numpy as np
import random
import json
import uuid


class Panel(object):
    """
    A class to encapsulate a panel of the manga page.
    Since the script works in a parent-child relationship
    where each panel child is an area subset of some parent panel,
    some panels aren't leaf nodes and thus not rendered.

    :param coords: Coordinates of the boundary of the panel

    :type coords: list

    :param name: Unique name for the panel

    :type name: str

    :param parent: The panel which this panel is a child of

    :type parent: Panel

    :param orientation: Whether the panel consists of lines that are vertically
    or horizotnally oriented in reference to the page

    :type orientation: str

    :children: Children panels of this panel

    :type children: list

    :non_rect: Whether the panel was transformed to be non rectangular
    and thus has less or more than 4 coords

    :type non_rect: bool, optional
    """

    def __init__(self,
                 coords,
                 name,
                 parent,
                 orientation,
                 children=[],
                 non_rect=False):
        """
        Constructor methods
        """

        coords = [tuple(c) for c in coords]

        self.x1y1 = coords[0]
        self.x2y2 = coords[1]
        self.x3y3 = coords[2]
        self.x4y4 = coords[3]

        self.lines = [
            (self.x1y1, self.x2y2),
            (self.x2y2, self.x3y3),
            (self.x3y3, self.x4y4),
            (self.x4y4, self.x1y1)
        ]

        self.name = name
        self.parent = parent

        self.coords = coords
        self.non_rect = non_rect

        self.width = float((self.x2y2[0] - self.x1y1[0]))
        self.height = float((self.x3y3[1] - self.x2y2[1]))

        self.area = float(self.width*self.height)

        area_proportion = round(self.area/(2400*1700), 2)
        self.area_proportion = area_proportion

        if len(children) > 0:
            self.children = children
        else:
            self.children = []

        self.orientation = orientation

        # Whether or not this panel has been transformed by slicing into two
        self.sliced = False

        # Whether or not to render this panel
        self.no_render = False

        # Image from the illustration dataset which is
        # the background of this panel
        self.image = None

        # A list of speech bubble objects to render around this panel
        self.speech_bubbles = []

    def load_data(self, data):
        """
        This method reverses the dump_data function and
        load's the metadata of the panel from the subsection
        of the JSON file that has been loaded

        :param data: A dictionary of this panel's data

        :type data: dict
        """

        self.sliced = data['sliced']
        self.no_render = data['no_render']
        self.image = data['image']

        if len(data['speech_bubbles']) > 0:
            for speech_bubble in data['speech_bubbles']:
                transform_metadata = speech_bubble['transform_metadata']
                bubble = SpeechBubble(
                            texts=speech_bubble['texts'],
                            text_indices=speech_bubble['text_indices'],
                            font=speech_bubble['font'],
                            speech_bubble=speech_bubble['speech_bubble'],
                            writing_areas=speech_bubble['writing_areas'],
                            resize_to=speech_bubble['resize_to'],
                            location=speech_bubble['location'],
                            width=speech_bubble['width'],
                            height=speech_bubble['height'],
                            transforms=speech_bubble['transforms'],
                            transform_metadata=transform_metadata,
                            text_orientation=speech_bubble['text_orientation']
                            )

                self.speech_bubbles.append(bubble)

        # Recursively load children
        children = []
        if len(data['children']) > 0:
            for child in data['children']:
                panel = Panel(
                    coords=child['coordinates'],
                    name=child['name'],
                    parent=self,
                    orientation=child['orientation'],
                    non_rect=child['non_rect']
                )

                panel.load_data(child)
                children.append(panel)

        self.children = children


class Page(Panel):
    """
    A class that represents a full page consiting of multiple child panels

    :param coords: A list of the boundary coordinates of a page

    :type coords: list

    :param page_type: Signifies whether a page consists of vertical
    or horizontal panels or both

    :type page_type: str

    :param num_panels: Number of panels in this page

    :type num_panels: int

    :param children: List of direct child panels of this page

    :type children: list, optional:
    """

    def __init__(self,
                 coords=[],
                 page_type="",
                 num_panels=1,
                 children=[],
                 name=None
                 ):
        """
        Constructor method
        """

        if len(coords) < 1:
            topleft = (0.0, 0.0)
            topright = (1700, 0.0)
            bottomleft = (0.0, 2400)
            bottomright = (1700, 2400)
            coords = [
                topleft,
                topright,
                bottomright,
                bottomleft
            ]

        if name is None:
            self.name = str(uuid.uuid1())
        else:
            self.name = name

        # Initalize the panel super class
        super().__init__(coords=coords,
                         name=self.name,
                         parent=None,
                         orientation=None,
                         children=[]
                         )

        self.num_panels = num_panels
        self.page_type = page_type

        # Whether this page needs to be rendered with a background
        self.background = None

        # The leaf children of tree of panels
        # These are the panels that are actually rendered
        self.leaf_children = []

        # Size of the page
        self.page_size = (1700, 2400)

    def load_data(self, filename):

        """
        This method reverses the dump_data function and
        load's the metadata of the page from the JSON
        file that has been loaded.

        :param filename: JSON filename to load

        :type filename: str
        """
        with open(filename, "rb") as json_file:

            data = json.load(json_file)

            self.name = data['name']
            self.num_panels = int(data['num_panels'])
            self.page_type = data['page_type']
            self.background = data['background']

            if len(data['speech_bubbles']) > 0:
                for speech_bubble in data['speech_bubbles']:
                    # Line constraints
                    text_orientation = speech_bubble['text_orientation']
                    transform_metadata = speech_bubble['transform_metadata']
                    bubble = SpeechBubble(
                                texts=speech_bubble['texts'],
                                text_indices=speech_bubble['text_indices'],
                                font=speech_bubble['font'],
                                speech_bubble=speech_bubble['speech_bubble'],
                                writing_areas=speech_bubble['writing_areas'],
                                resize_to=speech_bubble['resize_to'],
                                location=speech_bubble['location'],
                                width=speech_bubble['width'],
                                height=speech_bubble['height'],
                                transforms=speech_bubble['transforms'],
                                transform_metadata=transform_metadata,
                                text_orientation=text_orientation
                                )

                    self.speech_bubbles.append(bubble)

            # Recursively load children
            if len(data['children']) > 0:
                for child in data['children']:
                    panel = Panel(
                        coords=child['coordinates'],
                        name=child['name'],
                        parent=self,
                        orientation=child['orientation'],
                        non_rect=child['non_rect']
                    )
                    panel.load_data(child)
                    self.children.append(panel)


class SpeechBubble(object):
    """
    A class to represent the metadata to render a speech bubble

    :param texts: A list of texts from the text corpus to render in this
    bubble

    :type texts: lists

    :param text_indices: The indices of the text from the dataframe
    for easy retrival

    :type text_indices: lists

    :param font: The path to the font used in the bubble

    :type font: str

    :param speech_bubble: The path to the base speech bubble file
    used for this bubble

    :type speech_bubble: str

    :param writing_areas: The areas within the bubble where it is okay
    to render text

    :type writing_areas: list

    :param resize_to: The amount of area this text bubble should consist of
    which is a ratio of the panel's area

    :type resize_to: float

    :param location: The location of the top left corner of the speech bubble
    on the page

    :type location: list

    :param width: Width of the speech bubble, defaults to 0

    :type width: float, optional

    :param height: Height of the speech bubble

    :type height: float

    :param transforms: A list of transformations to change
    the shape of the speech bubble

    :type transforms: list

    :param transform_metadata: Metadata associated with transformations

    :type transform_metadata: dict

    :param text_orientation: Whether the text of this speech bubble
    is written left to right ot top to bottom

    :type text_orientation: str
    """
    def __init__(self,
                 texts,
                 text_indices,
                 font,
                 speech_bubble,
                 writing_areas,
                 resize_to,
                 location,
                 width,
                 height,
                 transforms,
                 transform_metadata,
                 text_orientation):
        """
        Constructor method
        """

        self.texts = texts
        # Index of dataframe for the text
        self.text_indices = text_indices
        self.font = font
        self.speech_bubble = speech_bubble
        self.writing_areas = writing_areas
        self.resize_to = resize_to

        # Location on panel
        self.location = location
        self.width = width
        self.height = height

        self.transforms = transforms
        self.transform_metadata = transform_metadata

        self.text_orientation = text_orientation


def get_leaf_panels(page, panels):
    """
    Get panels which are to actually
    be rendered recursively i.e. they
    are the leaves of the Page-Panel
    tree

    :param page: Page to be searched

    :type page: Page

    :param panels: A list of panels to be
    returned by refernce

    :type panels: list
    """

    # For each child
    for child in page.children:

        # See if panel has no children
        # Therefore is a leaf
        if len(child.children) > 0:
            # If has children keep going
            get_leaf_panels(child, panels)
        else:
            # Otherwise put in list to return
            panels.append(child)
