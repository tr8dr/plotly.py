import _plotly_utils.basevalidators


class ButtonsValidator(_plotly_utils.basevalidators.CompoundValidator):
    def __init__(
        self, plotly_name="buttons", parent_name="layout.xaxis.rangeselector", **kwargs
    ):
        super(ButtonsValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            data_class_str=kwargs.pop("data_class_str", "Buttons"),
            data_docs=kwargs.pop(
                "data_docs",
                """
""",
            ),
            **kwargs
        )
