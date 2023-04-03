import pandas as pd
import plotly.express as px


def get_grad_flow_figure(named_params):
    """Creates figure to visualize gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Use this function after loss.backwards()
    """
    data_dict = {
        'name': list(),
        'grad_abs': list(),
    }
    for name, param in named_params:
        if param.requires_grad and param.grad is not None:
            grad_abs = param.grad.abs()
            data_dict['name'].append(name)
            data_dict['grad_abs'].append(grad_abs.mean().cpu().item())

    data_frame = pd.DataFrame.from_dict(data_dict)

    fig = px.bar(data_frame, x='name', y='grad_abs')
    return fig
