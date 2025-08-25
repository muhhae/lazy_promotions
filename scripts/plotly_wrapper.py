from typing import List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.io import templates

templates.default = "plotly_white"


def Line(
    df: pd.DataFrame, x, y, size=False, count=1, include_zero: bool = False, **kwargs
) -> go.Figure:
    fig = px.line(
        df,
        x=x,
        y=y,
        **kwargs,
    )
    fig.update_layout(
        font=dict(size=32),
        height=750 * count,
        width=1000,
    )
    fig.update_traces(
        opacity=0.8,
        line=dict(width=4),
    )
    fig.update_xaxes(showgrid=True, nticks=12)
    fig.update_yaxes(showgrid=True, nticks=18)
    if size:
        fig.update_layout(yaxis_tickformat="s")

    if include_zero:
        fig.update_xaxes(rangemode="tozero")
        fig.update_yaxes(range=[0, 1])
    return fig


def Scatter(df: pd.DataFrame, include_zero: bool = False, **kwargs) -> go.Figure:
    fig = px.scatter(df, **kwargs)
    fig.update_layout(
        font=dict(size=32),
        # height=750,
        width=1000,
    )
    fig.update_traces(
        opacity=0.8,
        marker_size=28,
        marker_line=dict(width=4),
        selector=dict(mode="markers"),
    )
    fig.update_xaxes(showgrid=True, nticks=12)
    fig.update_yaxes(showgrid=True, nticks=18)
    if include_zero:
        fig.update_xaxes(rangemode="tozero")
        fig.update_yaxes(range=[0, 1])
    return fig


def Box(df: pd.DataFrame, x: str, y: str, **kwargs) -> go.Figure:
    fig = px.box(
        df,
        x=x,
        y=y,
        points="outliers",
        **kwargs,
    )
    fig.update_traces(marker_opacity=0)
    fig.update_layout(
        font=dict(size=24),
        height=800,
        width=1000,
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, nticks=20)
    return fig


def Box_2(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    category_col: str,
    lower_fence: float = 0.1,
    upper_fence: float = 0.9,
    **kwargs,
) -> go.Figure:
    df.sort_values(by=x_col)
    stats = (
        df.groupby([x_col, category_col])[y_col]
        .quantile(np.array([lower_fence, 0.25, 0.50, 0.75, upper_fence]))
        .unstack()
    )

    stats.columns = ["l_fence", "q1", "median", "q3", "u_fence"]
    stats = stats.reset_index()
    fig = go.Figure()
    for category in df[category_col].unique():
        category_stats = stats[stats[category_col] == category]
        fig.add_trace(
            go.Box(
                name=category,
                x=category_stats[x_col],
                q1=category_stats["q1"],
                median=category_stats["median"],
                q3=category_stats["q3"],
                lowerfence=category_stats["l_fence"],
                upperfence=category_stats["u_fence"],
                boxpoints=False,
                boxmean="sd",
                **kwargs,
            ),
        )
    fig.update_layout(boxmode="group")
    return fig


def VerticalCompositionBar(
    df: pd.DataFrame,
    X: str,
    Ys: List[str | tuple[str, str]],
    title: str | None = None,
    yaxis_title: str | None = None,
    xaxis_title: str | None = None,
    mode: str = "stack",
    y_max=None,
) -> go.Figure:
    df = df.sort_values(by=X)
    fig = go.Figure()
    for Y in Ys:
        fig.add_trace(
            go.Bar(
                x=df[X],
                y=df[Y] if isinstance(Y, str) else df[Y[0]],
                name=Y if isinstance(Y, str) else Y[1],
            )
        )
    fig.update_layout(
        barmode=mode,
        font=dict(size=32),
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.update_xaxes(showgrid=True, nticks=10)
    fig.update_yaxes(showgrid=True, nticks=10, range=[0, y_max])
    fig.update_xaxes(type="category")
    return fig
