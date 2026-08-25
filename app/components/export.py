"""
Export module – handles downloading charts as PNG, SVG, JPEG, HTML, GIF,
and exporting data as CSV, Excel, JSON.
"""

import io
import base64
from typing import Optional, List
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Check if kaleido is available (for static image export)
_KALEIDO_AVAILABLE = False
try:
    import kaleido  # noqa: F401
    _KALEIDO_AVAILABLE = True
except ImportError:
    pass

# Optional GIF dependencies
_GIF_AVAILABLE = False
try:
    import imageio
    from PIL import Image
    _GIF_AVAILABLE = True
except ImportError:
    pass


def fig_to_bytes(
    fig: go.Figure,
    format: str = "png",
    width: int = 1200,
    height: int = 700,
    scale: int = 2,
) -> Optional[bytes]:
    """
    Convert a Plotly figure to bytes using kaleido (if available).
    Returns None if conversion fails.
    """
    if not _KALEIDO_AVAILABLE:
        return None
    try:
        return fig.to_image(format=format, width=width, height=height, scale=scale)
    except Exception:
        return None


def fig_to_html(fig: go.Figure, include_plotlyjs: str = "cdn") -> str:
    """Convert figure to a standalone HTML string."""
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)


def _download_button(
    label: str,
    data: bytes,
    file_name: str,
    mime: str,
    key: str,
    use_container_width: bool = True,
    disabled: bool = False,
) -> None:
    """Helper to create a download button with consistent styling."""
    if disabled or data is None:
        st.button(
            label,
            disabled=True,
            key=f"{key}_disabled",
            use_container_width=use_container_width,
        )
    else:
        st.download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key,
            use_container_width=use_container_width,
        )


def create_export_panel(
    fig: go.Figure,
    chart_name: str,
    key_prefix: str = "",
    include_gif: bool = False,
    gif_frames: Optional[List[go.Figure]] = None,
) -> None:
    """
    Create an expandable panel with download buttons for PNG, JPEG, SVG, HTML,
    Hi‑Res PNG, and optionally a GIF.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to export.
    chart_name : str
        Base filename (without extension).
    key_prefix : str
        Unique prefix for Streamlit widget keys.
    include_gif : bool
        Whether to include a GIF export button (requires imageio & Pillow).
    gif_frames : Optional[List[go.Figure]]
        If provided, use these frames for the GIF (otherwise single frame).
    """
    with st.expander("📥 Download Chart", expanded=False):
        # Pre‑compute static images (if kaleido is available)
        png_data = fig_to_bytes(fig, "png", width=1200, height=700, scale=2)
        jpeg_data = fig_to_bytes(fig, "jpeg", width=1200, height=700, scale=2)
        svg_data = fig_to_bytes(fig, "svg", width=1200, height=700, scale=1)  # SVG scale=1
        hires_data = fig_to_bytes(fig, "png", width=3000, height=1800, scale=3)

        # HTML always available
        html_data = fig_to_html(fig).encode("utf-8")

        cols = st.columns(5)

        with cols[0]:
            _download_button(
                "🖼️ PNG",
                png_data,
                f"{chart_name}.png",
                "image/png",
                f"{key_prefix}_png",
            )

        with cols[1]:
            _download_button(
                "📷 JPEG",
                jpeg_data,
                f"{chart_name}.jpg",
                "image/jpeg",
                f"{key_prefix}_jpeg",
            )

        with cols[2]:
            _download_button(
                "🎨 SVG",
                svg_data,
                f"{chart_name}.svg",
                "image/svg+xml",
                f"{key_prefix}_svg",
            )

        with cols[3]:
            _download_button(
                "🌐 HTML",
                html_data,
                f"{chart_name}.html",
                "text/html",
                f"{key_prefix}_html",
            )

        with cols[4]:
            _download_button(
                "🔍 Hi‑Res",
                hires_data,
                f"{chart_name}_4K.png",
                "image/png",
                f"{key_prefix}_hires",
            )

        # Optional GIF export
        if include_gif and _GIF_AVAILABLE:
            gif_data = _create_gif_from_frames(fig, gif_frames)
            if gif_data:
                st.download_button(
                    label="🎞️ GIF",
                    data=gif_data,
                    file_name=f"{chart_name}.gif",
                    mime="image/gif",
                    key=f"{key_prefix}_gif",
                    use_container_width=True,
                )
            else:
                st.button("🎞️ GIF", disabled=True, key=f"{key_prefix}_gif_disabled")

        # Show a warning if kaleido is missing
        if not _KALEIDO_AVAILABLE:
            st.warning(
                "⚠️ **Kaleido not installed** – image exports (PNG, JPEG, SVG, Hi‑Res) are disabled. "
                "Install with `pip install kaleido` or use HTML export."
            )


def _create_gif_from_frames(
    fig: go.Figure,
    frames: Optional[List[go.Figure]] = None,
    duration: float = 0.5,
) -> Optional[bytes]:
    """
    Create a GIF from a list of frame figures. If no frames are provided,
    fall back to a single‑frame GIF.
    """
    if not _GIF_AVAILABLE:
        return None

    try:
        # If frames not provided, try to extract from fig.frames (Plotly animation)
        if frames is None:
            if hasattr(fig, "frames") and fig.frames:
                frames = []
                for f in fig.frames:
                    # Create a new figure with the frame's data and original layout
                    temp_fig = go.Figure(data=f.data, layout=fig.layout)
                    frames.append(temp_fig)
            else:
                # Single static figure – create one frame
                frames = [fig]

        # Convert each frame to PNG bytes
        image_bytes_list = []
        for frame_fig in frames:
            img_data = fig_to_bytes(frame_fig, "png", width=800, height=500, scale=1)
            if img_data is None:
                return None
            image_bytes_list.append(img_data)

        # Use imageio to assemble GIF
        images = [Image.open(io.BytesIO(img)) for img in image_bytes_list]
        buffer = io.BytesIO()
        imageio.mimsave(buffer, images, format="GIF", duration=duration, loop=0)
        return buffer.getvalue()

    except Exception:
        return None


# ─── Data Export ──────────────────────────────────────────────────────────────

def export_dataframe(df, filename: str, format: str = "csv") -> Optional[bytes]:
    """Export a pandas DataFrame to bytes in the specified format."""
    try:
        if format == "csv":
            return df.to_csv(index=False).encode("utf-8")
        elif format == "excel":
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine="openpyxl")
            return buffer.getvalue()
        elif format == "json":
            return df.to_json(orient="records", indent=2).encode("utf-8")
        else:
            return df.to_csv(index=False).encode("utf-8")
    except Exception:
        return None


def create_data_export_panel(df, filename: str = "data", key_prefix: str = "") -> None:
    """Create a panel with buttons to download data as CSV, Excel, JSON."""
    st.markdown("##### 📊 Download Data")

    csv_data = export_dataframe(df, filename, "csv")
    excel_data = export_dataframe(df, filename, "excel")
    json_data = export_dataframe(df, filename, "json")

    cols = st.columns(3)

    with cols[0]:
        _download_button(
            "📄 CSV",
            csv_data,
            f"{filename}.csv",
            "text/csv",
            f"{key_prefix}_csv",
        )

    with cols[1]:
        _download_button(
            "📊 Excel",
            excel_data,
            f"{filename}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"{key_prefix}_excel",
            disabled=excel_data is None,
        )

    with cols[2]:
        _download_button(
            "📋 JSON",
            json_data,
            f"{filename}.json",
            "application/json",
            f"{key_prefix}_json",
        )



# """
# Export module - Handles downloading charts as PNG, SVG, HTML, and GIF.
# """

# import io
# import base64
# from typing import Optional
# import plotly.graph_objects as go
# import streamlit as st


# def fig_to_bytes(fig: go.Figure, format: str = 'png', width: int = 1200, height: int = 700) -> bytes:
#     """Convert Plotly figure to bytes."""
#     return fig.to_image(format=format, width=width, height=height, scale=2)


# def fig_to_html(fig: go.Figure) -> str:
#     """Convert Plotly figure to HTML string."""
#     return fig.to_html(include_plotlyjs='cdn', full_html=True)


# def create_download_button(
#     fig: go.Figure,
#     filename: str,
#     format: str,
#     button_text: str,
#     key: str
# ) -> None:
#     """Create a Streamlit download button for chart export."""
    
#     if format in ['png', 'svg', 'jpeg', 'webp']:
#         try:
#             img_bytes = fig_to_bytes(fig, format=format)
#             mime_type = f'image/{format}'
            
#             st.download_button(
#                 label=button_text,
#                 data=img_bytes,
#                 file_name=f"{filename}.{format}",
#                 mime=mime_type,
#                 key=key,
#             )
#         except Exception as e:
#             st.error(f"Export failed: {e}. Make sure 'kaleido' is installed.")
    
#     elif format == 'html':
#         html_str = fig_to_html(fig)
        
#         st.download_button(
#             label=button_text,
#             data=html_str,
#             file_name=f"{filename}.html",
#             mime='text/html',
#             key=key,
#         )


# def create_export_panel(fig: go.Figure, chart_name: str, key_prefix: str = '') -> None:
#     """Create a panel with multiple export options — PNG, SVG, JPEG, HTML, Hi-Res."""

#     with st.expander("📥 Download Chart", expanded=False):
#         cols = st.columns(5)

#         with cols[0]:
#             try:
#                 img_bytes = fig.to_image(format='png', width=1200, height=700, scale=2)
#                 st.download_button(
#                     label='🖼️ PNG',
#                     data=img_bytes,
#                     file_name=f'{chart_name}.png',
#                     mime='image/png',
#                     key=f'{key_prefix}_png',
#                     use_container_width=True,
#                 )
#             except Exception:
#                 st.button('🖼️ PNG', disabled=True, key=f'{key_prefix}_png_d', use_container_width=True)

#         with cols[1]:
#             try:
#                 img_bytes = fig.to_image(format='jpeg', width=1200, height=700, scale=2)
#                 st.download_button(
#                     label='📷 JPEG',
#                     data=img_bytes,
#                     file_name=f'{chart_name}.jpg',
#                     mime='image/jpeg',
#                     key=f'{key_prefix}_jpeg',
#                     use_container_width=True,
#                 )
#             except Exception:
#                 st.button('📷 JPEG', disabled=True, key=f'{key_prefix}_jpeg_d', use_container_width=True)

#         with cols[2]:
#             try:
#                 img_bytes = fig.to_image(format='svg', width=1200, height=700)
#                 st.download_button(
#                     label='🎨 SVG',
#                     data=img_bytes,
#                     file_name=f'{chart_name}.svg',
#                     mime='image/svg+xml',
#                     key=f'{key_prefix}_svg',
#                     use_container_width=True,
#                 )
#             except Exception:
#                 st.button('🎨 SVG', disabled=True, key=f'{key_prefix}_svg_d', use_container_width=True)

#         with cols[3]:
#             html_str = fig.to_html(include_plotlyjs=False, full_html=True)
#             st.download_button(
#                 label='🌐 HTML',
#                 data=html_str.encode('utf-8'),
#                 file_name=f'{chart_name}.html',
#                 mime='text/html',
#                 key=f'{key_prefix}_html',
#                 use_container_width=True,
#             )

#         with cols[4]:
#             try:
#                 hires_bytes = fig.to_image(format='png', width=3000, height=1800, scale=3)
#                 st.download_button(
#                     label='🔍 Hi-Res',
#                     data=hires_bytes,
#                     file_name=f'{chart_name}_4K.png',
#                     mime='image/png',
#                     key=f'{key_prefix}_hires',
#                     use_container_width=True,
#                 )
#             except Exception:
#                 st.button('🔍 Hi-Res', disabled=True, key=f'{key_prefix}_hires_d', use_container_width=True)


# def export_dataframe(df, filename: str, format: str = 'csv') -> bytes:
#     """Export DataFrame to bytes."""
#     if format == 'csv':
#         return df.to_csv(index=False).encode('utf-8')
#     elif format == 'excel':
#         buffer = io.BytesIO()
#         df.to_excel(buffer, index=False, engine='openpyxl')
#         return buffer.getvalue()
#     elif format == 'json':
#         return df.to_json(orient='records', indent=2).encode('utf-8')
#     else:
#         return df.to_csv(index=False).encode('utf-8')


# def create_data_export_panel(df, filename: str = 'data', key_prefix: str = '') -> None:
#     """Create panel for exporting data."""
    
#     st.markdown("##### 📊 Download Data")
    
#     cols = st.columns(3)
    
#     with cols[0]:
#         csv_data = export_dataframe(df, filename, 'csv')
#         st.download_button(
#             label='📄 CSV',
#             data=csv_data,
#             file_name=f'{filename}.csv',
#             mime='text/csv',
#             key=f'{key_prefix}_csv'
#         )
    
#     with cols[1]:
#         try:
#             excel_data = export_dataframe(df, filename, 'excel')
#             st.download_button(
#                 label='📊 Excel',
#                 data=excel_data,
#                 file_name=f'{filename}.xlsx',
#                 mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
#                 key=f'{key_prefix}_excel'
#             )
#         except Exception:
#             st.button('📊 Excel', disabled=True, key=f'{key_prefix}_excel_disabled')
    
#     with cols[2]:
#         json_data = export_dataframe(df, filename, 'json')
#         st.download_button(
#             label='📋 JSON',
#             data=json_data,
#             file_name=f'{filename}.json',
#             mime='application/json',
#             key=f'{key_prefix}_json'
#         )


# def create_animated_gif(frames: list, duration: float = 0.5) -> bytes:
#     """Create animated GIF from list of image frames."""
#     try:
#         import imageio
#         from PIL import Image
        
#         images = []
#         for frame_bytes in frames:
#             img = Image.open(io.BytesIO(frame_bytes))
#             images.append(img)
        
#         buffer = io.BytesIO()
#         imageio.mimsave(buffer, images, format='GIF', duration=duration, loop=0)
#         return buffer.getvalue()
    
#     except ImportError:
#         raise ImportError("Please install imageio and Pillow for GIF export")


# def capture_animation_frames(fig: go.Figure, num_frames: int = 10) -> list:
#     """Capture frames from an animated Plotly figure."""
#     frames = []
    
#     if hasattr(fig, 'frames') and fig.frames:
#         # Get frames from animated figure
#         for i, frame in enumerate(fig.frames[:num_frames]):
#             temp_fig = go.Figure(data=frame.data, layout=fig.layout)
#             frame_bytes = temp_fig.to_image(format='png', width=800, height=500)
#             frames.append(frame_bytes)
#     else:
#         # Single frame
#         frame_bytes = fig.to_image(format='png', width=800, height=500)
#         frames.append(frame_bytes)
    
#     return frames
 
