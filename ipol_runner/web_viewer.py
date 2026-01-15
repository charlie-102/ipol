"""Web viewer for IPOL methods - visual comparison by category."""
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

from .registry import get_all_methods, get_method
from .base import MethodCategory, InputType
from .validation import get_passed_methods, get_failed_methods, is_validated


def _encode_image(path: Path) -> Optional[str]:
    """Encode image as base64 data URI."""
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.webp': 'image/webp',
    }
    mime = mime_types.get(suffix, 'image/png')
    try:
        data = base64.b64encode(path.read_bytes()).decode('utf-8')
        return f"data:{mime};base64,{data}"
    except Exception:
        return None


def generate_gallery_html(
    output_base: Path,
    title: str = "IPOL Method Gallery",
    validated_only: bool = True
) -> str:
    """Generate HTML gallery of methods organized by category.

    Args:
        output_base: Base directory containing method output folders
        title: Page title
        validated_only: Only include methods that passed validation

    Returns:
        HTML string
    """
    all_methods = get_all_methods()  # Returns dict {name: method}
    passed_methods = get_passed_methods()
    failed_info = get_failed_methods()

    # Filter to validated methods only
    if validated_only:
        methods = [m for name, m in all_methods.items() if name in passed_methods]
    else:
        methods = list(all_methods.values())

    # Group by category
    by_category: Dict[str, List] = {}
    for method in methods:
        cat = method.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(method)

    # Build HTML
    html_parts = [f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        .category {{
            margin-bottom: 40px;
        }}
        .category-header {{
            background: #3498db;
            color: white;
            padding: 12px 20px;
            border-radius: 8px 8px 0 0;
            font-size: 1.3em;
            font-weight: 600;
            text-transform: capitalize;
        }}
        .category-content {{
            background: white;
            border-radius: 0 0 8px 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .method-card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .method-header {{
            background: #ecf0f1;
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .method-name {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
            margin: 0 0 5px 0;
        }}
        .method-desc {{
            color: #666;
            font-size: 0.9em;
            margin: 0;
        }}
        .method-meta {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 0.85em;
        }}
        .meta-badge {{
            background: #3498db;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
        }}
        .meta-badge.input {{
            background: #27ae60;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            padding: 15px;
        }}
        .image-cell {{
            text-align: center;
        }}
        .image-cell img {{
            max-width: 100%;
            max-height: 300px;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .image-label {{
            margin-top: 8px;
            font-size: 0.85em;
            color: #666;
        }}
        .no-output {{
            color: #999;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }}
        .nav {{
            position: sticky;
            top: 0;
            background: white;
            padding: 15px;
            margin: -20px -20px 20px -20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 100;
        }}
        .nav-links {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .nav-link {{
            color: #3498db;
            text-decoration: none;
            padding: 5px 15px;
            border-radius: 20px;
            background: #ecf0f1;
            text-transform: capitalize;
        }}
        .nav-link:hover {{
            background: #3498db;
            color: white;
        }}
        .params {{
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <nav class="nav">
        <div class="nav-links">
''']

    # Navigation links
    for cat in sorted(by_category.keys()):
        cat_display = cat.replace('_', ' ')
        html_parts.append(f'            <a class="nav-link" href="#{cat}">{cat_display}</a>\n')

    # Add failed link if there are failed methods
    if failed_info:
        html_parts.append(f'            <a class="nav-link" href="#failed" style="background:#ffcccc;color:#c0392b;">Failed ({len(failed_info)})</a>\n')

    html_parts.append('        </div>\n    </nav>\n')

    # Category sections
    for cat in sorted(by_category.keys()):
        cat_display = cat.replace('_', ' ')
        html_parts.append(f'''
    <div class="category" id="{cat}">
        <div class="category-header">{cat_display}</div>
        <div class="category-content">
''')

        for method in by_category[cat]:
            # Method card
            html_parts.append(f'''
            <div class="method-card">
                <div class="method-header">
                    <h3 class="method-name">{method.display_name}</h3>
                    <p class="method-desc">{method.description}</p>
                    <div class="method-meta">
                        <span class="meta-badge">{method.name}</span>
                        <span class="meta-badge input">{method.input_type.value}</span>
                    </div>
                </div>
''')

            # Check for outputs
            method_output = output_base / method.name
            has_images = False

            if method_output.exists():
                images = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.gif']:
                    images.extend(method_output.glob(ext))
                    images.extend(method_output.glob(f'**/{ext}'))

                if images:
                    has_images = True
                    html_parts.append('                <div class="image-grid">\n')

                    # Limit to first 6 images
                    for img_path in sorted(set(images))[:6]:
                        data_uri = _encode_image(img_path)
                        if data_uri:
                            label = img_path.stem
                            html_parts.append(f'''                    <div class="image-cell">
                        <img src="{data_uri}" alt="{label}">
                        <div class="image-label">{label}</div>
                    </div>
''')

                    html_parts.append('                </div>\n')

            if not has_images:
                html_parts.append('                <div class="no-output">No outputs yet. Run the method to see results.</div>\n')

            html_parts.append('            </div>\n')

        html_parts.append('        </div>\n    </div>\n')

    # Add failed methods section if any exist
    if failed_info:
        html_parts.append('''
    <div class="category" id="failed">
        <div class="category-header" style="background: #e74c3c;">Failed Validation (Not Shown in Gallery)</div>
        <div class="category-content">
            <p style="color: #666; margin-bottom: 20px;">
                These methods failed validation and need debugging before they can be included in the gallery.
                Run <code>python -m ipol_runner test &lt;method&gt;</code> to debug.
            </p>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Method</th>
                    <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Error</th>
                    <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Notes</th>
                </tr>
''')
        for method_name, info in sorted(failed_info.items()):
            error = info.get('error', 'Unknown error')
            notes = info.get('notes', '')
            html_parts.append(f'''                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>{method_name}</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; color: #c0392b;">{error}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; color: #666;">{notes}</td>
                </tr>
''')
        html_parts.append('''            </table>
        </div>
    </div>
''')

    html_parts.append('''
</body>
</html>
''')

    return ''.join(html_parts)


def generate_comparison_html(
    method_names: List[str],
    output_base: Path,
    title: str = "Method Comparison"
) -> str:
    """Generate HTML for side-by-side comparison of specific methods.

    Args:
        method_names: List of method names to compare
        output_base: Base directory containing method output folders
        title: Page title

    Returns:
        HTML string
    """
    html_parts = [f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat({len(method_names)}, 1fr);
            gap: 20px;
        }}
        .method-column {{
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
        }}
        .column-header {{
            background: #0f3460;
            padding: 15px;
            text-align: center;
            font-weight: 600;
        }}
        .column-content {{
            padding: 15px;
        }}
        .output-image {{
            width: 100%;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .image-label {{
            text-align: center;
            font-size: 0.9em;
            color: #aaa;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="comparison-grid">
''']

    for name in method_names:
        method = get_method(name)
        display = method.display_name if method else name

        html_parts.append(f'''        <div class="method-column">
            <div class="column-header">{display}</div>
            <div class="column-content">
''')

        method_output = output_base / name
        if method_output.exists():
            images = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images.extend(method_output.glob(ext))

            for img_path in sorted(images)[:4]:
                data_uri = _encode_image(img_path)
                if data_uri:
                    html_parts.append(f'''                <img class="output-image" src="{data_uri}" alt="{img_path.stem}">
                <div class="image-label">{img_path.stem}</div>
''')
        else:
            html_parts.append('                <p style="text-align:center;color:#666;">No outputs</p>\n')

        html_parts.append('            </div>\n        </div>\n')

    html_parts.append('''    </div>
</body>
</html>
''')

    return ''.join(html_parts)


def save_gallery(output_dir: Path, output_base: Optional[Path] = None) -> Path:
    """Generate and save gallery HTML.

    Args:
        output_dir: Directory to save gallery.html
        output_base: Base directory with method outputs (defaults to output_dir)

    Returns:
        Path to generated HTML file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_base is None:
        output_base = output_dir

    html = generate_gallery_html(output_base)
    gallery_path = output_dir / "gallery.html"
    gallery_path.write_text(html)
    return gallery_path
