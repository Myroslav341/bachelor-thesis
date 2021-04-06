import json
import logging

from lib import get_dist_between_dots
from rows_manager import RowsManager


def start_project():
    logging.basicConfig(level=logging.INFO)


def save_rows_data():
    rows_manager = RowsManager()

    with open('rows_data.json', 'r') as openfile:
        existing_info = json.load(openfile)

    record_id = len(existing_info["separate_digits"])

    new_record = {"id": record_id, "rows": []}
    for i, row in enumerate(rows_manager.rows):
        row_data = {"number": i, "digits": []}

        for j, cell in enumerate(row.voronoi_cells):
            row_data["digits"].append(
                {
                    "digit_width": cell.digit_width,
                    "relative_width": cell.relative_width,
                    "dist_to_next": (
                        get_dist_between_dots(cell.center, row.voronoi_cells[j + 1].center)
                        if not j == len(row.voronoi_cells) - 1
                        else None
                    ),
                    "connected_with_next": bool(cell.come_with)
                }
            )

        new_record["rows"].append(row_data)

    existing_info["separate_digits"].append(new_record)

    json_object = json.dumps(existing_info, indent=4)

    # Writing to sample.json
    with open("rows_data.json", "w") as outfile:
        outfile.write(json_object)
