import os
import argparse
import logging
from collections import defaultdict

def parse_gff_file(gff_path):
    """
    Parse a GFF file and yield each entry as a tuple of fields.

    Args:
        gff_path (str): Path to the GFF file.

    Yields:
        tuple: A tuple containing the 9 GFF fields.
    """
    with open(gff_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip headers and empty lines
            parts = line.split('\t')
            if len(parts) != 9:
                logging.warning(f"Invalid GFF line in {gff_path}: {line}")
                continue
            yield tuple(parts)

def generate_unique_key(entry):
    """
    Generate a unique key for a GFF entry to identify duplicates.

    Args:
        entry (tuple): A tuple containing the 9 GFF fields.

    Returns:
        tuple: A tuple representing the unique key.
    """
    # Unique key based on all GFF fields except attributes
    return (entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7], entry[8])

def combine_gff(gff_folders, extra_gff_folders, output_folder):
    """
    Combine multiple GFF files into a single GFF file, checking for duplicates.

    Args:
        gff_folders (list): List of folders containing main GFF files.
        extra_gff_folders (list): List of folders containing extra GFF files.
        output_folder (str): Directory to save the combined GFF file.
    """
    os.makedirs(output_folder, exist_ok=True)
    combined_gff_path = os.path.join(output_folder, "combined.gff")

    seen_entries = set()
    duplicate_count = 0
    total_entries = 0

    logging.info(f"Combining GFF files into {combined_gff_path}")

    with open(combined_gff_path, 'w') as out_gff:
        # Write GFF3 header
        out_gff.write("##gff-version 3\n")

        # Function to process folders
        def process_folders(folders, description):
            nonlocal duplicate_count, total_entries
            for folder in folders:
                if not os.path.isdir(folder):
                    logging.warning(f"Folder not found: {folder}")
                    continue
                for filename in os.listdir(folder):
                    if filename.endswith('.gff') or filename.endswith('.gff3'):
                        gff_path = os.path.join(folder, filename)
                        logging.info(f"Processing {description} GFF file: {gff_path}")
                        for entry in parse_gff_file(gff_path):
                            key = generate_unique_key(entry)
                            if key in seen_entries:
                                duplicate_count += 1
                                logging.warning(f"Duplicate entry found in {gff_path}: {entry}")
                                continue
                            seen_entries.add(key)
                            out_gff.write('\t'.join(entry) + '\n')
                            total_entries += 1

        # Process main GFF folders
        process_folders(gff_folders, "main")

        # Process extra GFF folders
        process_folders(extra_gff_folders, "extra")

    logging.info(f"Total entries combined: {total_entries}")
    if duplicate_count > 0:
        logging.warning(f"Total duplicate entries skipped: {duplicate_count}")
    else:
        logging.info("No duplicate entries found.")

def main():
    parser = argparse.ArgumentParser(description="Combine multiple GFF files into one, checking for duplicates.")
    parser.add_argument(
        "--gff_folders",
        nargs='+',
        required=True,
        help="Input folders containing main GFF files to combine."
    )
    parser.add_argument(
        "--extra_gff_folders",
        nargs='*',
        default=[],
        help="Additional folders containing extra GFF files to include."
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Output folder to save the combined GFF file."
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default is INFO."
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log),
        format='%(levelname)s: %(message)s'
    )

    combine_gff(args.gff_folders, args.extra_gff_folders, args.output_folder)

    logging.info("GFF files have been successfully combined.")

if __name__ == "__main__":
    main()