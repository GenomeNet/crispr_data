import os
import argparse
import logging
from collections import defaultdict

def parse_gff_file(gff_path):
    """
    Parse a GFF file and yield each entry as a tuple of fields.
    Also returns FASTA data if present.

    Args:
        gff_path (str): Path to the GFF file.

    Yields:
        tuple: A tuple containing the 9 GFF fields or FASTA data.
    """
    with open(gff_path, 'r') as f:
        in_fasta = False
        header_written = False
        for line in f:
            line = line.strip()
            if not header_written and not line.startswith('#'):
                yield ("##gff-version 3",)
                header_written = True
            if line == "##FASTA":
                in_fasta = True
                yield (line,)
            elif in_fasta:
                yield (line,)
            elif not line or line.startswith('#'):
                if line:  # Only yield non-empty header lines
                    yield (line,)
            else:
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
    Combine pairs of GFF files from gff_folders and extra_gff_folders.

    Args:
        gff_folders (list): List of folders containing main GFF files.
        extra_gff_folders (list): List of folders containing extra GFF files.
        output_folder (str): Directory to save the combined GFF files.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Create a dictionary to store extra GFF file paths
    extra_gff_files = defaultdict(list)
    for folder in extra_gff_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.gff') or filename.endswith('.gff3'):
                extra_gff_files[filename].append(os.path.join(folder, filename))

    # Process each main GFF file
    for folder in gff_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.gff') or filename.endswith('.gff3'):
                main_gff_path = os.path.join(folder, filename)
                extra_gff_paths = extra_gff_files.get(filename, [])

                output_gff_path = os.path.join(output_folder, filename)
                logging.info(f"Processing main file: {main_gff_path}")
                logging.info(f"Extra files to combine: {extra_gff_paths}")
                logging.info(f"Generating output file: {output_gff_path}")

                seen_entries = set()
                duplicate_count = 0
                total_entries = 0
                fasta_data = []
                fasta_written = False
                header_written = False

                # Function to process a single GFF file
                def process_gff_file(gff_path):
                    nonlocal duplicate_count, total_entries, fasta_written, header_written
                    is_crt_gff = 'crt_gff' in gff_path
                    for entry in parse_gff_file(gff_path):
                        if len(entry) == 1:  # Header or FASTA data
                            if entry[0] == "##FASTA":
                                if not fasta_written:
                                    fasta_data.append(entry[0])
                            elif entry[0].startswith('#'):
                                if not header_written:  # Only write headers once
                                    if entry[0] != "##gff-version 3" or not seen_entries:
                                        out_gff.write(entry[0] + '\n')
                                        if entry[0] == "##gff-version 3":
                                            header_written = True
                            else:
                                fasta_data.append(entry[0])
                        else:
                            if is_crt_gff and len(entry) == 9:
                                # Replace 'note=' with 'ID=' for crt_gff files
                                attributes = entry[8].split(';')
                                new_attributes = []
                                for attr in attributes:
                                    if attr.startswith('note='):
                                        new_attributes.append('ID=' + attr[5:])
                                    else:
                                        new_attributes.append(attr)
                                entry = list(entry)
                                entry[8] = ';'.join(new_attributes)
                                entry = tuple(entry)

                            key = generate_unique_key(entry)
                            if key not in seen_entries:
                                seen_entries.add(key)
                                out_gff.write('\t'.join(entry) + '\n')
                                total_entries += 1
                            else:
                                duplicate_count += 1

                with open(output_gff_path, 'w') as out_gff:
                    # Process main GFF file
                    process_gff_file(main_gff_path)

                    # Process extra GFF files
                    for extra_gff_path in extra_gff_paths:
                        process_gff_file(extra_gff_path)

                    # Write FASTA data if present
                    if fasta_data:
                        out_gff.write("##FASTA\n")
                        for line in fasta_data:
                            if line != "##FASTA":  # Avoid writing duplicate ##FASTA lines
                                out_gff.write(line + "\n")
                        fasta_written = True

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