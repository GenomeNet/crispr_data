import os
import json
import argparse
from collections import Counter

def json_to_gff(json_data, seqid):
    """
    Convert CRISPRCasFinder JSON data to GFF3 formatted lines.

    Args:
        json_data (dict): Parsed JSON data from CRISPRCasFinder.
        seqid (str): Sequence identifier for the GFF3 entries.

    Returns:
        list: A list of GFF3 formatted strings.
    """
    gff_lines = []
    crisprs = json_data.get('Crisprs', [])

    # Add the "gnl|X|" prefix to the seqid
    seqid_with_prefix = f"gnl|X|{seqid}"

    for idx, crispr in enumerate(crisprs, 1):
        start = crispr.get('Start')
        end = crispr.get('End')
        dr_consensus = crispr.get('DR_Consensus', '')
        num_repeats = crispr.get('Spacers', 0)  # Assuming 'Spacers' indicates number of repeats
        potential_orientation = crispr.get('Potential_Orientation', 'unknown')
        evidence_level = crispr.get('Evidence_Level', 'NA')

        repeat_seq = dr_consensus  # Using DR_Consensus as the repeat sequence

        attributes = (
            f"ID=CRISPR{idx};"
            f"num_repeats={num_repeats};"
            f"repeat_seq={repeat_seq};"
            f"orientation={potential_orientation};"
            f"evidence_level={evidence_level}"
        )

        gff_line = (
            f"{seqid_with_prefix}\tCRISPRCasFinder\trepeat_region\t{start}\t{end}\t.\t.\t.\t{attributes}"
        )
        gff_lines.append(gff_line)

    return gff_lines

def process_json_file(json_file, output_folder):
    """
    Process a single JSON file and convert it to a GFF3 file.

    Args:
        json_file (str): Path to the JSON file.
        output_folder (str): Directory to save the GFF3 file.
    """
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    gff_file = os.path.join(output_folder, f"{base_name}.gff")

    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {json_file}: {e}")
            return

    sequences = data.get('Sequences', [])
    if not sequences:
        print(f"No sequences found in {json_file}.")
        return

    with open(gff_file, 'w') as out:
        # Write GFF3 header
        out.write("##gff-version 3\n")
        for sequence in sequences:
            seqid = sequence.get('Id', 'unknown_seqid')
            crisprs = sequence.get('Crisprs', [])
            if not crisprs:
                continue
            gff_entries = json_to_gff({'Crisprs': crisprs}, seqid)
            for line in gff_entries:
                out.write(line + "\n")

    print(f"GFF file created: {gff_file}")

def main():
    parser = argparse.ArgumentParser(description="Reformat CRISPRCasFinder JSON output to GFF3 format.")
    parser.add_argument(
        "--input_folder",
        default="datasets/output_json",
        help="Input folder containing CRISPRCasFinder JSON files."
    )
    parser.add_argument(
        "--output_folder",
        default="gff_files_cplus",
        help="Output folder to save the reformatted GFF3 files."
    )

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    json_files = [
        os.path.join(args.input_folder, fname)
        for fname in os.listdir(args.input_folder)
        if fname.endswith('.json')
    ]

    if not json_files:
        print(f"No JSON files found in {args.input_folder}.")
        return

    for json_file in json_files:
        process_json_file(json_file, args.output_folder)

if __name__ == "__main__":
    main()