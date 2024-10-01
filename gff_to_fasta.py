import argparse
import os

def extract_fasta_from_gff(input_file, output_folder):
    # Get the base name of the input file
    base_name = os.path.basename(input_file)
    # Create the output file name
    output_file = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}.fasta")

    fasta_content = []
    in_fasta_section = False

    with open(input_file, 'r') as gff_file:
        for line in gff_file:
            if line.startswith('##FASTA'):
                in_fasta_section = True
                continue
            if in_fasta_section:
                fasta_content.append(line)

    if fasta_content:
        with open(output_file, 'w') as fasta_file:
            fasta_file.writelines(fasta_content)
        print(f"FASTA content extracted and saved to: {output_file}")
    else:
        print(f"No FASTA content found in {input_file}")

def process_gff_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.gff'):
            input_file = os.path.join(input_folder, filename)
            extract_fasta_from_gff(input_file, output_folder)

def main():
    parser = argparse.ArgumentParser(description="Extract FASTA content from GFF3 files in a folder")
    parser.add_argument("--input_folder", required=True, help="Input folder containing GFF3 files")
    parser.add_argument("--output_folder", default=".", help="Output folder for FASTA files")

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    process_gff_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()